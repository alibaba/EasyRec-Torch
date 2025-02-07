# Copyright (c) 2024-2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# HSTUEncoder is from generative-recommenders,
# https://github.com/facebookresearch/generative-recommenders,
# thanks to their public work.

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tzrec.modules.hstu import (
    HSTUCacheState,
    RelativeBucketedTimeAndPositionBasedBias,
    SequentialTransductionUnitJagged,
)
from tzrec.modules.mlp import MLP
from tzrec.protos.seq_encoder_pb2 import SeqEncoderConfig
from tzrec.utils import config_util
from tzrec.utils.load_class import get_register_class_meta


@torch.fx.wrap
def _arange(end: int, device: torch.device) -> torch.Tensor:
    return torch.arange(end, device=device)


_SEQ_ENCODER_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_SEQ_ENCODER_CLASS_MAP)


class SequenceEncoder(nn.Module, metaclass=_meta_cls):
    """Base module of sequence encoder."""

    def __init__(self, input: str) -> None:
        super().__init__()
        self._input = input

    def input(self) -> str:
        """Get sequence encoder input group name."""
        return self._input

    def output_dim(self) -> int:
        """Output dimension of the module."""
        raise NotImplementedError


class DINEncoder(SequenceEncoder):
    """DIN sequence encoder.

    Args:
        sequence_dim (int): sequence tensor channel dimension.
        query_dim (int): query tensor channel dimension.
        input(str): input feature group name.
        attn_mlp (dict): target attention MLP module parameters.
    """

    def __init__(
        self,
        sequence_dim: int,
        query_dim: int,
        input: str,
        attn_mlp: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._query_dim = query_dim
        self._sequence_dim = sequence_dim
        if self._query_dim > self._sequence_dim:
            raise ValueError("query_dim > sequence_dim not supported yet.")
        self.mlp = MLP(in_features=sequence_dim * 4, **attn_mlp)
        self.linear = nn.Linear(self.mlp.hidden_units[-1], 1)
        self._query_name = f"{input}.query"
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._sequence_dim

    def forward(self, sequence_embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the module."""
        query = sequence_embedded[self._query_name]
        sequence = sequence_embedded[self._sequence_name]
        sequence_length = sequence_embedded[self._sequence_length_name]
        max_seq_length = sequence.size(1)
        sequence_mask = _arange(
            max_seq_length, device=sequence_length.device
        ).unsqueeze(0) < sequence_length.unsqueeze(1)

        if self._query_dim < self._sequence_dim:
            query = F.pad(query, (0, self._sequence_dim - self._query_dim))
        queries = query.unsqueeze(1).expand(-1, max_seq_length, -1)

        attn_input = torch.cat(
            [queries, sequence, queries - sequence, queries * sequence], dim=-1
        )
        attn_output = self.mlp(attn_input)
        attn_output = self.linear(attn_output)
        attn_output = attn_output.transpose(1, 2)

        padding = torch.ones_like(attn_output) * (-(2**32) + 1)
        scores = torch.where(sequence_mask.unsqueeze(1), attn_output, padding)
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, sequence).squeeze(1)


class SimpleAttention(SequenceEncoder):
    """Simple attention encoder."""

    def __init__(
        self,
        sequence_dim: int,
        query_dim: int,
        input: str,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._sequence_dim = sequence_dim
        self._query_dim = query_dim
        self._query_name = f"{input}.query"
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._sequence_dim

    def forward(self, sequence_embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the module."""
        query = sequence_embedded[self._query_name]
        sequence = sequence_embedded[self._sequence_name]
        sequence_length = sequence_embedded[self._sequence_length_name]
        max_seq_length = sequence.size(1)
        sequence_mask = _arange(max_seq_length, sequence_length.device).unsqueeze(
            0
        ) < sequence_length.unsqueeze(1)

        attn_output = torch.matmul(sequence, query.unsqueeze(2)).squeeze(2)
        padding = torch.ones_like(attn_output) * (-(2**32) + 1)
        scores = torch.where(sequence_mask, attn_output, padding)
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores.unsqueeze(1), sequence).squeeze(1)


class PoolingEncoder(SequenceEncoder):
    """Mean/Sum pooling sequence encoder.

    Args:
        sequence_dim (int): sequence tensor channel dimension.
        input (str): input feature group name.
        pooling_type (str): pooling type, sum or mean.
    """

    def __init__(
        self,
        sequence_dim: int,
        input: str,
        pooling_type: str = "mean",
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._sequence_dim = sequence_dim
        self._pooling_type = pooling_type
        assert self._pooling_type in [
            "sum",
            "mean",
        ], "only sum|mean pooling type supported now."
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._sequence_dim

    def forward(self, sequence_embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the module."""
        sequence = sequence_embedded[self._sequence_name]
        feature = torch.sum(sequence, dim=1)
        if self._pooling_type == "mean":
            sequence_length = sequence_embedded[self._sequence_length_name]
            sequence_length = torch.max(
                sequence_length, torch.ones_like(sequence_length)
            )
            feature = feature / sequence_length.unsqueeze(1)
        return feature


class MultiWindowDINEncoder(SequenceEncoder):
    """Multi Window DIN module.

    Args:
        sequence_dim (int): sequence tensor channel dimension.
        query_dim (int): query tensor channel dimension.
        input(str): input feature group name.
        windows_len (list): time windows len.
        attn_mlp (dict): target attention MLP module parameters.
    """

    def __init__(
        self,
        sequence_dim: int,
        query_dim: int,
        input: str,
        windows_len: List[int],
        attn_mlp: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._query_dim = query_dim
        self._sequence_dim = sequence_dim
        self._windows_len = windows_len
        if self._query_dim > self._sequence_dim:
            raise ValueError("query_dim > sequence_dim not supported yet.")
        self.register_buffer("windows_len", torch.tensor(windows_len))
        self.register_buffer(
            "cumsum_windows_len", torch.tensor(np.cumsum([0] + list(windows_len)[:-1]))
        )
        self._sum_windows_len = sum(windows_len)
        self.mlp = MLP(in_features=sequence_dim * 3, **attn_mlp)
        self.linear = nn.Linear(self.mlp.hidden_units[-1], 1)
        self.active = nn.PReLU()
        self._query_name = f"{input}.query"
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._sequence_dim * (len(self._windows_len) + 1)

    def forward(self, sequence_embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the module."""
        query = sequence_embedded[self._query_name]
        sequence = sequence_embedded[self._sequence_name]
        sequence_length = sequence_embedded[self._sequence_length_name]
        max_seq_length = sequence.size(1)
        sequence_mask = _arange(
            max_seq_length, device=sequence_length.device
        ).unsqueeze(0) < sequence_length.unsqueeze(1)

        if self._query_dim < self._sequence_dim:
            query = F.pad(query, (0, self._sequence_dim - self._query_dim))
        queries = query.unsqueeze(1).expand(-1, max_seq_length, -1)  # [B, T, C]

        attn_input = torch.cat([sequence, queries * sequence, queries], dim=-1)
        attn_output = self.mlp(attn_input)
        attn_output = self.linear(attn_output)
        attn_output = self.active(attn_output)  # [B, T, 1]

        att_sequences = attn_output * sequence_mask.unsqueeze(2) * sequence

        pad = (0, 0, 0, self._sum_windows_len - max_seq_length)
        pad_att_sequences = F.pad(att_sequences, pad).transpose(0, 1)
        result = torch.segment_reduce(
            pad_att_sequences, reduce="sum", lengths=self.windows_len, axis=0
        ).transpose(0, 1)  # [B, L, C]

        segment_length = torch.min(
            sequence_length.unsqueeze(1) - self.cumsum_windows_len.unsqueeze(0),
            self.windows_len,
        )
        result = result / torch.max(
            segment_length, torch.ones_like(segment_length)
        ).unsqueeze(2)

        return torch.cat([result, query.unsqueeze(1)], dim=1).reshape(
            result.shape[0], -1
        )  # [B, (L+1)*C]


class HSTUEncoder(SequenceEncoder):
    """HSTU sequence encoder.

    Args:
        sequence_dim (int): sequence tensor channel dimension.
        query_dim (int): query tensor channel dimension.
        input(str): input feature group name.
        attn_mlp (dict): target attention MLP module parameters.
    """

    def __init__(
        self,
        sequence_dim: int,
        attn_dim: int,
        linear_dim: int,
        input: str,
        max_seq_length: int,
        pos_dropout_rate: float = 0.2,
        linear_dropout_rate: float = 0.2,
        attn_dropout_rate: float = 0.0,
        normalization: str = "rel_bias",
        linear_activation: str = "silu",
        linear_config: str = "uvqk",
        num_heads: int = 1,
        num_blocks: int = 2,
        max_output_len: int = 10,
        time_bucket_size: int = 128,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__(input)
        self._sequence_dim = sequence_dim
        self._attn_dim = attn_dim
        self._linear_dim = linear_dim
        self._max_seq_length = max_seq_length
        self._query_name = f"{input}.query"
        self._sequence_name = f"{input}.sequence"
        self._sequence_length_name = f"{input}.sequence_length"
        max_output_len = max_output_len + 1  # for target
        self.position_embed = nn.Embedding(
            self._max_seq_length + max_output_len, self._sequence_dim, padding_idx=0
        )
        self.dropout_rate = pos_dropout_rate
        self.enable_relative_attention_bias = True
        self.autocast_dtype = None
        self._attention_layers: nn.ModuleList = nn.ModuleList(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._sequence_dim,
                    linear_hidden_dim=self._linear_dim,
                    attention_dim=self._attn_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_seq_length + max_output_len,
                            num_buckets=time_bucket_size,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if self.enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=False,
                )
                for _ in range(num_blocks)
            ]
        )
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_seq_length + max_output_len,
                        self._max_seq_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._autocast_dtype = None

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._sequence_dim

    def forward(
        self, sequence_embedded: Dict[str, torch.Tensor], is_train: bool = False
    ) -> torch.Tensor:
        """Forward the module."""
        sequence = sequence_embedded[self._sequence_name]  # B, N, E
        sequence_length = sequence_embedded[self._sequence_length_name]  # N
        # max_seq_length = sequence.size(1)
        float_dtype = sequence.dtype

        # Add positional embeddings and apply dropout
        positions = (
            _arange(sequence.size(1), device=sequence.device)
            .unsqueeze(0)
            .expand(sequence.size(0), -1)
        )
        sequence = sequence * (self._sequence_dim**0.5) + self.position_embed(positions)
        sequence = F.dropout(sequence, p=self.dropout_rate, training=self.training)
        sequence_mask = _arange(
            sequence.size(1), device=sequence_length.device
        ).unsqueeze(0) < sequence_length.unsqueeze(1)
        sequence = sequence * sequence_mask.unsqueeze(-1).to(float_dtype)

        invalid_attn_mask = 1.0 - self._attn_mask.to(float_dtype)
        sequence_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            sequence_length
        )
        sequence = torch.ops.fbgemm.dense_to_jagged(sequence, [sequence_offsets])[0]

        all_timestamps = None
        jagged_x, cache_states = self.jagged_forward(
            x=sequence,
            x_offsets=sequence_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            delta_x_offsets=None,
            cache=None,
            return_cache_states=False,
        )
        # output_embeddings = torch.ops.fbgemm.jagged_to_padded_dense(
        #     values=jagged_x,
        #     offsets=[sequence_offsets],
        #     max_lengths=[invalid_attn_mask.size(1)],
        #     padding_value=0.0,
        # )
        # post processing: L2 Normalization
        output_embeddings = jagged_x
        output_embeddings = output_embeddings[..., : self._sequence_dim]
        output_embeddings = output_embeddings / torch.clamp(
            torch.linalg.norm(output_embeddings, ord=None, dim=-1, keepdim=True),
            min=1e-6,
        )
        if not is_train:
            output_embeddings = self.get_current_embeddings(
                sequence_length, output_embeddings
            )
        return output_embeddings

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        r"""Jagged forward.

        Args:
            x: (\sum_i N_i, D) x float
            x_offsets: (B + 1) x int32
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}
            delta_x_offsets: offsets for x
            cache: cache contents
            return_cache_states: bool. True if we should return cache states.

        Returns:
            x' = f(x), (\sum_i N_i, D) x float
        """
        cache_states: List[HSTUCacheState] = []

        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                x, cache_states_i = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    delta_x_offsets=delta_x_offsets,
                    cache=cache[i] if cache is not None else None,
                    return_cache_states=return_cache_states,
                )
                if return_cache_states:
                    cache_states.append(cache_states_i)

        return x, cache_states

    def get_current_embeddings(
        self,
        lengths: torch.Tensor,
        encoded_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Get the embeddings of the last past_id as the current embeds.

        Args:
            lengths: (B,) x int
            encoded_embeddings: (B, N, D,) x float

        Returns:
            (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
        """
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        indices = offsets[:-1] + lengths - 1
        # B, N, D = encoded_embeddings.size()
        # flattened_offsets = (lengths - 1) + _arange(B, device=lengths.device) * N
        # return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)
        return encoded_embeddings[indices]


def create_seq_encoder(
    seq_encoder_config: SeqEncoderConfig, group_total_dim: Dict[str, int]
) -> SequenceEncoder:
    """Build seq encoder model..

    Args:
        seq_encoder_config:  a SeqEncoderConfig.group_total_dim.
        group_total_dim: a dict contain all seq group dim info.

    Return:
        model: a SequenceEncoder cls.
    """
    model_cls_name = config_util.which_msg(seq_encoder_config, "seq_module")
    # pyre-ignore [16]
    model_cls = SequenceEncoder.create_class(model_cls_name)
    seq_type = seq_encoder_config.WhichOneof("seq_module")
    seq_type_config = getattr(seq_encoder_config, seq_type)
    input_name = seq_type_config.input
    query_dim = group_total_dim[f"{input_name}.query"]
    sequence_dim = group_total_dim[f"{input_name}.sequence"]
    seq_config_dict = config_util.config_to_kwargs(seq_type_config)
    seq_config_dict["sequence_dim"] = sequence_dim
    seq_config_dict["query_dim"] = query_dim
    seq_encoder = model_cls(**seq_config_dict)
    return seq_encoder
