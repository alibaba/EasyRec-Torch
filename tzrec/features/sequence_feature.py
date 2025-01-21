# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyfg

from tzrec.datasets.utils import (
    ParsedData,
    SequenceDenseData,
    SequenceSparseData,
)
from tzrec.features.feature import MAX_HASH_BUCKET_SIZE
from tzrec.features.id_feature import FgMode, IdFeature
from tzrec.features.raw_feature import RawFeature
from tzrec.protos import feature_pb2
from tzrec.protos.feature_pb2 import FeatureConfig
from tzrec.utils.logging_util import logger


def _parse_fg_encoded_sequence_sparse_feature_impl(
    name: str,
    feat: pa.Array,
    sequence_delim: str = ";",
    multival_sep: str = chr(3),
    default_value: Optional[List[int]] = None,
) -> SequenceSparseData:
    """Parse fg encoded sequence sparse feature.

    Args:
        name (str): feature name.
        feat (pa.Array): input feature data.
        sequence_delim (str): sequence delimiter.
        multival_sep (str): string separator for multi-val data.
        default_value (int): default value.

    Returns:
        an instance of SequenceSparseFeature.
    """
    if pa.types.is_string(feat.type):
        # dtype = string
        is_empty = pa.compute.equal(feat, pa.scalar(""))
        feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
        if default_value is not None:
            feat = feat.fill_null(multival_sep.join(map(str, default_value)))
        list_seq_feat = pa.compute.split_pattern(feat, sequence_delim)
        list_feat = pa.compute.split_pattern(list_seq_feat.values, multival_sep)
        seq_offsets = list_seq_feat.offsets.to_numpy()
        seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
        feat_values = list_feat.values.cast(pa.int64()).to_numpy()
        feat_offsets = list_feat.offsets.to_numpy()
        feat_lengths = feat_offsets[1:] - feat_offsets[:-1]
    elif pa.types.is_list(feat.type):
        if pa.types.is_list(feat.type.value_type):
            # dtype = list<list<int>> or others can cast to list<list<int>>
            feat = feat.cast(pa.list_(pa.list_(pa.int64())), safe=False)
            if default_value is not None:
                is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
                feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
                feat = feat.fill_null([default_value])
            seq_offsets = feat.offsets.to_numpy()
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            feat_values = feat.values.values.to_numpy()
            feat_offsets = feat.values.offsets.to_numpy()
            feat_lengths = feat_offsets[1:] - feat_offsets[:-1]
        else:
            # dtype = list<int> or others can cast to list<int>
            feat = feat.cast(pa.list_(pa.int64()), safe=False)
            if default_value is not None:
                is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
                feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
                feat = feat.fill_null(default_value)
            seq_offsets = feat.offsets.to_numpy()
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            feat_values = feat.values.to_numpy()
            feat_lengths = np.ones_like(feat_values, dtype=np.int32)
    else:
        raise ValueError(
            f"{name} only support str|list<int>|list<list<int>> dtype input,"
            f" but get {feat.dtype}."
        )
    return SequenceSparseData(name, feat_values, feat_lengths, seq_lengths)


def _parse_fg_encoded_sequence_dense_feature_impl(
    name: str,
    feat: pa.Array,
    sequence_delim: str = ";",
    multival_sep: str = chr(3),
    value_dim: int = 1,
    default_value: Optional[List[float]] = None,
) -> SequenceDenseData:
    """Parse fg encoded sequence dense feature.

    Args:
        name (str): feature name.
        feat (pa.Array): input feature data.
        sequence_delim (str): sequence delimiter.
        multival_sep (str): string separator for multi-val data.
        value_dtype (pa.DataType): value dtype.
        value_dim (int): value dimension.
        default_value (list): default value.

    Returns:
        an instance of SequenceSparseFeature.
    """
    if pa.types.is_string(feat.type):
        is_empty = pa.compute.equal(feat, pa.scalar(""))
        feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
        if default_value is not None:
            feat = feat.fill_null(multival_sep.join(map(str, default_value)))
        list_seq_feat = pa.compute.split_pattern(feat, sequence_delim)
        list_feat = pa.compute.split_pattern(list_seq_feat.values, multival_sep)
        seq_offsets = list_seq_feat.offsets.to_numpy()
        seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
        feat_values = (
            list_feat.values.cast(pa.float32(), safe=False)
            .to_numpy()
            .reshape(-1, value_dim)
        )
    elif pa.types.is_list(feat.type):
        if pa.types.is_list(feat.type.value_type):
            # dtype = list<list<float>> or others can cast to list<list<float>>
            feat = feat.cast(pa.list_(pa.list_(pa.float32())), safe=False)
            if default_value is not None:
                is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
                feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
                feat = feat.fill_null([default_value])
            seq_offsets = feat.offsets.to_numpy()
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            feat_values = feat.values.values.to_numpy().reshape(-1, value_dim)
        else:
            # dtype = list<float> or others can cast to list<float>
            feat = feat.cast(pa.list_(pa.float32()), safe=False)
            if default_value is not None:
                is_empty = pa.compute.equal(pa.compute.list_value_length(feat), 0)
                feat = pa.compute.if_else(is_empty, pa.nulls(len(feat)), feat)
                feat = feat.fill_null(default_value)
            seq_offsets = feat.offsets.to_numpy()
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            feat_values = feat.values.to_numpy().reshape(-1, value_dim)
    else:
        raise ValueError(
            f"{name} only support str|list<float>|list<float<float>> dtype input,"
            f" but get {feat.dtype}."
        )
    return SequenceDenseData(name, feat_values, seq_lengths)


class SequenceIdFeature(IdFeature):
    """SequenceIdFeature class.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
        sequence_name (str): sequence group name.
        sequence_delim (str): separator for sequence feature.
        sequence_length (int): max sequence length.
        sequence_pk (str): sequence primary key name for serving.
        fg_mode (FgMode): input data fg mode.
        fg_encoded_multival_sep (str, optional): multival_sep when fg_mode=FG_NONE
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        sequence_name: Optional[str] = None,
        sequence_delim: Optional[str] = None,
        sequence_length: Optional[int] = None,
        sequence_pk: Optional[str] = None,
        fg_mode: FgMode = FgMode.FG_NONE,
        fg_encoded_multival_sep: Optional[str] = None,
    ) -> None:
        fc_type = feature_config.WhichOneof("feature")
        config = getattr(feature_config, fc_type)
        self._is_grouped_seq = False
        self.sequence_name = None
        self.sequence_delim = None
        self.sequence_length = None
        self.sequence_pk = None
        if isinstance(config, feature_pb2.IdFeature):
            self._is_grouped_seq = True
            self.sequence_name = sequence_name
            self.sequence_delim = sequence_delim
            self.sequence_length = sequence_length
            if not sequence_pk:
                self.sequence_pk = f"user:{sequence_name}"
            else:
                self.sequence_pk = sequence_pk
        else:
            self.sequence_delim = config.sequence_delim
            self.sequence_length = config.sequence_length
        super().__init__(feature_config, fg_mode, fg_encoded_multival_sep)

    @property
    def name(self) -> str:
        """Feature name."""
        if self._is_grouped_seq:
            return f"{self.sequence_name}__{self.config.feature_name}"
        else:
            return self.config.feature_name

    @property
    def is_sequence(self) -> bool:
        """Feature is sequence or not."""
        return True

    @property
    def is_grouped_sequence(self) -> bool:
        """Feature is grouped sequence or not."""
        return self._is_grouped_seq

    def _build_side_inputs(self) -> List[Tuple[str, str]]:
        """Input field names with side."""
        if self._is_grouped_seq:
            side, name = self.config.expression.split(":")
            return [(side, f"{self.sequence_name}__{name}")]
        else:
            return [tuple(self.config.expression.split(":"))]

    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse input data for the feature impl.

        Args:
            input_data (dict): raw input feature data.

        Return:
            parsed feature data.
        """
        if self.fg_mode == FgMode.FG_NONE:
            feat = input_data[self.name]
            parsed_feat = _parse_fg_encoded_sequence_sparse_feature_impl(
                self.name,
                feat,
                sequence_delim=self.sequence_delim,
                **self._fg_encoded_kwargs,
            )
        elif self.fg_mode == FgMode.FG_NORMAL:
            input_feat = input_data[self.inputs[0]]
            if pa.types.is_list(input_feat.type):
                input_feat = input_feat.fill_null([])
            input_feat = input_feat.tolist()
            values, key_lengths, seq_lengths = self._fg_op.to_bucketized_jagged_tensor(
                input_feat
            )
            parsed_feat = SequenceSparseData(
                name=self.name,
                values=values,
                lengths=key_lengths,
                seq_lengths=seq_lengths,
            )
        else:
            raise ValueError(
                f"fg_mode: {self.fg_mode} is not supported without fg handler."
            )
        return parsed_feat

    def init_fg(self) -> None:
        """Init fg op."""
        cfgs = self.fg_json()
        is_rank_zero = os.environ.get("RANK", "0") == "0"
        if self._is_grouped_seq:
            # pyre-ignore [16]
            self._fg_op = pyfg.FeatureFactory.create(
                cfgs[0],
                self.sequence_name,
                self.sequence_delim,
                self.sequence_length,
                is_rank_zero,
            )
        else:
            # pyre-ignore [16]
            self._fg_op = pyfg.FeatureFactory.create(
                cfgs[0],
                is_rank_zero,
            )

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        if self.config.default_value == "":
            logger.warning(
                "SequenceIdFeature not support empty default value now. reset to zero."
            )
            self.config.default_value = "0"
        fg_cfg = {
            "feature_type": "id_feature"
            if self._is_grouped_seq
            else "sequence_id_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.config.default_value,
            "expression": self.config.expression,
            "value_type": "string",
            "need_prefix": False,
        }
        if not self._is_grouped_seq:
            fg_cfg["sequence_delim"] = self.config.sequence_delim
            fg_cfg["sequence_length"] = self.config.sequence_length
        if self.config.separator != "\x1d":
            fg_cfg["separator"] = self.config.separator
        if self.config.HasField("zch"):
            fg_cfg["hash_bucket_size"] = MAX_HASH_BUCKET_SIZE
        elif self.config.HasField("hash_bucket_size"):
            fg_cfg["hash_bucket_size"] = self.config.hash_bucket_size
        elif self.config.HasField("num_buckets"):
            fg_cfg["num_buckets"] = self.config.num_buckets
        elif len(self.vocab_list) > 0:
            fg_cfg["vocab_list"] = self.vocab_list
            fg_cfg["default_bucketize_value"] = self.default_bucketize_value
        elif len(self.config.vocab_dict) > 0:
            fg_cfg["vocab_dict"] = self.vocab_dict
            fg_cfg["default_bucketize_value"] = self.default_bucketize_value
        if self.config.HasField("value_dim"):
            assert (
                self.config.value_dim == 1
            ), f"SequenceIdFeature {self.name} not support multi-value now."
            fg_cfg["value_dim"] = self.config.value_dim
        else:
            fg_cfg["value_dim"] = 1

        return [fg_cfg]


class SequenceRawFeature(RawFeature):
    """SequenceIdFeature class.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
        sequence_name (str): sequence group name.
        sequence_delim (str): separator for sequence feature.
        sequence_length (int): max sequence length.
        sequence_pk (str): sequence primary key name for serving.
        fg_mode (FgMode): input data fg mode.
        fg_encoded_multival_sep (str, optional): multival_sep when fg_mode=FG_NONE
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        sequence_name: Optional[str] = None,
        sequence_delim: Optional[str] = None,
        sequence_length: Optional[int] = None,
        sequence_pk: Optional[str] = None,
        fg_mode: FgMode = FgMode.FG_NONE,
        fg_encoded_multival_sep: Optional[str] = None,
    ) -> None:
        fc_type = feature_config.WhichOneof("feature")
        config = getattr(feature_config, fc_type)
        self._is_grouped_seq = False
        self.sequence_name = None
        self.sequence_delim = None
        self.sequence_length = None
        self.sequence_pk = None
        if isinstance(config, feature_pb2.RawFeature):
            self._is_grouped_seq = True
            self.sequence_name = sequence_name
            self.sequence_delim = sequence_delim
            self.sequence_length = sequence_length
            if not sequence_pk:
                self.sequence_pk = f"user:{sequence_name}"
            else:
                self.sequence_pk = sequence_pk
        else:
            self.sequence_delim = config.sequence_delim
            self.sequence_length = config.sequence_length
        super().__init__(feature_config, fg_mode, fg_encoded_multival_sep)

    @property
    def name(self) -> str:
        """Feature name."""
        if self._is_grouped_seq:
            return f"{self.sequence_name}__{self.config.feature_name}"
        else:
            return self.config.feature_name

    @property
    def is_sequence(self) -> bool:
        """Feature is sequence or not."""
        return True

    @property
    def is_grouped_sequence(self) -> bool:
        """Feature is grouped sequence or not."""
        return self._is_grouped_seq

    @property
    def _dense_emb_type(self) -> Optional[str]:
        # TODO: support dense embedding for sequence raw feature.
        return None

    def _build_side_inputs(self) -> List[Tuple[str, str]]:
        """Input field names with side."""
        if self._is_grouped_seq:
            side, name = self.config.expression.split(":")
            return [(side, f"{self.sequence_name}__{name}")]
        else:
            return [tuple(self.config.expression.split(":"))]

    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse input data for the feature impl.

        Args:
            input_data (dict): raw input feature data.

        Return:
            parsed feature data.
        """
        if self.fg_mode == FgMode.FG_NONE:
            feat = input_data[self.name]
            if self.is_sparse:
                parsed_feat = _parse_fg_encoded_sequence_sparse_feature_impl(
                    self.name,
                    feat,
                    sequence_delim=self.sequence_delim,
                    **self._fg_encoded_kwargs,
                )
            else:
                parsed_feat = _parse_fg_encoded_sequence_dense_feature_impl(
                    self.name,
                    feat,
                    sequence_delim=self.sequence_delim,
                    value_dim=self.config.value_dim,
                    **self._fg_encoded_kwargs,
                )
        elif self.fg_mode == FgMode.FG_NORMAL:
            input_feat = input_data[self.inputs[0]]
            if pa.types.is_list(input_feat.type):
                input_feat = input_feat.fill_null([])
            input_feat = input_feat.tolist()
            if self._fg_op.is_sparse:
                values, lengths = self._fg_op.to_bucketized_jagged_tensor(input_feat)
                parsed_feat = SequenceSparseData(
                    name=self.name,
                    values=values,
                    lengths=np.array([self._fg_op.value_dimension()] * sum(lengths)),
                    seq_lengths=lengths,
                )
            else:
                values, lengths = self._fg_op.to_jagged_tensor(input_feat)
                parsed_feat = SequenceDenseData(
                    name=self.name,
                    values=values,
                    seq_lengths=lengths,
                )
        else:
            raise ValueError(
                f"fg_mode: {self.fg_mode} is not supported without fg handler."
            )
        return parsed_feat

    def init_fg(self) -> None:
        """Init fg op."""
        cfgs = self.fg_json()
        is_rank_zero = os.environ.get("RANK", "0") == "0"
        if self._is_grouped_seq:
            # pyre-ignore [16]
            self._fg_op = pyfg.FeatureFactory.create(
                cfgs[0],
                self.sequence_name,
                self.sequence_delim,
                self.sequence_length,
                is_rank_zero,
            )
        else:
            # pyre-ignore [16]
            self._fg_op = pyfg.FeatureFactory.create(
                cfgs[0],
                is_rank_zero,
            )

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        fg_cfg = {
            "feature_type": "raw_feature"
            if self._is_grouped_seq
            else "sequence_raw_feature",
            "feature_name": self.config.feature_name,
            "default_value": self.config.default_value,
            "expression": self.config.expression,
            "value_type": "float",
        }
        if not self._is_grouped_seq:
            fg_cfg["sequence_delim"] = self.config.sequence_delim
            fg_cfg["sequence_length"] = self.config.sequence_length
        if self.config.value_dim > 1:
            if self.config.separator != "\x1d":
                fg_cfg["separator"] = self.config.separator
            fg_cfg["value_dim"] = self.config.value_dim
        if self.config.normalizer != "":
            fg_cfg["normalizer"] = self.config.normalizer
        if len(self.config.boundaries) > 0:
            fg_cfg["boundaries"] = list(self.config.boundaries)
        return [fg_cfg]
