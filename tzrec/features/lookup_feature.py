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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from tzrec.datasets.utils import (
    CROSS_NEG_DATA_GROUP,
    DenseData,
    ParsedData,
    SparseData,
)
from tzrec.features.feature import (
    MAX_HASH_BUCKET_SIZE,
    BaseFeature,
    FgMode,
    _parse_fg_encoded_dense_feature_impl,
    _parse_fg_encoded_sparse_feature_impl,
)
from tzrec.protos.feature_pb2 import FeatureConfig


class LookupFeature(BaseFeature):
    """LookupFeature class.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
        fg_mode (FgMode): input data fg mode.
        fg_encoded_multival_sep (str, optional): multival_sep when fg_mode=FG_NONE
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        fg_mode: FgMode = FgMode.FG_NONE,
        fg_encoded_multival_sep: Optional[str] = None,
    ) -> None:
        super().__init__(feature_config, fg_mode, fg_encoded_multival_sep)

    @property
    def name(self) -> str:
        """Feature name."""
        return self.config.feature_name

    # pyre-ignore [56]
    @BaseFeature.is_neg.setter
    def is_neg(self, value: bool) -> None:
        """Feature is negative sampled or not."""
        self._is_neg = value
        self._data_group = CROSS_NEG_DATA_GROUP

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        if self.config.HasField("value_dim"):
            return self.config.value_dim
        elif self._is_sparse:
            return 0
        else:
            return 1

    @property
    def output_dim(self) -> int:
        """Output dimension of the feature after embedding."""
        if self.has_embedding:
            return self.config.embedding_dim
        else:
            return max(self.config.value_dim, 1)

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        if self._is_sparse is None:
            self._is_sparse = (
                self.config.HasField("hash_bucket_size")
                or self.config.HasField("num_buckets")
                or len(self.vocab_list) > 0
                or len(self.vocab_dict) > 0
                or len(self.config.boundaries) > 0
            )
        return self._is_sparse

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        if self.config.HasField("zch"):
            num_embeddings = self.config.zch.zch_size
        elif self.config.HasField("hash_bucket_size"):
            num_embeddings = self.config.hash_bucket_size
        elif self.config.HasField("num_buckets"):
            num_embeddings = self.config.num_buckets
        elif len(self.vocab_list) > 0:
            num_embeddings = len(self.vocab_list)
        elif len(self.vocab_dict) > 0:
            num_embeddings = max(list(self.vocab_dict.values())) + 1
        else:
            num_embeddings = len(self.config.boundaries) + 1
        return num_embeddings

    @property
    def _dense_emb_type(self) -> Optional[str]:
        return self.config.WhichOneof("dense_emb")

    def _build_side_inputs(self) -> List[Tuple[str, str]]:
        """Input field names with side."""
        return [tuple(x.split(":")) for x in [self.config.map, self.config.key]]

    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse input data for the feature impl.

        Args:
            input_data (dict): raw input feature data.

        Return:
            parsed feature data.
        """
        if self.fg_mode == FgMode.FG_NONE:
            # input feature is already lookuped
            feat = input_data[self.name]
            if self.is_sparse:
                parsed_feat = _parse_fg_encoded_sparse_feature_impl(
                    self.name, feat, **self._fg_encoded_kwargs
                )
            else:
                parsed_feat = _parse_fg_encoded_dense_feature_impl(
                    self.name, feat, **self._fg_encoded_kwargs
                )
        elif self.fg_mode == FgMode.FG_NORMAL:
            input_feats = []
            for name in self.inputs:
                x = input_data[name]
                if pa.types.is_list(x.type):
                    x = x.fill_null([])
                elif pa.types.is_map(x.type):
                    x = x.fill_null({})
                input_feats.append(x.tolist())
            if self.config.value_dim > 1:
                fgout, status = self._fg_op.process(dict(zip(self.inputs, input_feats)))
                assert status.ok(), status.message()
                if self.is_sparse:
                    values = np.asarray(fgout[self.name].values, np.int64)
                    lengths = np.asarray(fgout[self.name].lengths, np.int32)
                    parsed_feat = SparseData(
                        name=self.name, values=values, lengths=lengths
                    )
                else:
                    values = fgout[self.name].dense_values
                    parsed_feat = DenseData(name=self.name, values=values)
            else:
                if self.is_sparse:
                    values, lengths = self._fg_op.to_bucketized_jagged_tensor(
                        *input_feats
                    )
                    parsed_feat = SparseData(
                        name=self.name, values=values, lengths=lengths
                    )
                else:
                    values = self._fg_op.transform(*input_feats)
                    parsed_feat = DenseData(name=self.name, values=values)
        else:
            raise ValueError(
                f"fg_mode: {self.fg_mode} is not supported without fg handler."
            )
        return parsed_feat

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        fg_cfg = {
            "feature_type": "lookup_feature",
            "feature_name": self.name,
            "map": self.config.map,
            "key": self.config.key,
            "default_value": self.config.default_value,
            "value_type": "float",
            "needDiscrete": self.config.need_discrete,
            "needKey": self.config.need_key,
            "combiner": self.config.combiner.lower(),
        }
        raw_fg_cfg = None
        if self.config.value_dim > 1:
            fg_cfg["feature_name"] = self.name + "__lookup"
            fg_cfg["default_value"] = ""
            fg_cfg["value_type"] = "string"
            fg_cfg["value_dim"] = 1
            fg_cfg["needDiscrete"] = True
            fg_cfg["combiner"] = ""
            fg_cfg["stub_type"] = True
            raw_fg_cfg = {
                "feature_type": "raw_feature",
                "feature_name": self.name,
                "default_value": self.config.default_value,
                "expression": "feature:" + fg_cfg["feature_name"],
                "separator": self.config.value_separator,
                "value_dim": self.config.value_dim,
                "value_type": "float",
            }
            if self.config.HasField("normalizer"):
                raw_fg_cfg["normalizer"] = self.config.normalizer
            if len(self.config.boundaries) > 0:
                raw_fg_cfg["boundaries"] = list(self.config.boundaries)
        else:
            if self.config.separator != "\x1d":
                fg_cfg["separator"] = self.config.separator
            if self.config.HasField("normalizer"):
                fg_cfg["normalizer"] = self.config.normalizer
            if self.config.HasField("zch"):
                fg_cfg["hash_bucket_size"] = MAX_HASH_BUCKET_SIZE
                fg_cfg["value_type"] = "string"
                fg_cfg["needDiscrete"] = True
            elif self.config.HasField("hash_bucket_size"):
                fg_cfg["hash_bucket_size"] = self.config.hash_bucket_size
                fg_cfg["value_type"] = "string"
                fg_cfg["needDiscrete"] = True
            elif self.config.HasField("num_buckets"):
                fg_cfg["num_buckets"] = self.config.num_buckets
                fg_cfg["value_type"] = "int64"
                fg_cfg["needDiscrete"] = False
                fg_cfg["combiner"] = ""
            elif len(self.vocab_list) > 0:
                fg_cfg["vocab_list"] = self.vocab_list
                fg_cfg["default_bucketize_value"] = self.default_bucketize_value
                fg_cfg["value_type"] = "string"
                fg_cfg["needDiscrete"] = True
            elif len(self.vocab_dict) > 0:
                fg_cfg["vocab_dict"] = self.vocab_dict
                fg_cfg["default_bucketize_value"] = self.default_bucketize_value
                fg_cfg["value_type"] = "string"
                fg_cfg["needDiscrete"] = True
            elif len(self.config.boundaries) > 0:
                fg_cfg["boundaries"] = list(self.config.boundaries)

            if fg_cfg["needDiscrete"]:
                fg_cfg["combiner"] = ""
            if fg_cfg["combiner"] == "":
                fg_cfg["value_dim"] = self.config.value_dim

        fg_cfgs = [fg_cfg]
        if raw_fg_cfg is not None:
            fg_cfgs.append(raw_fg_cfg)
        return fg_cfgs
