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

import pyarrow as pa

from tzrec.datasets.utils import (
    ParsedData,
    SparseData,
)
from tzrec.features.feature import (
    MAX_HASH_BUCKET_SIZE,
    BaseFeature,
    FgMode,
    _parse_fg_encoded_sparse_feature_impl,
)
from tzrec.protos import feature_pb2
from tzrec.protos.feature_pb2 import FeatureConfig


class IdFeature(BaseFeature):
    """IdFeature class.

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

        if isinstance(self.config, feature_pb2.IdFeature) and self.config.HasField(
            "weighted"
        ):
            self._is_weighted = self.config.weighted

    @property
    def name(self) -> str:
        """Feature name."""
        return self.config.feature_name

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        return self.config.value_dim

    @property
    def output_dim(self) -> int:
        """Output dimension of the feature after embedding."""
        return self.config.embedding_dim

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        if self._is_sparse is None:
            self._is_sparse = True
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
            raise ValueError(
                f"{self.__class__.__name__}[{self.name}] must set hash_bucket_size"
                " or num_buckets or vocab_list or vocab_dict or zch.zch_size"
            )
        return num_embeddings

    @property
    def inputs(self) -> List[str]:
        """Input field names."""
        if not self._inputs:
            if self.fg_mode == FgMode.FG_NONE:
                if self.is_weighted:
                    self._inputs = [f"{self.name}__values", f"{self.name}__weights"]
                else:
                    self._inputs = [self.name]
            else:
                self._inputs = [v for _, v in self.side_inputs]
        return self._inputs

    def _build_side_inputs(self) -> List[Tuple[str, str]]:
        """Input field names with side."""
        return [tuple(self.config.expression.split(":"))]

    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse input data for the feature impl.

        Args:
            input_data (dict): raw input feature data.

        Return:
            parsed feature data.
        """
        if self.fg_mode == FgMode.FG_NONE:
            feat = input_data[self.inputs[0]]
            weight = None
            if len(self.inputs) == 2:
                weight = input_data[self.inputs[1]]
            parsed_feat = _parse_fg_encoded_sparse_feature_impl(
                self.name, feat, weight=weight, **self._fg_encoded_kwargs
            )
        elif self.fg_mode == FgMode.FG_NORMAL:
            input_feat = input_data[self.inputs[0]]
            if pa.types.is_list(input_feat.type):
                input_feat = input_feat.fill_null([])
            input_feat = input_feat.tolist()
            if self._is_weighted:
                values, lengths, weights = self._fg_op.to_weighted_jagged_tensor(
                    input_feat
                )
            else:
                values, lengths = self._fg_op.to_bucketized_jagged_tensor(input_feat)
                weights = None
            parsed_feat = SparseData(
                name=self.name, values=values, lengths=lengths, weights=weights
            )
        else:
            raise ValueError(
                f"fg_mode: {self.fg_mode} is not supported without fg handler."
            )
        return parsed_feat

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        fg_cfg = {
            "feature_type": "id_feature",
            "feature_name": self.name,
            "default_value": self.config.default_value,
            "expression": self.config.expression,
            "value_type": "string",
            "need_prefix": False,
        }
        if self.config.separator != "\x1d":
            fg_cfg["separator"] = self.config.separator
        if self.config.HasField("zch"):
            fg_cfg["hash_bucket_size"] = MAX_HASH_BUCKET_SIZE
            fg_cfg["value_type"] = "string"
        elif self.config.HasField("hash_bucket_size"):
            fg_cfg["hash_bucket_size"] = self.config.hash_bucket_size
            fg_cfg["value_type"] = "string"
        elif len(self.vocab_list) > 0:
            fg_cfg["vocab_list"] = self.vocab_list
            fg_cfg["default_bucketize_value"] = self.default_bucketize_value
            fg_cfg["value_type"] = "string"
        elif len(self.vocab_dict) > 0:
            fg_cfg["vocab_dict"] = self.vocab_dict
            fg_cfg["default_bucketize_value"] = self.default_bucketize_value
            fg_cfg["value_type"] = "string"
        elif self.config.HasField("num_buckets"):
            fg_cfg["num_buckets"] = self.config.num_buckets
            if self.config.default_value:
                fg_cfg["value_type"] = "int64"
            else:
                fg_cfg["value_type"] = "string"
        if self.config.weighted:
            fg_cfg["weighted"] = True
        if self.config.HasField("value_dim"):
            fg_cfg["value_dim"] = self.config.value_dim
        else:
            fg_cfg["value_dim"] = 0
        return [fg_cfg]
