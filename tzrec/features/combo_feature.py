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
    CROSS_NEG_DATA_GROUP,
    ParsedData,
    SparseData,
)
from tzrec.features.feature import (
    MAX_HASH_BUCKET_SIZE,
    FgMode,
    _parse_fg_encoded_sparse_feature_impl,
)
from tzrec.features.id_feature import IdFeature
from tzrec.protos.feature_pb2 import FeatureConfig


class ComboFeature(IdFeature):
    """ComboFeature class.

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

    # pyre-ignore [56]
    @IdFeature.is_neg.setter
    def is_neg(self, value: bool) -> None:
        """Feature is negative sampled or not."""
        self._is_neg = value
        self._data_group = CROSS_NEG_DATA_GROUP

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        if self.config.HasField("zch"):
            num_embeddings = self.config.zch.zch_size
        elif self.config.HasField("hash_bucket_size"):
            num_embeddings = self.config.hash_bucket_size
        elif len(self.vocab_list) > 0:
            num_embeddings = len(self.vocab_list)
        elif len(self.vocab_dict) > 0:
            num_embeddings = max(list(self.vocab_dict.values())) + 1
        else:
            raise ValueError(
                f"{self.__class__.__name__}[{self.name}] must set hash_bucket_size"
                " or vocab_list or vocab_dict or zch.zch_size"
            )
        return num_embeddings

    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse input data for the feature impl.

        Args:
            input_data (dict): raw input feature data.

        Return:
            parsed feature data.
        """
        if self.fg_mode == FgMode.FG_NONE:
            # input feature is already bucktized
            feat = input_data[self.name]
            parsed_feat = _parse_fg_encoded_sparse_feature_impl(
                self.name, feat, **self._fg_encoded_kwargs
            )
        elif self.fg_mode == FgMode.FG_NORMAL:
            input_feats = []
            for name in self.inputs:
                x = input_data[name]
                if pa.types.is_list(x.type):
                    x = x.fill_null([])
                input_feats.append(x.tolist())
            values, lengths = self._fg_op.to_bucketized_jagged_tensor(input_feats)
            parsed_feat = SparseData(name=self.name, values=values, lengths=lengths)
        else:
            raise ValueError(
                "fg_mode: {self.fg_mode} is not supported without fg handler."
            )
        return parsed_feat

    def _build_side_inputs(self) -> List[Tuple[str, str]]:
        """Input field names with side."""
        return [tuple(x.split(":")) for x in self.config.expression]

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        fg_cfg = {
            "feature_type": "combo_feature",
            "feature_name": self.name,
            "default_value": self.config.default_value,
            "expression": list(self.config.expression),
            "value_type": "string",
            "need_prefix": False,
            "value_dim": self.config.value_dim,
        }
        if self.config.separator != "\x1d":
            fg_cfg["separator"] = self.config.separator
        if self.config.HasField("zch"):
            fg_cfg["hash_bucket_size"] = MAX_HASH_BUCKET_SIZE
        elif self.config.HasField("hash_bucket_size"):
            fg_cfg["hash_bucket_size"] = self.config.hash_bucket_size
        elif len(self.vocab_list) > 0:
            fg_cfg["vocab_list"] = self.vocab_list
            fg_cfg["default_bucketize_value"] = self.default_bucketize_value
        elif len(self.vocab_dict) > 0:
            fg_cfg["vocab_dict"] = self.vocab_dict
            fg_cfg["default_bucketize_value"] = self.default_bucketize_value
        return [fg_cfg]
