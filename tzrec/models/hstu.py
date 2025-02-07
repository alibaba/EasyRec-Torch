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

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel, MatchTower
from tzrec.modules.sequence import HSTUEncoder
from tzrec.protos import model_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils import config_util


@torch.fx.wrap
def _update_dict_tensor(
    tensor_dict: Dict[str, torch.Tensor],
    new_tensor_dict: Optional[Dict[str, Optional[torch.Tensor]]],
) -> None:
    if new_tensor_dict:
        for k, v in new_tensor_dict.items():
            if v is not None:
                tensor_dict[k] = v


class HSTUMatchTower(MatchTower):
    """HSTU Match model user/item tower.

    Args:
        tower_config (Tower): user/item tower config.
        output_dim (int): user/item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        feature_group (FeatureGroupConfig): feature group config.
        features (list): list of features.
    """

    def __init__(
        self,
        tower_config: tower_pb2.Tower,
        output_dim: int,
        similarity: match_model_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        features: List[BaseFeature],
        model_config: model_pb2.ModelConfig,
    ) -> None:
        super().__init__(
            tower_config, output_dim, similarity, feature_group, features, model_config
        )
        self.init_input()
        self.tower_config = tower_config
        if "user" in self.tower_config.input:
            encoder_config = tower_config.hstu_encoder
            seq_config_dict = config_util.config_to_kwargs(encoder_config)
            sequence_dim = self.embedding_group.group_total_dim(
                f"{self.tower_config.input}.sequence"
            )
            seq_config_dict["sequence_dim"] = sequence_dim
            self.seq_encoder = HSTUEncoder(**seq_config_dict)

    def forward(self, batch: Batch, is_train: bool = False) -> torch.Tensor:
        """Forward the tower.

        Args:
            batch: Input batch containing the data to process
            is_train: Boolean flag indicating whether the model is in training mode

        Returns:
            torch.Tensor: The output tensor from the tower
        """
        grouped_features = self.build_input(batch)
        if "user" in self.tower_config.input:
            if is_train:
                output = self.seq_encoder(grouped_features, is_train=True)
            else:
                output = self.seq_encoder(grouped_features, is_train=False)
        else:
            output = grouped_features[self._group_name]

        if self.tower_config.input == "item":
            output = F.normalize(output, p=2.0, dim=1, eps=1e-6)

        return output


class HSTUMatch(MatchModel):
    """HSTU Match model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
    """

    def __init__(
        self,
        model_config: model_pb2.ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}

        user_group = name_to_feature_group[self._model_config.user_tower.input]
        item_group = name_to_feature_group[self._model_config.item_tower.input]

        name_to_feature = {x.name: x for x in features}
        user_features = OrderedDict(
            [(x, name_to_feature[x]) for x in user_group.feature_names]
        )
        for sequence_group in user_group.sequence_groups:
            for x in sequence_group.feature_names:
                user_features[x] = name_to_feature[x]
        item_features = [name_to_feature[x] for x in item_group.feature_names]

        self.user_tower = HSTUMatchTower(
            self._model_config.user_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            user_group,
            list(user_features.values()),
            model_config,
        )

        self.item_tower = HSTUMatchTower(
            self._model_config.item_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            item_group,
            item_features,
            model_config,
        )

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        item_tower_emb = self.item_tower(batch)
        user_tower_emb = self.user_tower(batch, is_train=self.training)
        _update_dict_tensor(
            self._loss_collection, self.user_tower.group_variational_dropout_loss
        )
        _update_dict_tensor(
            self._loss_collection, self.item_tower.group_variational_dropout_loss
        )
        ui_sim = (
            self.sim(user_tower_emb, item_tower_emb, neg_for_each_sample=True)
            / self._model_config.temperature
        )
        return {"similarity": ui_sim}
