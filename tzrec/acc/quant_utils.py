# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import torch
import torch.nn as nn
import torch.quantization as quant
from torch import Tensor
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    EmbeddingBagConfig,
    QuantConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
)
from torchrec.modules.mc_modules import ManagedCollisionCollection
from torchrec.quant.embedding_modules import (
    DEFAULT_ROW_ALIGNMENT,
    MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    MODULE_ATTR_REGISTER_TBES_BOOL,
    MODULE_ATTR_ROW_ALIGNMENT_INT,
    _update_embedding_configs,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


def _get_device(module: nn.Module) -> torch.device:
    device = torch.device("cpu")

    for _, tensor in module.state_dict().items():
        device = tensor.device
        break
    return device


class QuantManagedCollisionEmbeddingBagCollection(QuantEmbeddingBagCollection):
    """QuantManagedCollisionEmbeddingBagCollection represents a quantized EBC module.

    The inputs into the MC-EC/EBC will first be modified by the managed collision module
    before being passed into the embedding collection.

    Args:
        tables (List[EmbeddingBagConfig]): A list of EmbeddingBagConfig
            objects representing the embedding tables in the collection.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        device (torch.device): The device on which the embedding bag collection will
            be allocated.
        output_dtype (torch.dtype, optional): The data type of the output embeddings.
            Defaults to torch.float.
        table_name_to_quantized_weights (Dict[str, Tuple[Tensor, Tensor]], optional):
            A dictionary mapping table names to their corresponding quantized weights.
            Defaults to None.
        register_tbes (bool, optional): Whether to register the TBEs in the model.
            Defaults to False.
        quant_state_dict_split_scale_bias (bool, optional): Whether to split the scale
            and bias parameters when saving the quantized state dict. Defaults to False.
        row_alignment (int, optional): The alignment of rows in the quantized weights.
            Defaults to DEFAULT_ROW_ALIGNMENT.
        managed_collision_collection (ManagedCollisionCollection, optional): The managed
            collision collection to use for managing collisions. Defaults to None.
        return_remapped_features (bool, optional): Whether to return the remapped input
            features in addition to the embeddings. Defaults to False.
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool,
        device: torch.device,
        output_dtype: torch.dtype = torch.float,
        table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None,
        register_tbes: bool = False,
        quant_state_dict_split_scale_bias: bool = False,
        row_alignment: int = DEFAULT_ROW_ALIGNMENT,
        managed_collision_collection: Optional[ManagedCollisionCollection] = None,
        return_remapped_features: bool = False,
    ) -> None:
        super().__init__(
            tables,
            is_weighted,
            device,
            output_dtype,
            table_name_to_quantized_weights,
            register_tbes,
            quant_state_dict_split_scale_bias,
            row_alignment,
        )
        assert (
            managed_collision_collection
        ), "Managed collision collection cannot be None"
        self._managed_collision_collection: ManagedCollisionCollection = (
            managed_collision_collection
        )
        self._return_remapped_features = return_remapped_features

        assert str(self.embedding_bag_configs()) == str(
            self._managed_collision_collection.embedding_configs()
        ), (
            "EmbeddingBagCollection and Managed Collision Collection must contain the "
            "same Embedding Configs"
        )

    def to(
        self, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> "QuantManagedCollisionEmbeddingBagCollection":
        """To device and dtype."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(
            *args,  # pyre-ignore
            **kwargs,  # pyre-ignore
        )
        for param in self.parameters():
            if param.device.type != "meta":
                param.to(device)

        for buffer in self.buffers():
            if buffer.device.type != "meta":
                buffer.to(device)
        # Skip device movement and continue with other args
        super().to(
            dtype=dtype,
            non_blocking=non_blocking,
        )
        return self

    # pyre-ignore[15]
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Tuple[KeyedTensor, Optional[KeyedJaggedTensor]]:
        """Forward the module."""
        features = self._managed_collision_collection(features)
        embedding_res = super().forward(features)

        if not self._return_remapped_features:
            return embedding_res, None
        return embedding_res, features

    def _get_name(self) -> str:
        return "QuantManagedCollisionEmbeddingBagCollection"

    @classmethod
    # pyre-ignore[14]
    def from_float(
        cls,
        module: ManagedCollisionEmbeddingBagCollection,
        return_remapped_features: bool = False,
    ) -> "QuantManagedCollisionEmbeddingBagCollection":
        """Convert MC-EBC to Quant MC-EBC."""
        mc_ebc = module
        ebc = module._embedding_module

        qconfig: torch.quantization.QConfig = module.qconfig
        assert hasattr(module, "qconfig"), (
            "QuantManagedCollisionEmbeddingBagCollection input float module must "
            "have qconfig defined"
        )

        embedding_configs = copy.deepcopy(ebc.embedding_bag_configs())
        _update_embedding_configs(
            cast(List[BaseEmbeddingConfig], embedding_configs),
            qconfig,
        )
        _update_embedding_configs(
            mc_ebc._managed_collision_collection._embedding_configs,
            qconfig,
        )

        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]] | None = (
            ebc._table_name_to_quantized_weights
            if hasattr(ebc, "_table_name_to_quantized_weights")
            else None
        )
        device = _get_device(ebc)
        return cls(
            embedding_configs,
            ebc.is_weighted(),
            device=device,
            output_dtype=qconfig.activation().dtype,
            table_name_to_quantized_weights=table_name_to_quantized_weights,
            register_tbes=getattr(module, MODULE_ATTR_REGISTER_TBES_BOOL, False),
            quant_state_dict_split_scale_bias=getattr(
                ebc, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            ),
            row_alignment=getattr(
                ebc, MODULE_ATTR_ROW_ALIGNMENT_INT, DEFAULT_ROW_ALIGNMENT
            ),
            managed_collision_collection=mc_ebc._managed_collision_collection,
            return_remapped_features=mc_ebc._return_remapped_features,
        )


def quantize_embeddings(
    module: nn.Module,
    dtype: torch.dtype,
    inplace: bool,
    additional_qconfig_spec_keys: Optional[List[Type[nn.Module]]] = None,
    additional_mapping: Optional[Dict[Type[nn.Module], Type[nn.Module]]] = None,
    output_dtype: torch.dtype = torch.float,
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
) -> nn.Module:
    """Converts a float embedding to quantized embedding."""
    qconfig = QuantConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_dtype),
        weight=quant.PlaceholderObserver.with_args(dtype=dtype),
        per_table_weight_dtype=per_table_weight_dtype,
    )
    # TODO: add EmbeddingCollection and ManagedCollisionEmbeddingCollection into qconfig
    qconfig_spec: Dict[Type[nn.Module], QuantConfig] = {
        EmbeddingBagCollection: qconfig,
        ManagedCollisionEmbeddingBagCollection: qconfig,
    }
    mapping: Dict[Type[nn.Module], Type[nn.Module]] = {
        EmbeddingBagCollection: QuantEmbeddingBagCollection,
        ManagedCollisionEmbeddingBagCollection: QuantManagedCollisionEmbeddingBagCollection,  # NOQA
    }
    if additional_qconfig_spec_keys is not None:
        for t in additional_qconfig_spec_keys:
            qconfig_spec[t] = qconfig
    if additional_mapping is not None:
        mapping.update(additional_mapping)
    return quant.quantize_dynamic(
        module,
        qconfig_spec=qconfig_spec,
        mapping=mapping,
        inplace=inplace,
    )
