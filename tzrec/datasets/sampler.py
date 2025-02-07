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

import math
import os
import random
import socket
import time
from typing import Dict, List, Optional, Tuple, Union

import graphlearn as gl
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import torch
from graphlearn.python.data.values import Values
from graphlearn.python.nn.pytorch.data.utils import launch_server
from torch import distributed as dist
from torch.utils.data import get_worker_info

from tzrec.protos import sampler_pb2
from tzrec.utils.env_util import use_hash_node_id
from tzrec.utils.load_class import get_register_class_meta
from tzrec.utils.misc_util import get_free_port


# patch graph-learn string_attrs for utf-8
@property
def string_attrs(self):  # NOQA
    self._init()
    return self._string_attrs


# pyre-ignore [56]
@string_attrs.setter
# pyre-ignore [2, 3]
def string_attrs(self, string_attrs):  # NOQA
    self._string_attrs = self._reshape(string_attrs, expand_shape=True)
    self._inited = True


Values.string_attrs = string_attrs


def _get_gl_type(field_type: pa.DataType) -> str:
    type_map = {
        pa.int32(): "int",
        pa.int64(): "int",
        pa.float32(): "float",
        pa.float64(): "float",
    }
    if field_type in type_map:
        return type_map[field_type]
    else:
        return "string"


def _get_np_type(field_type: pa.DataType) -> npt.DTypeLike:
    type_map = {
        pa.int32(): np.int32,
        pa.int64(): np.int64,
        pa.float32(): np.float32,
        pa.float64(): np.double,
    }
    if field_type in type_map:
        return type_map[field_type]
    else:
        return np.str_


def _bootstrap(group_size: int, local_rank: int, group_rank: int) -> str:
    def addr_to_tensor(ip: str, port: str) -> torch.Tensor:
        addr_array = [int(i) for i in (ip.split("."))] + [int(port)]
        addr_tensor = torch.tensor(addr_array, dtype=torch.int)
        return addr_tensor

    def tensor_to_addr(tensor: torch.Tensor) -> str:
        addr_array = tensor.tolist()
        addr = ".".join([str(i) for i in addr_array[:-1]]) + ":" + str(addr_array[-1])
        return addr

    def exchange_gl_server_info(
        addr_tensor: torch.Tensor, group_size: int, group_rank: int
    ) -> str:
        comm_tensor = torch.zeros([group_size, 5], dtype=torch.int32)
        comm_tensor[group_rank] = addr_tensor
        if dist.get_backend() == dist.Backend.NCCL:
            comm_tensor = comm_tensor.cuda()
        dist.all_reduce(comm_tensor, op=dist.ReduceOp.MAX)
        cluster_server_info = ",".join([tensor_to_addr(t) for t in comm_tensor])
        return cluster_server_info

    if local_rank == 0:
        local_ip = socket.gethostbyname(socket.gethostname())
        port = str(get_free_port(local_ip))
    else:
        local_ip = "0.0.0.0"
        port = "0"

    if not dist.is_initialized():  # stand-alone
        return local_ip + ":" + port

    gl_server_info = exchange_gl_server_info(
        addr_to_tensor(local_ip, port), group_size, group_rank
    )
    return gl_server_info


def _get_cluster_spec(num_client_per_rank: int = 1) -> Dict[str, Union[int, str]]:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    group_size = world_size // int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    group_rank = int(os.environ.get("GROUP_RANK", 0))
    num_client = num_client_per_rank
    gl_server_info = _bootstrap(group_size, local_rank, group_rank)
    return {"server": gl_server_info, "client_count": world_size * num_client}


_SAMPLER_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_SAMPLER_CLASS_MAP)


SAMPLER_CFG_TYPES = Union[
    sampler_pb2.NegativeSampler,
    sampler_pb2.NegativeSamplerV2,
    sampler_pb2.HardNegativeSampler,
    sampler_pb2.HardNegativeSamplerV2,
    sampler_pb2.TDMSampler,
]


def _to_arrow_array(
    x: npt.NDArray, field_type: pa.DataType, multival_sep: str = chr(29)
) -> pa.Array:
    if pa.types.is_list(field_type) or pa.types.is_map(field_type):
        x = pa.array(x, type=pa.string())
        is_empty = pa.compute.equal(x, pa.scalar(""))
        x = pa.compute.if_else(is_empty, pa.nulls(len(x)), x)
        if pa.types.is_list(field_type):
            result = pa.compute.split_pattern(x, pattern=multival_sep).cast(
                field_type, safe=False
            )
        else:
            kv = pa.compute.split_pattern_regex(x, pattern=multival_sep)
            offsets = kv.offsets
            kv_list = pa.compute.split_pattern(kv.values, ":").values
            keys = kv_list.take(list(range(0, len(kv_list), 2))).cast(
                field_type.key_type, safe=False
            )
            items = kv_list.take(list(range(1, len(kv_list), 2))).cast(
                field_type.item_type, safe=False
            )
            result = pa.MapArray.from_arrays(offsets, keys, items)

    else:
        result = pa.array(x, type=field_type)
    if isinstance(result, pa.ChunkedArray):
        result = result.combine_chunks()
    return result


def _pa_ids_to_npy(ids: pa.Array) -> npt.NDArray:
    """Convert pyarrow id array to numpy array."""
    if use_hash_node_id():
        ids = ids.cast(pa.string()).to_numpy(zero_copy_only=False)
    else:
        ids = ids.cast(pa.int64()).fill_null(0).to_numpy()
    return ids


class BaseSampler(metaclass=_meta_cls):
    """Negative Sampler base class."""

    def __init__(
        self,
        config: SAMPLER_CFG_TYPES,
        fields: List[pa.Field],
        batch_size: int,
        is_training: bool = True,
        multival_sep: str = chr(29),
    ) -> None:
        self._batch_size = batch_size
        self._multival_sep = multival_sep
        self._g = None
        if hasattr(config, "num_sample"):
            # pyre-ignore [16]
            self._num_sample = config.num_sample
        else:
            self._num_sample = None
        if not is_training and config.HasField("num_eval_sample"):
            self._num_sample = config.num_eval_sample

        self._cluster = None

        input_fields = {f.name: f for f in fields}
        self._attr_names = []
        self._attr_types = []
        self._attr_gl_types = []
        self._attr_np_types = []
        for field_name in config.attr_fields:
            field = input_fields[field_name]
            self._attr_names.append(field.name)
            self._attr_types.append(field.type)
            self._attr_gl_types.append(_get_gl_type(field.type))
            self._attr_np_types.append(_get_np_type(field.type))

        if config.HasField("field_delimiter"):
            gl.set_field_delimiter(config.field_delimiter)
        if use_hash_node_id():
            gl.set_use_string_hash_id(1)

        self._num_client_per_rank = 1
        self._client_id_bias = 0

    def init_cluster(
        self,
        num_client_per_rank: int = 1,
        client_id_bias: int = 0,
        cluster: Optional[Dict[str, Union[int, str]]] = None,
    ) -> None:
        """Set client in cluster info."""
        gl.set_load_graph_thread_num(max(num_client_per_rank // 2, 1))
        self._num_client_per_rank = num_client_per_rank
        self._client_id_bias = client_id_bias
        if cluster:
            self._cluster = cluster
        else:
            self._cluster = _get_cluster_spec(self._num_client_per_rank)

    def launch_server(self) -> None:
        """Launch sampler server."""
        assert self._cluster, "should init cluster first."
        gl.set_tracker_mode(0)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            launch_server(self._g, self._cluster, int(os.environ.get("GROUP_RANK", 0)))

    def init(self, client_id: int = -1) -> None:
        """Init sampler client and samplers."""
        gl.set_tracker_mode(0)
        assert self._cluster, "should init cluster first."
        if client_id < 0:
            worker_info = get_worker_info()
            if worker_info is None:
                client_id = 0
            else:
                client_id = worker_info.id
        client_id += self._client_id_bias
        task_index = (
            self._num_client_per_rank * int(os.environ.get("RANK", 0)) + client_id
        )
        # print(f"Init task {task_index} in cluster {self._cluster}")
        self._g.init(task_index=task_index, job_name="client", cluster=self._cluster)

    def __del__(self) -> None:
        if self._g is not None:
            self._g.close()

    def _parse_nodes(self, nodes: gl.Nodes) -> List[pa.Array]:
        features = []
        int_idx = 0
        float_idx = 0
        string_idx = 0
        for attr_type, attr_gl_type, attr_np_type in zip(
            self._attr_types, self._attr_gl_types, self._attr_np_types
        ):
            if attr_gl_type == "int":
                feature = nodes.int_attrs[:, :, int_idx]
                int_idx += 1
            elif attr_gl_type == "float":
                feature = nodes.float_attrs[:, :, float_idx]
                float_idx += 1
            elif attr_gl_type == "string":
                feature = nodes.string_attrs[:, :, string_idx].astype(np.string_)
                feature = np.char.decode(feature, "utf-8")
                string_idx += 1
            else:
                raise ValueError("Unknown attr type %s" % attr_gl_type)
            feature = np.reshape(feature, [-1])[: self._num_sample].astype(attr_np_type)
            feature = _to_arrow_array(feature, attr_type)
            features.append(feature)
        return features

    def _parse_sparse_nodes(
        self, nodes: gl.Nodes
    ) -> Tuple[List[pa.Array], npt.NDArray]:
        features = []
        int_idx = 0
        float_idx = 0
        string_idx = 0
        for attr_type, attr_gl_type, attr_np_type in zip(
            self._attr_types, self._attr_gl_types, self._attr_np_types
        ):
            if attr_gl_type == "int":
                feature = nodes.int_attrs[:, int_idx]
                int_idx += 1
            elif attr_gl_type == "float":
                feature = nodes.float_attrs[:, float_idx]
                float_idx += 1
            elif attr_gl_type == "string":
                feature = nodes.string_attrs[:, string_idx].astype(np.string_)
                feature = np.char.decode(feature, "utf-8")
                string_idx += 1
            else:
                raise ValueError("Unknown attr type %s" % attr_gl_type)
            feature = feature.astype(attr_np_type)
            feature = _to_arrow_array(feature, attr_type)
            features.append(feature)
        # pyre-ignore [16]
        return features, nodes.indices

    @property
    def estimated_sample_num(self) -> int:
        """Max number of sampled num examples."""
        raise NotImplementedError


class NegativeSampler(BaseSampler):
    """Negative Sampler.

    Weighted random sampling items not in batch.

    Args:
        config (NegativeSampler): negative sampler config.
        fields (list): item input fields.
        batch_size (int): mini-batch size.
        is_training (bool): train or eval.
        multival_sep (str): multi value separator.
    """

    def __init__(
        self,
        config: sampler_pb2.NegativeSampler,
        fields: List[pa.Field],
        batch_size: int,
        is_training: bool = True,
        multival_sep: str = chr(29),
    ) -> None:
        super(NegativeSampler, self).__init__(
            config, fields, batch_size, is_training, multival_sep
        )
        self._g = gl.Graph().node(
            config.input_path,
            node_type="item",
            decoder=gl.Decoder(
                attr_types=self._attr_gl_types,
                weighted=True,
                attr_delimiter=config.attr_delimiter,
            ),
        )
        self._item_id_field = config.item_id_field
        self._sampler = None

    def init(self, client_id: int = -1) -> None:
        """Init sampler client and samplers."""
        super().init(client_id)
        expand_factor = int(math.ceil(self._num_sample / self._batch_size))
        self._sampler = self._g.negative_sampler(
            "item", expand_factor, strategy="node_weight"
        )

    def get(self, input_data: Dict[str, pa.Array]) -> Dict[str, pa.Array]:
        """Sampling method.

        Args:
            input_data (dict): input data with item_id.

        Returns:
            Negative sampled feature dict.
        """
        ids = _pa_ids_to_npy(input_data[self._item_id_field])
        # ids = np.pad(ids, (0, self._batch_size - len(ids)), "edge")
        # nodes = self._sampler.get(ids)
        # features = self._parse_nodes(nodes)
        # result_dict = dict(zip(self._attr_names, features))
        return {
            "item_id": pa.array(
                np.random.randint(0, 3953, size=int(ids.shape[0] * self._num_sample))
            ),
        }

    @property
    def estimated_sample_num(self) -> int:
        """Estimated number of sampled num examples."""
        return self._num_sample


class NegativeSamplerV2(BaseSampler):
    """Negative Sampler V2.

    Weighted random sampling items which do not have positive edge with the user.

    Args:
        config (NegativeSampler): negative sampler config.
        fields (list): item input fields.
        batch_size (int): mini-batch size.
        is_training (bool): train or eval.
        multival_sep (str): multi value separator.
    """

    def __init__(
        self,
        config: sampler_pb2.NegativeSamplerV2,
        fields: List[pa.Field],
        batch_size: int,
        is_training: bool = True,
        multival_sep: str = chr(29),
    ) -> None:
        super(NegativeSamplerV2, self).__init__(
            config, fields, batch_size, is_training, multival_sep
        )
        self._g = (
            gl.Graph()
            .node(
                config.user_input_path,
                node_type="user",
                decoder=gl.Decoder(weighted=True),
            )
            .node(
                config.item_input_path,
                node_type="item",
                decoder=gl.Decoder(
                    attr_types=self._attr_gl_types,
                    weighted=True,
                    attr_delimiter=config.attr_delimiter,
                ),
            )
            .edge(
                config.pos_edge_input_path,
                edge_type=("user", "item", "edge"),
                decoder=gl.Decoder(weighted=True),
            )
        )
        self._item_id_field = config.item_id_field
        self._user_id_field = config.user_id_field
        self._sampler = None

    def init(self, client_id: int = -1) -> None:
        """Init sampler client and samplers."""
        super().init(client_id)
        expand_factor = int(math.ceil(self._num_sample / self._batch_size))
        self._sampler = self._g.negative_sampler(
            "edge", expand_factor, strategy="random", conditional=True
        )

        # prevent gl timeout
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        time.sleep(random.randint(0, num_workers * local_world_size))
        self.get(
            {self._user_id_field: pa.array([0]), self._item_id_field: pa.array([0])}
        )

    def get(self, input_data: Dict[str, pa.Array]) -> Dict[str, pa.Array]:
        """Sampling method.

        Args:
            input_data (dict): input data with user_id and item_id.

        Returns:
            Negative sampled feature dict.
        """
        src_ids = _pa_ids_to_npy(input_data[self._user_id_field])
        dst_ids = _pa_ids_to_npy(input_data[self._item_id_field])
        src_ids = np.pad(src_ids, (0, self._batch_size - len(src_ids)), "edge")
        dst_ids = np.pad(dst_ids, (0, self._batch_size - len(dst_ids)), "edge")
        nodes = self._sampler.get(src_ids, dst_ids)
        features = self._parse_nodes(nodes)
        result_dict = dict(zip(self._attr_names, features))
        return result_dict

    @property
    def estimated_sample_num(self) -> int:
        """Estimated number of sampled num examples."""
        return self._num_sample


class HardNegativeSampler(BaseSampler):
    """HardNegativeSampler.

    Weighted random sampling items not in batch as negative samples, and sampling
    destination nodes in hard_neg_edge as hard negative samples

    Args:
        config (NegativeSampler): negative sampler config.
        fields (list): item input fields.
        batch_size (int): mini-batch size.
        is_training (bool): train or eval.
        multival_sep (str): multi value separator.
    """

    def __init__(
        self,
        config: sampler_pb2.HardNegativeSampler,
        fields: List[pa.Field],
        batch_size: int,
        is_training: bool = True,
        multival_sep: str = chr(29),
    ) -> None:
        super(HardNegativeSampler, self).__init__(
            config, fields, batch_size, is_training, multival_sep
        )
        self._num_hard_sample = config.num_hard_sample
        self._g = (
            gl.Graph()
            .node(
                config.user_input_path,
                node_type="user",
                decoder=gl.Decoder(weighted=True),
            )
            .node(
                config.item_input_path,
                node_type="item",
                decoder=gl.Decoder(
                    attr_types=self._attr_gl_types,
                    weighted=True,
                    attr_delimiter=config.attr_delimiter,
                ),
            )
            .edge(
                config.hard_neg_edge_input_path,
                edge_type=("user", "item", "hard_neg_edge"),
                decoder=gl.Decoder(weighted=True),
            )
        )
        self._item_id_field = config.item_id_field
        self._user_id_field = config.user_id_field
        self._neg_sampler = None
        self._hard_neg_sampler = None

    def init(self, client_id: int = -1) -> None:
        """Init sampler client and samplers."""
        super().init(client_id)
        expand_factor = int(math.ceil(self._num_sample / self._batch_size))
        self._neg_sampler = self._g.negative_sampler(
            "item", expand_factor, strategy="node_weight"
        )
        self._hard_neg_sampler = self._g.neighbor_sampler(
            ["hard_neg_edge"], self._num_hard_sample, strategy="full"
        )

    def get(self, input_data: Dict[str, pa.Array]) -> Dict[str, pa.Array]:
        """Sampling method.

        Args:
            input_data (dict): input data with user_id and item_id.

        Returns:
            Negative sampled feature dict. The first batch_size is negative samples,
                remainder is hard negative samples
        """
        src_ids = _pa_ids_to_npy(input_data[self._user_id_field])
        dst_ids = _pa_ids_to_npy(input_data[self._item_id_field])
        dst_ids = np.pad(dst_ids, (0, self._batch_size - len(dst_ids)), "edge")
        nodes = self._neg_sampler.get(dst_ids)
        neg_features = self._parse_nodes(nodes)
        sparse_nodes = self._hard_neg_sampler.get(src_ids).layer_nodes(1)
        hard_neg_features, hard_neg_indices = self._parse_sparse_nodes(sparse_nodes)

        results = []
        for i, v in enumerate(hard_neg_features):
            results.append(pa.concat_arrays([neg_features[i], v]))

        result_dict = dict(zip(self._attr_names, results))
        result_dict["hard_neg_indices"] = pa.array(hard_neg_indices)
        return result_dict

    @property
    def estimated_sample_num(self) -> int:
        """Estimated number of sampled num examples."""
        return self._num_sample + min(self._num_hard_sample, 8) * self._batch_size


class HardNegativeSamplerV2(BaseSampler):
    """HardNegativeSampler.

    Weighted random sampling items which do not have positive edge with the user,
    and sampling destination nodes in hard_neg_edge as hard negative samples.

    Args:
        config (NegativeSampler): negative sampler config.
        fields (list): item input fields.
        batch_size (int): mini-batch size.
        is_training (bool): train or eval.
        multival_sep (str): multi value separator.
    """

    def __init__(
        self,
        config: sampler_pb2.HardNegativeSamplerV2,
        fields: List[pa.Field],
        batch_size: int,
        is_training: bool = True,
        multival_sep: str = chr(29),
    ) -> None:
        super(HardNegativeSamplerV2, self).__init__(
            config, fields, batch_size, is_training, multival_sep
        )
        self._num_hard_sample = config.num_hard_sample
        self._g = (
            gl.Graph()
            .node(
                config.user_input_path,
                node_type="user",
                decoder=gl.Decoder(weighted=True),
            )
            .node(
                config.item_input_path,
                node_type="item",
                decoder=gl.Decoder(
                    attr_types=self._attr_gl_types,
                    weighted=True,
                    attr_delimiter=config.attr_delimiter,
                ),
            )
            .edge(
                config.pos_edge_input_path,
                edge_type=("user", "item", "edge"),
                decoder=gl.Decoder(weighted=True),
            )
            .edge(
                config.hard_neg_edge_input_path,
                edge_type=("user", "item", "hard_neg_edge"),
                decoder=gl.Decoder(weighted=True),
            )
        )
        self._item_id_field = config.item_id_field
        self._user_id_field = config.user_id_field
        self._neg_sampler = None
        self._hard_neg_sampler = None

    def init(self, client_id: int = -1) -> None:
        """Init sampler client and samplers."""
        super().init(client_id)
        expand_factor = int(math.ceil(self._num_sample / self._batch_size))
        self._neg_sampler = self._g.negative_sampler(
            "edge", expand_factor, strategy="random", conditional=True
        )
        self._hard_neg_sampler = self._g.neighbor_sampler(
            ["hard_neg_edge"], self._num_hard_sample, strategy="full"
        )

    def get(self, input_data: Dict[str, pa.Array]) -> Dict[str, pa.Array]:
        """Sampling method.

        Args:
            input_data (dict): input data with user_id and item_id.

        Returns:
            Negative sampled feature dict. The first batch_size is negative samples,
                remainder is hard negative samples
        """
        src_ids = _pa_ids_to_npy(input_data[self._user_id_field])
        dst_ids = _pa_ids_to_npy(input_data[self._item_id_field])
        padded_src_ids = np.pad(src_ids, (0, self._batch_size - len(src_ids)), "edge")
        dst_ids = np.pad(dst_ids, (0, self._batch_size - len(dst_ids)), "edge")
        nodes = self._neg_sampler.get(padded_src_ids, dst_ids)
        neg_features = self._parse_nodes(nodes)
        sparse_nodes = self._hard_neg_sampler.get(src_ids).layer_nodes(1)
        hard_neg_features, hard_neg_indices = self._parse_sparse_nodes(sparse_nodes)

        results = []
        for i, v in enumerate(hard_neg_features):
            results.append(pa.concat_arrays([neg_features[i], v]))

        result_dict = dict(zip(self._attr_names, results))
        result_dict["hard_neg_indices"] = pa.array(hard_neg_indices)
        return result_dict

    @property
    def estimated_sample_num(self) -> int:
        """Estimated number of sampled num examples."""
        return self._num_sample + min(self._num_hard_sample, 8) * self._batch_size


class TDMSampler(BaseSampler):
    """TDM training sampler.

    According to the leaf nodes corresponding to the items clicked by the user,
    sample all ancestor nodes as positive samples,
    and then sample negative samples layer by layer.

    Args:
        config (NegativeSampler): negative sampler config.
        fields (list): item input fields.
        batch_size (int): mini-batch size.
        is_training (bool): train or eval.
        multival_sep (str): multi value separator.
    """

    def __init__(
        self,
        config: sampler_pb2.TDMSampler,
        fields: List[pa.Field],
        batch_size: int,
        is_training: bool = True,
        multival_sep: str = chr(29),
    ) -> None:
        fields = [pa.field("tree_level", pa.int64())] + fields
        super().__init__(config, fields, batch_size, is_training, multival_sep)
        self._g = (
            gl.Graph()
            .node(
                config.item_input_path,
                node_type="item",
                decoder=gl.Decoder(
                    attr_types=self._attr_gl_types,
                    weighted=True,
                    attr_delimiter=config.attr_delimiter,
                ),
            )
            .edge(
                config.edge_input_path,
                edge_type=("item", "item", "ancestor"),
                decoder=gl.Decoder(weighted=True),
            )
        )
        self._item_id_field = config.item_id_field
        self._max_level = len(config.layer_num_sample)
        self._layer_num_sample = config.layer_num_sample
        assert self._layer_num_sample[0] == 0, "sample num of tree root must be 0"
        self._last_layer_num_sample = config.layer_num_sample[-1]
        self._pos_sampler = None
        self._neg_sampler_list = []

        self._remain_ratio = config.remain_ratio
        if self._remain_ratio < 1.0:
            if config.probability_type == "UNIFORM":
                p = np.array([1 / (self._max_level - 2)] * (self._max_level - 2))
            elif config.probability_type == "ARITHMETIC":
                p = np.arange(1, self._max_level - 1) / sum(
                    np.arange(1, self._max_level - 1)
                )
            elif config.probability_type == "RECIPROCAL":
                p = 1 / np.arange(self._max_level - 2, 0, -1)
                p = p / sum(p)
            else:
                raise ValueError(
                    f"probability_type: [{config.probability_type}]"
                    "is not supported now."
                )
            self._remain_p = p

    def init(self, client_id: int = -1) -> None:
        """Init sampler client and samplers."""
        super().init(client_id)
        self._pos_sampler = self._g.neighbor_sampler(
            meta_path=["ancestor"],
            expand_factor=self._max_level - 2,
            strategy="random_without_replacement",
        )

        # TODO: only use one conditional smapler
        for i in range(1, self._max_level):
            self._neg_sampler_list.append(
                self._g.negative_sampler(
                    "item",
                    expand_factor=self._layer_num_sample[i],
                    strategy="node_weight",
                    conditional=True,
                    int_cols=[0],
                    int_props=[1],
                    samplewise_unique=True,
                )
            )

        # prevent gl timeout
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        time.sleep(random.randint(0, num_workers * local_world_size))
        if use_hash_node_id():
            self.get({self._item_id_field: pa.array(["0"], type=np.object_)})
        else:
            self.get({self._item_id_field: pa.array([0])})

    def get(self, input_data: Dict[str, pa.Array]) -> Dict[str, pa.Array]:
        """Sampling method.

        Args:
            input_data (dict): input data with item_id.

        Returns:
            Positive and negative sampled feature dict.
        """
        ids = _pa_ids_to_npy(input_data[self._item_id_field]).reshape(-1, 1)
        batch_size = len(ids)
        num_fea = len(self._attr_names[1:])

        # positive node.
        pos_nodes = self._pos_sampler.get(ids).layer_nodes(1)

        # the ids of non-leaf nodes is arranged in ascending order.
        pos_non_leaf_ids = np.sort(pos_nodes.ids, axis=1)
        pos_ids = np.concatenate((pos_non_leaf_ids, ids), axis=1)
        pos_fea_result = self._parse_nodes(pos_nodes)[1:]

        # randomly select layers to keep
        if self._remain_ratio < 1.0:
            remain_layer = np.random.choice(
                range(1, self._max_level - 1),
                int(round(self._remain_ratio * (self._max_level - 2))),
                replace=False,
                p=self._remain_p,
            )
        else:
            remain_layer = np.array(range(1, self._max_level - 1))
        remain_layer.sort()

        if self._remain_ratio < 1.0:
            pos_fea_index = np.concatenate(
                [
                    remain_layer - 1 + j * (self._max_level - 2)
                    for j in range(batch_size)
                ]
            )
            pos_fea_result = [
                pos_fea_result[i].take(pos_fea_index) for i in range(num_fea)
            ]

        # negative sample layer by layer.
        neg_fea_layer = []
        for i in np.append(remain_layer, self._max_level - 1):
            neg_nodes = self._neg_sampler_list[i - 1].get(
                pos_ids[:, i - 1], pos_ids[:, i - 1]
            )
            features = self._parse_nodes(neg_nodes)[1:]
            neg_fea_layer.append(features)

        # concatenate the features of each layer and
        # ensure that the negative sample features of the same user are adjacent.
        neg_fea_result = []
        cum_layer_num = np.cumsum(
            [0]
            + [
                self._layer_num_sample[i] if i in remain_layer else 0
                for i in range(self._max_level - 1)
            ]
        )
        neg_fea_index = np.concatenate(
            [
                np.concatenate(
                    [
                        np.arange(self._layer_num_sample[i])
                        + j * self._layer_num_sample[i]
                        + batch_size * cum_layer_num[i]
                        for i in np.append(remain_layer, self._max_level - 1)
                    ]
                )
                for j in range(batch_size)
            ]
        )
        neg_fea_result = [
            pa.concat_arrays([array[i] for array in neg_fea_layer]).take(neg_fea_index)
            for i in range(num_fea)
        ]

        pos_result_dict = dict(zip(self._attr_names[1:], pos_fea_result))
        neg_result_dict = dict(zip(self._attr_names[1:], neg_fea_result))

        return pos_result_dict, neg_result_dict

    @property
    def estimated_sample_num(self) -> int:
        """Estimated number of sampled num examples."""
        return (
            sum(self._layer_num_sample) + len(self._layer_num_sample) - 2
        ) * self._batch_size


class TDMPredictSampler(BaseSampler):
    """TDM predict sampler.

    Args:
        config (NegativeSampler): negative sampler config.
        fields (list): item input fields.
        batch_size (int): mini-batch size.
        is_training (bool): train or eval.
        multival_sep (str): multi value separator.
    """

    def __init__(
        self,
        config: sampler_pb2.TDMSampler,
        fields: List[pa.Field],
        batch_size: int,
        is_training: bool = True,
        multival_sep: str = chr(29),
    ) -> None:
        fields = [pa.field("tree_level", pa.int64())] + fields
        super().__init__(config, fields, batch_size, is_training, multival_sep)
        self._g = (
            gl.Graph()
            .node(
                config.item_input_path,
                node_type="item",
                decoder=gl.Decoder(
                    attr_types=self._attr_gl_types,
                    weighted=True,
                    attr_delimiter=config.attr_delimiter,
                ),
            )
            .edge(
                config.predict_edge_input_path,
                edge_type=("item", "item", "children"),
                decoder=gl.Decoder(weighted=True),
            )
        )
        self._item_id_field = config.item_id_field
        self._max_level = len(config.layer_num_sample)
        self._pos_sampler = None

    def init_sampler(self, expand_factor: int) -> None:
        """Init samplers with different expand_factor.

        During prediction, the first sampling selects all nodes from the first
        layer larger than the recall number, starting from the root node. Then,
        for each node, all its child nodes are sampled. The expand_factor is
        different in the two rounds of sampling.
        """
        self._pos_sampler = self._g.neighbor_sampler(
            meta_path=["children"],
            expand_factor=expand_factor,
            strategy="random_without_replacement",
        )

    def get(self, input_data: Dict[str, pa.Array]) -> Dict[str, pa.Array]:
        """Sampling method.

        Args:
            input_data (dict): input data with item_id.

        Returns:
            Positive and negative sampled feature dict.
        """
        ids = _pa_ids_to_npy(input_data[self._item_id_field]).reshape(-1, 1)

        pos_nodes = self._pos_sampler.get(ids).layer_nodes(1)
        pos_fea_result = self._parse_nodes(pos_nodes)[1:]
        pos_result_dict = dict(zip(self._attr_names[1:], pos_fea_result))

        return pos_result_dict

    @property
    def estimated_sample_num(self) -> int:
        """Estimated number of sampled num examples."""
        return min((2 ** (self._max_level - 1)), 800) * self._batch_size
