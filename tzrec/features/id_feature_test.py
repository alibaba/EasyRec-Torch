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


import unittest
from functools import partial

import numpy as np
import pyarrow as pa
import torch
from parameterized import parameterized
from torch import nn
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)

from tzrec.datasets.utils import C_SAMPLE_MASK
from tzrec.features import id_feature as id_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class IdFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [["1\x032", "", None, "3"], "", [1, 2, 3], [2, 0, 0, 1]],
            [["1\x032", "", None, "3"], "0", [1, 2, 0, 0, 3], [2, 1, 1, 1]],
            [[1, 2, None, 3], "", [1, 2, 3], [1, 1, 0, 1]],
            [[1, 2, None, 3], "0", [1, 2, 0, 3], [1, 1, 1, 1]],
        ]
    )
    def test_fg_encoded_id_feature(
        self, input_feat, default_value, expected_values, expected_lengths
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                fg_encoded_default_value=default_value,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        self.assertEqual(id_feat.output_dim, 16)
        self.assertEqual(id_feat.is_sparse, True)
        self.assertEqual(id_feat.inputs, ["id_feat"])

        input_data = {"id_feat": pa.array(input_feat)}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    def test_init_fn_id_feature(self):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                num_buckets=100,
                init_fn="nn.init.uniform_,b=0.01",
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.SUM,
            init_fn=partial(nn.init.uniform_, b=0.01),
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            init_fn=partial(nn.init.uniform_, b=0.01),
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

    @parameterized.expand(
        [
            ["lambda x: probabilistic_threshold_filter(x,0.05)"],
            ["lambda x: (x > 10, 10)"],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_zch_id_feature(self, threshold_filtering_func):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                zch=feature_pb2.ZeroCollisionHash(
                    zch_size=100,
                    eviction_interval=5,
                    distance_lfu=feature_pb2.DistanceLFU_EvictionPolicy(
                        decay_exponent=1.0,
                    ),
                    threshold_filtering_func=threshold_filtering_func,
                ),
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))
        mc_module = id_feat.mc_module(torch.device("meta"))
        self.assertEqual(mc_module._zch_size, 100)
        self.assertEqual(mc_module._eviction_interval, 5)
        self.assertTrue(
            mc_module._eviction_policy._threshold_filtering_func is not None
        )

    def test_fg_encoded_with_weighted(self):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cate",
                hash_bucket_size=10,
                embedding_dim=16,
                expression="item:cate",
                weighted=True,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        self.assertEqual(id_feat.inputs[0], "cate__values")
        self.assertEqual(id_feat.inputs[1], "cate__weights")

        input_data = {
            "cate__values": pa.array([1, 2, 3]),
            "cate__weights": pa.array([1.0, 1.5, 2.0]),
        }
        parsed_feat = id_feat.parse(input_data)
        expected_values = [1, 2, 3]
        expected_lengths = [1, 1, 1]
        expected_weights = [1.0, 1.5, 2.0]
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))
        np.testing.assert_allclose(parsed_feat.weights, np.array(expected_weights))

    def test_fg_encoded_id_feature_with_mask(self):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                use_mask=True,
                fg_encoded_default_value="",
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        self.assertEqual(id_feat.output_dim, 16)
        self.assertEqual(id_feat.is_sparse, True)
        self.assertEqual(id_feat.inputs, ["id_feat"])

        input_data = {
            "id_feat": pa.array(["1\x032", "", None, "3"]),
            C_SAMPLE_MASK: pa.array([True, False, False, False]),
        }
        np.random.seed(42)
        parsed_feat = id_feat.parse(input_data, is_training=True)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([3]))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([0, 0, 0, 1]))

    def test_id_feature_with_weighted(self):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cate",
                num_buckets=100000,
                embedding_dim=16,
                expression="item:cate",
                weighted=True,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)
        self.assertEqual(id_feat.inputs, ["cate"])

        input_data = {
            "cate": pa.array(["123:0.5", "1391:0.3", None, "12:0.9\035123:0.21", ""])
        }
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "cate")
        np.testing.assert_allclose(parsed_feat.values, np.array([123, 1391, 12, 123]))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([1, 1, 0, 2, 0]))
        self.assertTrue(
            np.allclose(parsed_feat.weights, np.array([0.5, 0.3, 0.9, 0.21]))
        )

    @parameterized.expand(
        [
            ["", ["abc\x1defg", None, "hij"], [33, 44, 66], [2, 0, 1], None],
            ["xyz", ["abc\x1defg", None, "hij"], [33, 44, 13, 66], [2, 1, 1], 13],
            ["xyz", [["abc", "efg"], None, ["hij"]], [33, 44, 13, 66], [2, 1, 1], 13],
            ["", [1, 2, None, 3], [95, 70, 13], [1, 1, 0, 1], None],
            ["4", [1, 2, None, 3], [95, 70, 56, 13], [1, 1, 1, 1], 56],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_hash_bucket_size(
        self,
        default_value,
        input_data,
        expected_values,
        expected_lengths,
        expected_fg_default,
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                hash_bucket_size=100,
                embedding_dim=16,
                expression="user:id_input",
                default_value=default_value,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)
        self.assertEqual(id_feat.inputs, ["id_input"])

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        fg_default = id_feat.fg_encoded_default_value()
        if expected_fg_default:
            np.testing.assert_allclose(fg_default, expected_fg_default)
        else:
            self.assertEqual(fg_default, expected_fg_default)
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

        input_data = {"id_input": pa.array(input_data)}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            ["", ["abc", "efg"], None, [2, 3, 1], [2, 0, 1]],
            ["xyz", ["abc", "efg"], None, [2, 3, 0, 1], [2, 1, 1]],
            ["", ["xyz", "abc", "efg"], 0, [1, 2, 0], [2, 0, 1]],
            ["xyz", ["xyz", "abc", "efg"], 0, [1, 2, 0, 0], [2, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_vocab_list(
        self,
        default_value,
        vocab_list,
        default_bucketize_value,
        expected_values,
        expected_lengths,
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                vocab_list=vocab_list,
                expression="user:id_str",
                pooling="mean",
                default_value=default_value,
            )
        )
        if default_bucketize_value is not None:
            id_feat_cfg.id_feature.default_bucketize_value = default_bucketize_value

        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=4 if default_bucketize_value is None else 3,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.MEAN,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=4 if default_bucketize_value is None else 3,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

        input_data = {"id_str": pa.array(["abc\x1defg", "", "hij"])}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            ["", {"abc": 2, "efg": 2}, None, [2, 2, 1], [2, 0, 1]],
            ["xyz", {"abc": 2, "efg": 2}, None, [2, 2, 0, 1], [2, 1, 1]],
            ["", {"abc": 1, "efg": 1}, 0, [1, 1, 0], [2, 0, 1]],
            ["xyz", {"xyz": 0, "abc": 1, "efg": 1}, 0, [1, 1, 0, 0], [2, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_vocab_dict(
        self,
        default_value,
        vocab_dict,
        default_bucketize_value,
        expected_values,
        expected_lengths,
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                vocab_dict=vocab_dict,
                expression="user:id_str",
                pooling="mean",
                default_value=default_value,
            )
        )
        if default_bucketize_value is not None:
            id_feat_cfg.id_feature.default_bucketize_value = default_bucketize_value

        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=3 if default_bucketize_value is None else 2,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.MEAN,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=3 if default_bucketize_value is None else 2,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

        input_data = {"id_str": pa.array(["abc\x1defg", "", "hij"])}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [["", [0, 1, 2], [2, 0, 1]], ["3", [0, 1, 3, 2], [2, 1, 1]]],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_num_buckets(
        self, default_value, expected_values, expected_lengths
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                num_buckets=100,
                expression="user:id_int",
                default_value=default_value,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

        input_data = {"id_int": pa.array(["0\x1d1", "", "2"])}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))


if __name__ == "__main__":
    unittest.main()
