# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from jax import random
import numpy as np

from connectomics.jax.models import point
from absl.testing import absltest


class PointTest(absltest.TestCase):

  def get_encoder_config(self):
    sa_mlp_config = point.GroupMLPConfig(
        num_layers=3, features=32, pool_type='max', normalizer='layer'
    )

    la_mlp_config = point.GroupMLPConfig(
        num_layers=1, features=32, pool_type='max', normalizer='layer'
    )

    sa_config = point.LocalAggregationConfig(
        mlp_config=sa_mlp_config, skip_conn=True, anchor_concat=True, k=16
    )

    stage_config = point.PointNeXtStageConfig(
        sa_config=sa_config,
        downsample=1,
        blocks=[
            point.InvResMLPConfig(
                normalizer='layer',
                features=[128, 32],
                la_config=point.LocalAggregationConfig(
                    mlp_config=la_mlp_config, k=16
                ),
            )
        ],
    )

    enc_config = point.PointNeXtEncoderConfig(
        stages=[stage_config, stage_config], embed_dim=64, normalizer='layer'
    )
    return enc_config

  def get_key_feat_coord(self):
    key = random.PRNGKey(0)
    feat = np.zeros([16, 128, 5])
    coord = np.zeros([16, 128, 3])
    return key, feat, coord

  def test_pointnext_classifier(self):

    enc_config = self.get_encoder_config()
    key, feat, coord = self.get_key_feat_coord()

    cfg = point.PointNeXtClassifierConfig(
        enc_config,
        'max',
        'layer',
        [32, 64],
        3,
        pca_align=False,
        num_vector_feats=0,
    )
    model = point.PointNeXtClassifier(cfg)

    params = model.init(key, feat, coord)
    res = model.apply(params, feat, coord)
    self.assertEqual(res.shape, (16, 3))

  def test_pointnext_encoder(self):
    enc_config = self.get_encoder_config()
    key, feat, coord = self.get_key_feat_coord()

    cond = np.zeros([16, 32])
    encoder_model = point.PointNeXtEncoder(enc_config)
    params_cond = encoder_model.init(key, feat, coord, cond=cond)
    res_cond = encoder_model.apply(params_cond, feat, coord, cond=cond)
    self.assertEqual(res_cond.shape, (16, 128, 32))

  def test_group_mlp(self):
    mlp_config = point.GroupMLPConfig(
        num_layers=3, features=32, pool_type='max', normalizer='layer'
    )

    model = point.GroupMLP(mlp_config)
    key = random.PRNGKey(0)
    feat = np.random.default_rng(42).random([16, 8, 128, 5])

    params = model.init(key, feat)
    res = model.apply(params, feat)

    # Pooling over neighbors (k=128) should produce [b, n, features].
    self.assertEqual(res.shape, (16, 8, 32))
    # ReLU output should be non-negative.
    self.assertGreaterEqual(np.min(res), 0.0)
    # With random input, the output should be non-trivial.
    self.assertGreater(np.mean(res), 0.0)
    self.assertGreater(np.max(res), np.mean(res))
    # Deterministic: same params and input should give the same output.
    res2 = model.apply(params, feat)
    np.testing.assert_array_equal(res, res2)

  def test_lp_pooling(self):
    x = np.array([[[3.0, 4.0], [0.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]])
    res = point._apply_pool(x, 'l2')
    expected = np.array([[3.0, 4.0], [3.0, 4.0]]) / np.sqrt(2)
    np.testing.assert_allclose(res, expected)

    x2 = np.array([[[1.0], [1.0]]])
    res_l2 = point._apply_pool(x2, 'l2')  # sqrt((1^2 + 1^2)/2) = 1
    np.testing.assert_allclose(res_l2, [[1.0]])

  def test_apply_pool_with_mask_max(self):
    x = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = np.array([[True, True, False]])
    res = point._apply_pool(x, 'max', mask=mask)
    np.testing.assert_allclose(res, [[3.0, 4.0]])

  def test_apply_pool_with_mask_mean(self):
    x = np.array([[[2.0, 4.0], [6.0, 8.0], [100.0, 200.0]]])
    mask = np.array([[True, True, False]])
    res = point._apply_pool(x, 'mean', mask=mask)
    np.testing.assert_allclose(res, [[4.0, 6.0]])

  def test_local_aggregation_radius(self):
    key = random.PRNGKey(0)
    feat = np.zeros([2, 32, 5])
    coord = np.random.default_rng(42).random([2, 32, 3]).astype(np.float32)

    mlp_config = point.GroupMLPConfig(
        num_layers=1, features=16, pool_type='max', normalizer='layer'
    )
    la_config = point.LocalAggregationConfig(
        mlp_config=mlp_config, radius=0.5, max_k=8
    )
    model = point.LocalAggregation(la_config)
    params = model.init(key, feat, coord, feat, coord)
    res = model.apply(params, feat, coord, feat, coord)
    self.assertEqual(res.shape, (2, 32, 16))


if __name__ == '__main__':
  absltest.main()
