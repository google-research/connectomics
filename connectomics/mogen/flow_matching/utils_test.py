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

from connectomics.mogen.flow_matching import utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import synjax
from absl.testing import absltest


class UtilsTest(absltest.TestCase):

  def test_simple_embs(self):
    pc = jax.random.uniform(jax.random.PRNGKey(0), (8, 256, 3))
    embs = utils.simple_embs(pc, mst=True)
    self.assertEqual(embs.shape, (8, 10))

  def test_mst(self):
    n_nodes = 128
    adj = jax.random.uniform(
        jax.random.PRNGKey(0), (2, n_nodes, n_nodes), minval=0.1, maxval=1.0
    )
    adj = (adj + adj.transpose((0, 2, 1))) / 2
    adj = adj.at[..., jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0)

    mst_adj_prim = utils.prim_mst(adj)
    mst_adj_synjax = synjax.SpanningTreeCRF(
        -adj, directed=False, projective=False, single_root_edge=False
    ).argmax()  # slow
    self.assertTrue(jnp.allclose(mst_adj_prim, mst_adj_synjax))

  def test_moment_embedding(self):
    pc = jax.random.uniform(jax.random.PRNGKey(0), (8, 256, 3))
    embs = utils.moment_embedding(pc, n=4)
    self.assertEqual(embs.shape, (8, 15))

  def test_prep_data_combine_samples_pointinfinity(
      self,
  ):
    batch = {
        'coord': jnp.array(np.random.rand(4, 10, 3), dtype=jnp.float32),
        'feat': jnp.array(np.random.rand(4, 10, 2), dtype=jnp.float32),
        '_dataset_index': jnp.array(
            np.arange(4).reshape(4, 1), dtype=jnp.int32
        ),
    }

    config = ml_collections.ConfigDict()
    config.model_type = 'pointinfinity'
    config.use_feat = True
    config = ml_collections.FrozenConfigDict(config)

    n_combine_samples = 2
    n_points = 5
    coord, feat, cond = utils.prep_data(
        batch,
        n_combine_samples=n_combine_samples,
        n_points=n_points,
        coord_scale=1.0,
        feat_scale=1.0,
        cond_mode='moment',
        use_feat=config.use_feat,
    )

    self.assertEqual(coord.shape, (2, 10, 3))
    self.assertEqual(feat.shape, (2, 10, 2))
    self.assertEqual(cond.shape, (2, 15))

  def test_plot_point_clouds_with_combine_mask(
      self,
  ):
    pc = jax.random.uniform(jax.random.PRNGKey(0), (2, 10, 3))
    fig = utils.plot_point_clouds(pc, n_combine_samples=2)
    self.assertIsNotNone(fig)


if __name__ == '__main__':
  absltest.main()
