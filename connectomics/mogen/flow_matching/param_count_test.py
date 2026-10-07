# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
from absl.testing import absltest
from connectomics.mogen.flow_matching import utils
import jax
import jax.numpy as jnp
import ml_collections


def _make_config(**kwargs) -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict({
      'model_type': 'pointinfinity',
      'pfty_point_dim': 128,
      'pfty_latent_dim': 256,
      'pfty_n_latents': 256,
      'pfty_n_blocks': 4,
      'pfty_n_subblocks': 2,
      'pfty_n_heads': 8,
      'pfty_k_nn': 16,
      'pfty_dropout': 0.0,
      'combine_z': 0,
      'use_feat': False,
      'cond_mode': 'no',
      'use_remat': False,
      'use_bf16': False,
  })
  config.update(kwargs)
  return config


class ParamCountTest(absltest.TestCase):

  def count_params(self, cfg_updates, name):
    config = _make_config(**cfg_updates)

    B = 1
    N = 256
    coord = jnp.zeros((B, N, 3), dtype=jnp.float32)
    feat = jnp.zeros((B, N, 11), dtype=jnp.float32) if config.use_feat else None
    cond = (
        jnp.zeros((B, 22), dtype=jnp.float32)
        if config.cond_mode != 'no'
        else None
    )

    rng = jax.random.PRNGKey(0)

    shapes = jax.eval_shape(
        lambda: utils.get_model(  # pylint: disable=g-long-lambda
            rng,
            init_coord=coord,
            init_feat=feat,
            config=config,
            cond=cond,
        )[1]
    )

    total = sum(x.size for x in jax.tree_util.tree_leaves(shapes))
    print(f'PARAM COUNT {name}: {total / 1e6:.2f}M params')
    self.assertGreater(total, 0)
    return total

  def test_counts(self):
    c_ctrl = self.count_params(
        {
            'pfty_n_blocks': 4,
            'pfty_latent_dim': 256,
            'pfty_point_dim': 128,
            'pfty_n_heads': 8,
        },
        'Control',
    )
    c_3x = self.count_params(
        {
            'pfty_n_blocks': 12,
            'pfty_latent_dim': 256,
            'pfty_point_dim': 128,
            'pfty_n_heads': 8,
        },
        '3x Params',
    )
    c_width = self.count_params(
        {
            'pfty_n_blocks': 4,
            'pfty_latent_dim': 512,
            'pfty_point_dim': 256,
            'pfty_n_heads': 16,
        },
        'Width Arm',
    )
    self.assertGreater(c_3x, c_ctrl)
    self.assertGreater(c_width, c_ctrl)


if __name__ == '__main__':
  absltest.main()

