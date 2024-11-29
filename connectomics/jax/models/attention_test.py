# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Test vision attention modules."""

from absl.testing import absltest
from connectomics.jax.models import attention as attn
import einops
import jax


class AttentionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.features = 3
    self.rng = jax.random.PRNGKey(0)

  def test_base_attention(self):
    seq = jax.random.normal(self.rng, (self.batch_size, 128, self.features))
    model = attn.Attention(num_heads=2, qkv_features=64)
    variables = model.init(self.rng, seq)
    seq_out = model.apply(variables, seq, train=True)
    self.assertSequenceEqual(seq_out.shape, seq.shape)

  def test_attention_dropout(self):
    seq = jax.random.normal(self.rng, (self.batch_size, 128, self.features))
    model = attn.Attention(num_heads=2, qkv_features=64, dropout=0.2)
    variables = model.init(self.rng, seq)
    seq_out_determined = model.apply(variables, seq, train=False).mean()
    seq_out_dropout = model.apply(
        variables, seq, train=True, rngs={"dropout": self.rng}
    ).mean()
    self.assertNotEqual(seq_out_dropout, seq_out_determined)

  def test_voxel_attention(self):
    x = jax.random.normal(self.rng, (self.batch_size, 4, 2, self.features))
    model = attn.VoxelAttention(num_heads=2, qkv_features=4)
    variables = model.init(self.rng, x)
    x_out = model.apply(variables, x)
    self.assertSequenceEqual(x_out.shape, x.shape)

  def test_patch_attention(self):
    x = jax.random.normal(self.rng, (self.batch_size, 8, 4, 2, self.features))
    ps = (2, 2, 2)
    model = attn.PatchAttention(num_heads=2, qkv_features=4, patch_sizes=ps)
    variables = model.init(self.rng, x)
    x_out = model.apply(variables, x)
    self.assertSequenceEqual(x_out.shape, x.shape)

  def test_grid_attention(self):
    x = jax.random.normal(self.rng, (self.batch_size, 8, 4, 2, self.features))
    ps = (2, 2, 2)
    model = attn.GridAttention(num_heads=2, qkv_features=4, patch_sizes=ps)
    variables = model.init(self.rng, x)
    x_out = model.apply(variables, x)
    self.assertSequenceEqual(x_out.shape, x.shape)

  def test_block_attention(self):
    x = jax.random.normal(self.rng, (self.batch_size, 8, 4, 2, self.features))
    ps = (2, 2, 2)
    model = attn.BlockAttention(num_heads=2, qkv_features=4, patch_sizes=ps)
    variables = model.init(self.rng, x)
    x_out = model.apply(variables, x)
    self.assertSequenceEqual(x_out.shape, x.shape)

  def test_grid_non_uniform_patches(self):
    x = jax.random.normal(self.rng, (self.batch_size, 1, 8, 2, self.features))
    ps = (1, 4, 2)
    model = attn.GridAttention(num_heads=2, qkv_features=4, patch_sizes=ps)
    variables = model.init(self.rng, x)
    x_out = model.apply(variables, x)
    self.assertSequenceEqual(x_out.shape, x.shape)

  def test_non_divisible_patches_fail(self):
    x = jax.random.normal(self.rng, (self.batch_size, 1, 8, 2, self.features))
    ps = (1, 3, 2)
    model = attn.GridAttention(num_heads=2, qkv_features=4, patch_sizes=ps)
    with self.assertRaises(einops.EinopsError):
      model.init(self.rng, x)


if __name__ == "__main__":
  absltest.main()
