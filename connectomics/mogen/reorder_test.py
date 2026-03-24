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
import jax
import jax.numpy as jnp
import numpy as np

from connectomics.mogen import reorder
from absl.testing import absltest


class ReorderTest(absltest.TestCase):

  def test_reorder_named_z_recursive(self):
    key = jax.random.PRNGKey(0)
    pc = jax.random.uniform(key, (1, 8, 3))

    norms = jnp.linalg.norm(pc, axis=2)
    min_norm_idx = jnp.argmin(norms, axis=1)[0]
    max_norm_idx = jnp.argmax(norms, axis=1)[0]
    closest_point = pc[0, min_norm_idx]
    farthest_point = pc[0, max_norm_idx]

    #  z_recursive_min: FPS pre-sort places the closest point in the first two
    reordered_min = reorder.reorder_named(pc, 'z_recursive_min')
    is_closest_at_0 = np.allclose(reordered_min[0, 0], closest_point, atol=1e-6)
    is_closest_at_1 = np.allclose(reordered_min[0, 1], closest_point, atol=1e-6)
    self.assertTrue(
        is_closest_at_0 or is_closest_at_1,
        'Closest point not in the first two positions for z_recursive_min',
    )

    # z_recursive_max: FPS pre-sort places the farthest point in the first two
    reordered_max = reorder.reorder_named(pc, 'z_recursive_max')
    is_farthest_at_0 = np.allclose(
        reordered_max[0, 0], farthest_point, atol=1e-6
    )
    is_farthest_at_1 = np.allclose(
        reordered_max[0, 1], farthest_point, atol=1e-6
    )
    self.assertTrue(
        is_farthest_at_0 or is_farthest_at_1,
        'Farthest point not in the first two positions for z_recursive_max',
    )

  def test_reorder_z_sfc_with_reverse(self):
    key = jax.random.PRNGKey(0)
    pc = jax.random.uniform(key, (1, 8, 3))

    reordered_pc = reorder.reorder_z_sfc(pc)
    reordered_pc_with_reverse, reverse_indices = (
        reorder.reorder_z_sfc_with_reverse(pc)
    )

    np.testing.assert_allclose(reordered_pc, reordered_pc_with_reverse)

    restored_pc = jnp.take_along_axis(
        reordered_pc_with_reverse,
        jnp.expand_dims(reverse_indices, axis=2),
        axis=1,
    )
    np.testing.assert_allclose(pc, restored_pc)


if __name__ == '__main__':
  absltest.main()
