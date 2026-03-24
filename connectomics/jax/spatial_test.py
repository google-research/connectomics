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
"""Tests for spatial routines."""

import jax
import jax.numpy as jnp
import numpy as np

from connectomics.jax import spatial
from absl.testing import absltest


class SpatialTest(absltest.TestCase):

  def test_pca_align_single_preserves_origin_and_distances(self):
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(42), 3)
    points = jax.random.uniform(key1, (32, 3))
    vec_feats = jax.random.normal(key2, (32, 3))
    scalar_feats = jax.random.normal(key3, (32, 2))
    feats = jnp.concatenate([vec_feats, scalar_feats], axis=-1)

    aligned, aligned_feats = spatial.pca_align_single(
        points, feats, num_vectors=1
    )

    np.testing.assert_allclose(
        jnp.linalg.norm(jnp.mean(aligned, axis=0)),
        jnp.linalg.norm(jnp.mean(points, axis=0)),
        atol=1e-5,
    )

    orig_dists = spatial.squared_distances(
        points[jnp.newaxis], points[jnp.newaxis]
    )
    aligned_dists = spatial.squared_distances(
        aligned[jnp.newaxis], aligned[jnp.newaxis]
    )
    np.testing.assert_allclose(aligned_dists, orig_dists, atol=1e-5)

    np.testing.assert_allclose(aligned_feats[:, 3:], scalar_feats, atol=1e-6)

    np.testing.assert_allclose(
        jnp.linalg.norm(aligned_feats[:, :3], axis=-1),
        jnp.linalg.norm(vec_feats, axis=-1),
        atol=1e-5,
    )
    self.assertFalse(jnp.allclose(aligned_feats[:, :3], vec_feats, atol=1e-5))

  def test_kdnn_output_matches_spatial_knn(self):
    pc_batch = jax.random.uniform(jax.random.PRNGKey(0), (2, 64, 3))
    _, kdnn_indices = spatial.kdnn(pc_batch, 5)
    spatial_indices = spatial.knn(pc_batch, pc_batch, k=5)
    self.assertTrue(
        jnp.array_equal(
            jnp.sort(kdnn_indices, axis=-1), jnp.sort(spatial_indices, axis=-1)
        )
    )

  def test_random_subsample_points(self):
    batch, n, num = 3, 32, 10
    key = jax.random.PRNGKey(7)
    points = jax.random.uniform(key, (batch, n, 3))

    sampled, sampled_idx = spatial.random_subsample_points(
        points, num, jax.random.PRNGKey(0)
    )

    # Output shapes.
    self.assertEqual(sampled.shape, (batch, num, 3))
    self.assertEqual(sampled_idx.shape, (batch, num))

    # Indices are valid (within [0, n)).
    self.assertTrue(jnp.all(sampled_idx >= 0))
    self.assertTrue(jnp.all(sampled_idx < n))

    # Indices are unique within each batch element (replace=False).
    for b in range(batch):
      self.assertLen(jnp.unique(sampled_idx[b]), num)

    # Sampled coords match input at the returned indices.
    for b in range(batch):
      np.testing.assert_array_equal(
          sampled[b], points[b, sampled_idx[b], :]
      )

    # Different keys give different samples.
    sampled2, _ = spatial.random_subsample_points(
        points, num, jax.random.PRNGKey(1)
    )
    self.assertFalse(jnp.array_equal(sampled, sampled2))

    # Same key is deterministic.
    sampled3, idx3 = spatial.random_subsample_points(
        points, num, jax.random.PRNGKey(0)
    )
    np.testing.assert_array_equal(sampled, sampled3)
    np.testing.assert_array_equal(sampled_idx, idx3)

  def test_ball_query_all_within_radius(self):
    points = jax.random.uniform(jax.random.PRNGKey(0), (2, 32, 3))
    idx, mask = spatial.ball_query(points, points, radius=100.0, max_k=8)
    self.assertEqual(idx.shape, (2, 32, 8))
    self.assertEqual(mask.shape, (2, 32, 8))
    self.assertTrue(jnp.all(mask))

  def test_ball_query_some_outside_radius(self):
    near = jnp.zeros((1, 4, 3))
    far = jnp.ones((1, 4, 3)) * 100.0
    points = jnp.concatenate([near, far], axis=1)  # [1, 8, 3]
    seed = jnp.zeros((1, 1, 3))
    _, mask = spatial.ball_query(seed, points, radius=1.0, max_k=8)
    self.assertEqual(mask.shape, (1, 1, 8))
    self.assertEqual(int(jnp.sum(mask)), 4)


if __name__ == "__main__":
  absltest.main()
