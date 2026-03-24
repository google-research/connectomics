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
"""Routines for processing spatial data (point clouds)."""

import functools
import typing
from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxkd import tree as jaxkd_tree


def subsample_points(
    points: ArrayLike, num: int, key: flax.typing.PRNGKey | None = None
) -> tuple[jax.Array, jax.Array]:
  """Farthest point subsampling.

  Args:
    points: [b, n, 3] point coordinates
    num: number of points to sample
    key: PRNG key

  Returns:
    [b, num, 3] sampled point coordinates, [b, num] sampled point indices
  """
  points = jnp.asarray(points)

  batch = points.shape[0]
  sampled = jnp.full((batch, num, 3), jnp.nan)
  sampled_idx = jnp.zeros((batch, num), dtype=int)

  if key is not None:
    idx = jax.random.randint(key, (batch,), 0, points.shape[-2], dtype=int)
    sampled_idx = sampled_idx.at[:, 0].set(idx)
    sampled = sampled.at[:, 0, :].set(points[jnp.arange(batch), idx, :])
  else:
    sampled = sampled.at[:, 0, :].set(points[:, 0, :])

  min_dist = jnp.full((batch, points.shape[1]), jnp.inf)

  def _find_point(i, v):
    sampled, sampled_idx, min_dist = v
    # linalg.norm is noticeably slower here
    dist = points - jax.lax.dynamic_slice(
        sampled, (0, i - 1, 0), (batch, 1, 3)
    )  # sampled[:, i-1:i, :]
    dist = (dist**2).sum(axis=-1)

    min_dist = jnp.minimum(min_dist, dist)
    idx = jnp.argmax(min_dist, axis=-1)

    sampled = sampled.at[:, i, :].set(points[jnp.arange(batch), idx, :])
    sampled_idx = sampled_idx.at[:, i].set(idx)
    return sampled, sampled_idx, min_dist

  sampled, sampled_idx, _ = jax.lax.fori_loop(
      1, num, _find_point, (sampled, sampled_idx, min_dist)
  )
  return sampled, sampled_idx


def squared_distances(a: ArrayLike, b: ArrayLike) -> jax.Array:
  """Returns the squared distances between all point pairs.

  Args:
    a: [b, n, 3] 1st set of points
    b: [b, m, 3] 2nd set of points

  Returns:
    [b, n, m] squared distances
  """
  a = jnp.asarray(a)
  b = jnp.asarray(b)

  return ((a[..., jnp.newaxis, :] - b[..., jnp.newaxis, :, :]) ** 2).sum(
      axis=-1
  )


def pca_align_single(
    points: ArrayLike, feats: ArrayLike, num_vectors: int = 1
) -> tuple[jax.Array, jax.Array]:
  """Aligns a point cloud to its principal axes.

  Variance is maximal along X and minimal along Z. Preserves chirality.

  Args:
    points: [n, 3] point coordinates
    feats: [n, f] point features
    num_vectors: number of vector features, stored in the first 3 * num_vectors
      elements of feats

  Returns:
    points, feats after alignment
  """

  centered = points - jnp.mean(points, axis=0)
  cov = jnp.dot(centered.T, centered)
  _, eigvecs = jnp.linalg.eigh(cov)

  # Invert order so that primary component is along X.
  rot_mtx = eigvecs[:, ::-1]

  projected = jnp.dot(centered, rot_mtx)

  # Ensure mass skews in the positive direction.
  flips = jnp.sign(jnp.sum(projected**3, axis=0))

  # Handle case where sign is 0 (perfect symmetry) by defaulting to 1.
  flips = jnp.where(flips == 0, 1.0, flips)

  rot_mtx = rot_mtx * flips

  # Ensure the transform is a true rotation (no mirroring).
  det = jnp.linalg.det(rot_mtx)
  rot_mtx = rot_mtx * jnp.array([1.0, 1.0, jnp.sign(det)])

  # Apply rotation to the original points so that the origin is unchanged.
  aligned = jnp.dot(points, rot_mtx)

  feats_scalar = feats[:, num_vectors * 3 :]
  feats_vec = feats[:, : num_vectors * 3].reshape(-1, num_vectors, 3)
  feats_vec = jnp.dot(feats_vec, rot_mtx).reshape(-1, num_vectors * 3)

  aligned_feats = jnp.concatenate([feats_vec, feats_scalar], axis=-1)
  return aligned, aligned_feats


pca_align = jax.jit(
    jax.vmap(pca_align_single, in_axes=(0, 0, None)), static_argnums=(2,)
)


def knn(seed: ArrayLike, points: ArrayLike, k: int) -> jax.Array:
  """Returns the nearest neighbors for every point.

  Args:
    seed: [b, n, 3] points for which to look for neighbors
    points: [b, m, 3] points among which to look for neighbors
    k: number of neighbors to look for

  Returns:
    [b, n, k] array of neighbor indieces in 'points'
  """
  seed = jnp.asarray(seed)
  points = jnp.asarray(points)

  # Ensure the input set is large enough to sample k neighbors. If not,
  # duplicate an arbitary point so that it is.
  missing = k - points.shape[-2]
  if missing > 0:
    points = jnp.concatenate(
        [points, jnp.repeat(points[:, 0:1, :], missing, axis=-2)], axis=-2
    )

  dist = squared_distances(seed, points)
  return jax.lax.top_k(-dist, k)[1]


def ball_query(
    seed: ArrayLike, points: ArrayLike, radius: float, max_k: int
) -> tuple[jax.Array, jax.Array]:
  """Returns neighbors within a radius for every point.

  Uses kNN to find max_k candidates, then masks by distance.

  Args:
    seed: [b, n, 3] points for which to look for neighbors
    points: [b, m, 3] points among which to look for neighbors
    radius: maximum distance for a point to be considered a neighbor
    max_k: maximum number of neighbors to return

  Returns:
    [b, n, max_k] array of neighbor indices in 'points'
    [b, n, max_k] boolean mask, True for valid neighbors within radius
  """
  idx = knn(seed, points, max_k)
  k_coord = _batch_lookup(jnp.asarray(points), idx)
  sq_dist = jnp.sum(
      (k_coord - jnp.asarray(seed)[..., None, :]) ** 2, axis=-1
  )
  mask = sq_dist <= radius**2
  return idx, mask


def _batch_lookup(x: jax.Array, idx: jax.Array) -> jax.Array:
  b = x.shape[0]
  n = idx.shape[-2] * idx.shape[-1]
  batch = jnp.repeat(jnp.arange(b), n).reshape(idx.shape)
  return x[batch, idx, :]


class CacheableModule(nn.Module):
  """Interface for operations that can be computed once and cached."""

  def _cached_result_or_none(self):
    if self.has_variable('cache', '__call__'):
      logging.info('Using cached results for %r', self.scope.path_text)
      v = self.get_variable('cache', '__call__')['0']
      if isinstance(v, dict):
        return tuple(v.values())
      else:
        return v


class KNN(CacheableModule):
  """Cacheable K nearest neigbors."""

  k: int

  @nn.compact
  def __call__(self, seed: ArrayLike, points: ArrayLike) -> jax.Array:
    ret = self._cached_result_or_none()
    if ret is not None:
      return typing.cast(jax.Array, ret)
    return knn(seed, points, self.k)


class FPS(CacheableModule):
  """Cacheable farthest point sampling."""

  num_points: int
  random_seed_node: bool

  @nn.compact
  def __call__(self, points: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Applies farthest point sampling to a point set.

    Args:
      points: [b, n, 3] point coordinates

    Returns:
      [b, num, 3] sampled point coordinates, [b, num] sampled point indices
    """
    ret = self._cached_result_or_none()
    if ret is not None:
      return ret

    if self.random_seed_node:
      rng = self.make_rng('dropout')
    else:
      rng = None
    return subsample_points(points, self.num_points, key=rng)


def random_subsample_points(
    points: ArrayLike, num: int, key: flax.typing.PRNGKey
) -> tuple[jax.Array, jax.Array]:
  """Random uniform point subsampling.

  Args:
    points: [b, n, 3] point coordinates
    num: number of points to sample
    key: PRNG key

  Returns:
    [b, num, 3] sampled point coordinates, [b, num] sampled point indices
  """
  points = jnp.asarray(points)
  batch = points.shape[0]

  def _sample_single(key, pts):
    idx = jax.random.choice(key, pts.shape[0], shape=(num,), replace=False)
    return pts[idx], idx

  keys = jax.random.split(key, batch)
  sampled, sampled_idx = jax.vmap(_sample_single)(keys, points)
  return sampled, sampled_idx


@functools.partial(jax.jit, static_argnames=('k',))
def kdnn_single(pc_single: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
  """Gets the k nearest neighbors for a single point cloud using jaxkd.

  Args:
    pc_single: A JAX array representing a single point cloud with shape (N, d).
    k: An integer.

  Returns:
    A tuple containing the distances and indices of the neighbors.
  """
  tree = jaxkd_tree.build_tree(pc_single)
  idx, dists = jaxkd_tree.query_neighbors(tree, pc_single, k=k)
  return dists, idx


kdnn = jax.vmap(kdnn_single, in_axes=(0, None))
"""Gets the k nearest neighbors for a batch of point clouds using jaxkd.

  Args:
    pc_batch: A JAX array representing a batch of point clouds with shape
      (B, N, d).
    k: An integer.

  Returns:
    A tuple containing the distances and indices of the neighbors for each
    point cloud in the batch.
  """
