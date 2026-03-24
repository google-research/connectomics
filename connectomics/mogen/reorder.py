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
"""Reordering methods for point clouds."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import ott_jax as ott

from connectomics.jax import spatial


@jax.jit
def reorder_axis(pc: jax.Array, axis_index: int) -> jax.Array:
  """Reorders points in a batched point cloud along a given axis index.

  Args:
    pc: Batched point cloud of shape (..., point_cloud_size, feature_dimension).
    axis_index: Index of the feature dimension (last dim) to sort by.

  Returns:
    Reordered batched point cloud of the same shape.
  """
  keys_to_sort = pc[..., axis_index]
  argsort_indices = jnp.argsort(keys_to_sort, axis=-1)
  # jnp.argpartition into halves is ~2x slower

  expanded_indices = jnp.expand_dims(argsort_indices, axis=-1)
  return jnp.take_along_axis(pc, expanded_indices, axis=-2)


@jax.jit
def reorder_z_sfc(
    pc: jax.Array,
) -> jax.Array:
  """Reorders a point cloud using Z-order-like Space-Filling Curve sorting.

  Args:
    pc: Batched point cloud of shape (batch_size, point_cloud_size,
      feature_dimension). Assumes point_cloud_size is a power of 2.

  Returns:
    Reordered batched point cloud of the same shape.
  """
  depth = int(np.log2(pc.shape[1]))
  batch_size, point_cloud_size, feature_dim = pc.shape

  current_pc = pc
  for d in range(depth):
    num_sections = 2**d
    section_size = point_cloud_size // num_sections
    axis_to_sort = d % feature_dim
    reshaped_pc = current_pc.reshape(
        batch_size, num_sections, section_size, feature_dim
    )
    processed_sections = reorder_axis(reshaped_pc, axis_to_sort)

    current_pc = processed_sections.reshape(
        batch_size, point_cloud_size, feature_dim
    )
  assert pc.shape == current_pc.shape
  return current_pc


@jax.jit
def reorder_z_sfc_with_reverse(
    pc: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Reorders a point cloud using Z-order-like Space-Filling Curve sorting.

  Args:
    pc: Batched point cloud of shape (batch_size, point_cloud_size,
      feature_dimension). Assumes point_cloud_size is a power of 2.

  Returns:
    A tuple containing:
      - Reordered batched point cloud of the same shape.
      - Reverse indices: An array of shape (batch_size, point_cloud_size) such
        that `pc == reordered_pc[jnp.arange(batch_size)[:, None],
        reverse_indices]`.
  """
  batch_size, point_cloud_size, feature_dimension = pc.shape
  depth = int(np.log2(point_cloud_size))
  assert (
      2**depth == point_cloud_size
  ), f'point_cloud_size must be a power of 2, but is {point_cloud_size}'

  current_pc = pc
  current_indices = jnp.arange(point_cloud_size)
  current_indices = jnp.broadcast_to(
      current_indices, (batch_size, point_cloud_size)
  )

  for d in range(depth):
    num_sections = 2**d
    section_size = current_pc.shape[1] // num_sections
    axis_to_sort = d % feature_dimension
    reshaped_pc = current_pc.reshape(
        batch_size, num_sections, section_size, feature_dimension
    )
    reshaped_indices = current_indices.reshape(
        batch_size, num_sections, section_size
    )
    keys_to_sort = reshaped_pc[..., :, axis_to_sort]
    argsort_indices_sections = jnp.argsort(keys_to_sort, axis=-1)
    expanded_argsort_for_pc = jnp.expand_dims(argsort_indices_sections, axis=-1)
    reordered_pc_sections = jnp.take_along_axis(
        reshaped_pc,
        expanded_argsort_for_pc,
        axis=-2,
    )
    reordered_indices_sections = jnp.take_along_axis(
        reshaped_indices,
        argsort_indices_sections,
        axis=-1,
    )
    current_pc = reordered_pc_sections.reshape(
        batch_size, point_cloud_size, feature_dimension
    )
    current_indices = reordered_indices_sections.reshape(
        batch_size, point_cloud_size
    )
  assert pc.shape == current_pc.shape

  reverse_indices = jnp.argsort(current_indices, axis=1)

  return current_pc, reverse_indices


def reorder_distance_from_centroid(
    pc: jax.Array, reverse: bool = False
) -> jax.Array:
  """Reorders points based on distance from the point cloud centroid.

  Args:
    pc: Batched point cloud (batch_size, point_cloud_size, feature_dimension).
    reverse: If True, sort from farthest to closest.

  Returns:
    Reordered batched point cloud.
  """
  centroids = jnp.mean(pc, axis=1, keepdims=True)
  distances_sq = jnp.sum(jnp.square(pc - centroids), axis=2)
  argsort_indices = jnp.argsort(distances_sq, axis=1)
  return jnp.take_along_axis(
      pc, jnp.expand_dims(argsort_indices, axis=2), axis=1
  )[:, :: (-1 if reverse else 1)]


def reorder_distance_from_origin(
    pc: jax.Array, reverse: bool = False
) -> jax.Array:
  """Reorders points based on distance from the origin (0,0,0).

  Args:
    pc: Batched point cloud (batch_size, point_cloud_size, feature_dimension).
    reverse: If True, sort from farthest to closest.

  Returns:
    Reordered batched point cloud.
  """
  distances_sq = jnp.sum(jnp.square(pc), axis=2)
  argsort_indices = jnp.argsort(distances_sq, axis=1)
  return jnp.take_along_axis(
      pc, jnp.expand_dims(argsort_indices, axis=2), axis=1
  )[:, :: (-1 if reverse else 1)]


def reorder_pc_ot(
    pc_a: jax.Array, pc_b: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Reorders point clouds using optimal transport matching via Sinkhorn.

  Args:
    pc_a: First point cloud (point_cloud_size, feature_dimension).
    pc_b: Second point cloud (point_cloud_size, feature_dimension).

  Returns:
    Tuple of reordered point clouds
  """
  assert pc_a.shape == pc_b.shape
  geom = ott.geometry.pointcloud.PointCloud(pc_a, pc_b)
  ot = ott.solvers.linear.sinkhorn.solve(geom)
  ind_a, ind_b = optax.assignment.hungarian_algorithm(-ot.matrix)
  # TODO(riegerfr): get indices directly without using optax/hungarian,
  # then assert valid permutation
  return pc_a[ind_a], pc_b[ind_b]


vmap_ot = jax.vmap(reorder_pc_ot, in_axes=(0, 0), out_axes=(0, 0))


def reorder_origin_fps(
    pc: jax.Array, closest_to_origin: bool = False
) -> jax.Array:
  """Reorders points using FPS starting near/far from origin.

  Args:
    pc: Point cloud data. (shape: (batch_size, n_points, 3))
    closest_to_origin: Whether to order by closest to origin.

  Returns:
    Reordered point cloud.
  """
  norms = jnp.linalg.norm(pc, axis=2)
  max_norm_indices = (
      jnp.argmin(norms, axis=1)
      if closest_to_origin
      else jnp.argmax(norms, axis=1)
  )
  batch_indices = jnp.arange(pc.shape[0])

  max_norm_point = pc[batch_indices, max_norm_indices]
  first_point = pc[:, 0]

  pc = pc.at[:, 0].set(max_norm_point)
  pc = pc.at[batch_indices, max_norm_indices].set(first_point)
  pc = spatial.subsample_points(pc, pc.shape[1])[0]
  return pc


def reorder_named(coord: jax.Array, name: str) -> jax.Array:
  """Reorders points based on a named reordering method.

  Args:
    coord: Point cloud data. (shape: (batch_size, n_points, 3))
    name: Name of the reordering method.

  Returns:
    Reordered point cloud.

  Raises:
    ValueError: If the reordering method is not recognized.
  """
  if name == 'no':
    return coord
  base_method = name.split('_')[0]
  if name.endswith('_min'):
    max_dist = False
  else:
    max_dist = True
  if '_recursive' in name:
    recursive = True
  else:
    recursive = False
  if '_first_' in name:
    first_n = int(name.split('_first_')[-1].split('_')[0])
  else:
    first_n = 0
  return reorder(coord, base_method, max_dist, recursive, first_n)


def reorder(
    coord: jax.Array,
    base_method: str | None = None,
    max_dist: bool = True,
    recursive: bool = False,
    first_n: int = 0,
) -> jax.Array:
  """Reorders points based on a named reordering method.

  Args:
    coord: Point cloud data. (shape: (batch_size, n_points, 3))
    base_method: Name of the reordering method.
    max_dist: Whether to order by max distance.
    recursive: Whether to reorder recursively.
    first_n: Number of points to reorder first.

  Returns:
    Reordered point cloud.

  Raises:
    ValueError: If the reordering method is not recognized.
  """

  if base_method is None:
    return coord

  reorder_fn = {
      'z': reorder_z_sfc,
      # TODO(riegerfr): add hilbert
      'origin': reorder_distance_from_origin,
      'revorigin': lambda x: reorder_distance_from_origin(x, reverse=True),
      'centroid': reorder_distance_from_centroid,
      'revcentroid': lambda x: reorder_distance_from_centroid(x, reverse=True),
      'fps': lambda x: reorder_origin_fps(x, closest_to_origin=max_dist),
  }[base_method]

  assert not (first_n > 0 and recursive)

  if first_n > 0:
    assert base_method != 'fps', 'fps cannot be used with first_n'
    coord = reorder_origin_fps(coord, closest_to_origin=not max_dist)
    coord = jnp.concatenate(
        (
            reorder_fn(coord[:, :first_n]),
            reorder_fn(coord[:, first_n:]),
        ),
        axis=1,
    )
  elif recursive:
    assert base_method != 'fps', 'fps cannot be used with recursive'
    coord = reorder_origin_fps(coord, closest_to_origin=not max_dist)
    newcoord = jnp.concatenate(
        [
            reorder_fn(chunk)
            for chunk in jnp.split(
                coord,
                [
                    2 ** (i + 1)
                    for i in range(int(math.log2(coord.shape[1])) - 1)
                ],
                axis=1,
            )
        ],
        axis=1,
    )
    assert newcoord.shape == coord.shape
    coord = newcoord
  else:
    coord = reorder_fn(coord)

  return coord
