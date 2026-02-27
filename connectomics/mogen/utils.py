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
"""General utilities for Mogen."""

import jax
import jax.numpy as jnp


def get_combine_map(
    n_samples_to_combine: int, n_total_points: int
) -> jax.Array | None:
  """Creates a map indicating which original sample each point belongs to.

  Args:
    n_samples_to_combine: Number of samples that were combined.
    n_total_points: Total number of points after combining samples. (Must be
      divisible by n_samples_to_combine)

  Returns:
    A JAX array of shape (1, n_total_points) where each element is an integer
    from 0 to n_samples_to_combine - 1, indicating the original sample index.
    Returns None if n_samples_to_combine is 1.
  """
  if n_samples_to_combine == 1:
    return None
  if n_total_points % n_samples_to_combine != 0:
    raise ValueError(
        f"Total number of points ({n_total_points}) must be divisible by "
        f"n_samples_to_combine ({n_samples_to_combine})."
    )
  points_per_sample = n_total_points // n_samples_to_combine
  point_map = jnp.arange(n_samples_to_combine).repeat(points_per_sample)
  return point_map[None, :]
