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
"""Minimal network utils to avoid scenic dependency.

Copied from
scenic/projects/modified_simple_diffusion/model.py
"""

import jax.numpy as jnp


def get_timestep_embedding(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    max_time: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
  """Builds sinusoidal embeddings for time.

  Args:
    timesteps: A 1-D array of timesteps.
    embedding_dim: The dimension of the embedding.
    max_time: The maximum time value.
    dtype: The data type of the embeddings.

  Returns:
    A 2-D array of shape `[len(timesteps), embedding_dim]`.
  """
  assert len(timesteps.shape) == 1
  timesteps *= 1000.0 / max_time
  half_dim = embedding_dim // 2
  emb = jnp.log(10000) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
  emb = timesteps[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jnp.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb
