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
"""Point Infinity model, see https://arxiv.org/pdf/2404.03566."""

from typing import Any
from connectomics.jax import spatial
from connectomics.jax.models import point
from connectomics.mogen import reorder
from connectomics.mogen import utils
from connectomics.mogen.models import minimal_network_utils
from connectomics.mogen.models import minimal_vit as vit
from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp


@struct.dataclass
class PointInfinityConfig:
  """Configuration for the Point Infinity model."""

  point_dim: int = 256
  latent_dim: int = 512
  n_latents: int = 256
  n_blocks: int = 6
  n_subblocks: int = 4
  n_heads: int = 8
  k_nn: int = 16
  dropout: float = 0.0
  combine_z: int = 1
  n_combine_samples: int = 1
  out_dim: int | None = None
  remat: bool = False
  dtype: Any = jnp.float32
  use_gemma: bool = False


class PointInfinityBlock(nn.Module):
  """Point Infinity block."""

  config: PointInfinityConfig

  @nn.compact
  def __call__(
      self,
      x,
      z,
  ):
    # TODO(riegerfr): ablate norms + their initialization
    z = z + vit.MultiHeadDotProductAttention(
        num_heads=self.config.n_heads,
        normalize_qk=True,
        dtype=self.config.dtype,
    )(
        nn.RMSNorm(dtype=self.config.dtype)(z),
        nn.RMSNorm(dtype=self.config.dtype)(x),
    )
    z = z + (
        vit.MlpBlock(
            dropout=self.config.dropout,
            dtype=self.config.dtype,
        )(nn.RMSNorm(dtype=self.config.dtype)(z), None, True)
    )

    for _ in range(self.config.n_subblocks):
      z_normed = nn.RMSNorm(dtype=self.config.dtype)(z)
      z = z + vit.MultiHeadDotProductAttention(
          num_heads=self.config.n_heads,
          normalize_qk=True,
          dtype=self.config.dtype,
      )(
          z_normed,
          z_normed,
      )
      z = z + (
          vit.MlpBlock(dropout=self.config.dropout, dtype=self.config.dtype)(
              nn.RMSNorm(dtype=self.config.dtype)(z), None, True
          )
      )

    x = x + vit.MultiHeadDotProductAttention(
        num_heads=self.config.n_heads,
        normalize_qk=True,
        dtype=self.config.dtype,
    )(
        nn.RMSNorm(dtype=self.config.dtype)(x),
        nn.RMSNorm(dtype=self.config.dtype)(z),
    )
    x = x + (
        vit.MlpBlock(dropout=self.config.dropout, dtype=self.config.dtype)(
            nn.RMSNorm(dtype=self.config.dtype)(x), None, True
        )
    )
    return x, z


class PointInfinity(nn.Module):
  """Point Infinity model."""

  config: PointInfinityConfig

  @nn.compact
  def __call__(
      self,
      coord: jax.Array,
      feat: jax.Array | None = None,
      t: jax.Array | None = None,
      cond: jax.Array | None = None,
      deterministic: bool = True,
      point_cond_mask: jax.Array | None = None,
  ) -> jax.Array:
    if t is not None:
      t_emb = vit.MlpBlock(
          dropout=self.config.dropout, dtype=self.config.dtype
      )(
          minimal_network_utils.get_timestep_embedding(
              t,
              embedding_dim=self.config.latent_dim,
              max_time=1.0,
          )[:, None, :],
          None,
          True,
      )[
          :, 0, :
      ]
    else:
      t_emb = None
    cond = (
        nn.Dense(self.config.latent_dim, dtype=self.config.dtype)(cond)
        if cond is not None
        else None
    )

    if self.config.combine_z > 1:
      if feat is not None:
        raise NotImplementedError(
            'Features and combine_z > 1 not implemented yet.'
        )
      coord, reverse_indices = reorder.reorder_z_sfc_with_reverse(coord)
      if self.config.n_combine_samples > 1:
        raise NotImplementedError('Combine mask not implemented yet.')
    else:
      reverse_indices = None

    x = coord
    if self.config.n_combine_samples > 1:
      if feat is not None:
        raise NotImplementedError(
            'Features and combine_samples > 1 not implemented yet.'
        )
      combine_map = utils.get_combine_map(
          self.config.n_combine_samples, coord.shape[1]
      )
      combine_map = combine_map[..., None]
      combine_map = jnp.repeat(combine_map, x.shape[0], axis=0)
      x = jnp.concatenate((x, combine_map), axis=-1)

    if point_cond_mask is not None:
      assert self.config.n_combine_samples == 1
      assert point_cond_mask.shape[1] == coord.shape[1]
      assert point_cond_mask.shape[0] == coord.shape[0]
      assert self.config.combine_z == 1
      x = jnp.concatenate((x, point_cond_mask[:, :, None]), axis=-1)

    if self.config.k_nn > 0:
      if (
          coord.shape[1] > 8192
      ):  # TODO(riegerfr): Consider a common wrapper function for dispatching

        _, k_nn_idx = spatial.kdnn(coord, self.config.k_nn + 1)
        k_nn_idx = k_nn_idx[..., 1:]  # Remove the point itself
      else:
        k_nn_idx = spatial.knn(coord, coord, self.config.k_nn)
      k_nn_coord = (
          point.batch_lookup(coord, k_nn_idx) - coord[:, :, None]
      ).reshape(coord.shape[0], coord.shape[1], -1)
      x = jnp.concatenate((x, k_nn_coord), axis=-1)

    if self.config.combine_z > 1:
      x = x.reshape(
          x.shape[0],
          x.shape[1] // self.config.combine_z,
          x.shape[2] * self.config.combine_z,
      )

    if feat is not None:
      x = jnp.concatenate((x, feat), axis=-1)

    x = nn.Dense(self.config.point_dim, dtype=self.config.dtype)(x)

    z = jnp.repeat(
        nn.Embed(
            self.config.n_latents,
            self.config.latent_dim,
            dtype=self.config.dtype,
        )(  # TODO(riegerfr): scale init
            jnp.arange(self.config.n_latents)
        )[
            None
        ],
        coord.shape[0],
        axis=0,
    )
    z = jnp.concat((z, t_emb[:, None, :]), axis=1) if t_emb is not None else z
    z = jnp.concat((z, cond[:, None, :]), axis=1) if cond is not None else z

    remat_fn = nn.remat if self.config.remat else lambda x: x
    for _ in range(self.config.n_blocks):
      if self.config.use_gemma:
        x, z = remat_fn(GemmaPointInfinityBlock)(self.config)(x, z)
      else:
        x, z = remat_fn(PointInfinityBlock)(self.config)(x, z)

    x = nn.Dense(
        (
            (coord.shape[-1] + (feat.shape[-1] if feat is not None else 0))
            if self.config.out_dim is None
            else self.config.out_dim
        )
        * self.config.combine_z,
        dtype=self.config.dtype,
    )(x)

    if self.config.combine_z > 1:
      x = x.reshape(x.shape[0], coord.shape[1], -1)
      x = x[
          jnp.arange(x.shape[0])[:, None],
          reverse_indices,
      ]

    return x

# Note: These classes are duplicated from gemma_pointinfinity.py to avoid
# circular dependency.
class GemmaCrossAttention(nn.Module):
  """Cross-attention with Gemma-style QK-norm and soft-capping."""

  num_heads: int
  attn_logits_soft_cap: float | None = None
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x_q, x_kv, mask=None):
    b, l_q, d_q = x_q.shape
    _, l_kv, d_kv = x_kv.shape

    head_dim = d_q // self.num_heads
    assert d_q % self.num_heads == 0

    # Projections to query dimension
    q = nn.Dense(d_q, name='query', dtype=self.dtype)(x_q)
    k = nn.Dense(d_q, name='key', dtype=self.dtype)(x_kv)
    v = nn.Dense(d_q, name='value', dtype=self.dtype)(x_kv)

    # Reshape for multi-head
    q = q.reshape(b, l_q, self.num_heads, head_dim)
    k = k.reshape(b, l_kv, self.num_heads, head_dim)
    v = v.reshape(b, l_kv, self.num_heads, head_dim)

    # Gemma style QK Norm
    q = nn.RMSNorm(name='query_norm', dtype=self.dtype)(q)
    k = nn.RMSNorm(name='key_norm', dtype=self.dtype)(k)

    # Compute logits
    logits = jnp.einsum('bqhd,bkhd->bqhk', q, k) / jnp.sqrt(head_dim)

    # Gemma style soft capping
    if self.attn_logits_soft_cap is not None:
      logits = (
          jnp.tanh(logits / self.attn_logits_soft_cap)
          * self.attn_logits_soft_cap
      )

    if mask is not None:
      logits = jnp.where(mask, logits, -1e9)

    probs = jax.nn.softmax(logits, axis=-1)

    out = jnp.einsum('bqhk,bkhd->bqhd', probs, v)
    out = out.reshape(b, l_q, d_q)

    out = nn.Dense(d_q, name='out')(out)
    return out


class GemmaFeedForward(nn.Module):
  """Feed-forward network with Gated GELU (GeGLU)."""

  hidden_dim: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    d = x.shape[-1]
    gate = nn.Dense(self.hidden_dim * 2, name='gate', dtype=self.dtype)(x)

    # Split and apply GELU
    gate_1, gate_2 = jnp.split(gate, 2, axis=-1)
    activations = nn.gelu(gate_1) * gate_2

    out = nn.Dense(d, name='out', dtype=self.dtype)(activations)
    return out


class GemmaPointInfinityBlock(nn.Module):
  """Point Infinity block with Gemma-style internals."""

  config: PointInfinityConfig
  attn_logits_soft_cap: float | None = 30.0
  use_skip_scale: bool = True

  @nn.compact
  def __call__(self, x, z):
    if self.use_skip_scale:
      skip_scale = self.param('skip_scale', nn.initializers.ones, (1,))
    else:
      skip_scale = 1.0

    # --- Step 1: Latents 'z' attend to Points 'x' (Cross-Attention) ---
    z_norm = nn.RMSNorm(dtype=self.config.dtype)(z)
    x_norm = nn.RMSNorm(dtype=self.config.dtype)(x)

    z = z + GemmaCrossAttention(
        num_heads=self.config.n_heads,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        dtype=self.config.dtype,
        name='cross_attn_z_to_x',
    )(z_norm, x_norm)

    z = z + GemmaFeedForward(
        hidden_dim=z.shape[-1] * 4, dtype=self.config.dtype, name='ffn_z'
    )(nn.RMSNorm(dtype=self.config.dtype)(z))

    # --- Step 2: Self-Attention among Latents 'z' ---
    for i in range(self.config.n_subblocks):
      z_norm = nn.RMSNorm(dtype=self.config.dtype)(z)
      z = z + GemmaCrossAttention(
          num_heads=self.config.n_heads,
          attn_logits_soft_cap=self.attn_logits_soft_cap,
          dtype=self.config.dtype,
          name=f'self_attn_z_{i}',
      )(z_norm, z_norm)

      z = z + GemmaFeedForward(
          hidden_dim=z.shape[-1] * 4, dtype=self.config.dtype, name=f'ffn_z_{i}'
      )(nn.RMSNorm(dtype=self.config.dtype)(z))

    # --- Step 3: Points 'x' attend to Latents 'z' (Cross-Attention) ---
    x_norm = nn.RMSNorm(dtype=self.config.dtype)(x)
    z_norm = nn.RMSNorm(dtype=self.config.dtype)(z)

    x = x + GemmaCrossAttention(
        num_heads=self.config.n_heads,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        dtype=self.config.dtype,
        name='cross_attn_x_to_z',
    )(x_norm, z_norm)

    x = x + GemmaFeedForward(
        hidden_dim=x.shape[-1] * 4, dtype=self.config.dtype, name='ffn_x'
    )(nn.RMSNorm(dtype=self.config.dtype)(x))

    return x * skip_scale, z * skip_scale
