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
"""Gemma-style Point Infinity block."""

from typing import Any
from connectomics.mogen.models.pointinfinity import PointInfinityConfig
import flax.linen as nn
import jax
import jax.numpy as jnp


class GemmaCrossAttention(nn.Module):
  """Cross-attention with Gemma-style QK-norm and soft-capping."""

  num_heads: int
  attn_logits_soft_cap: float | None = None
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x_q, x_kv, mask=None):
    b, l_q, d_q = x_q.shape
    _, l_kv, _ = x_kv.shape

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
      cap = self.attn_logits_soft_cap
      logits = jnp.tanh(logits / cap) * cap

    if mask is not None:
      logits = jnp.where(mask, logits, -1e9)

    probs = jax.nn.softmax(logits, axis=-1)

    out = jnp.einsum('bqhk,bkhd->bqhd', probs, v)
    out = out.reshape(b, l_q, d_q)

    out = nn.Dense(d_q, name='out', dtype=self.dtype)(out)
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
