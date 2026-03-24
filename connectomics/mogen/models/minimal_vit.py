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
"""Minimal ViT components to avoid scenic dependency.

Copied from
scenic/projects/modified_simple_diffusion/vit.py
"""

import functools
from typing import Any, Callable, Optional, Tuple, overload
import warnings

import flax.linen as nn
from flax.linen import initializers
from flax.linen.attention import dot_product_attention
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.linear import DotGeneralT
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
from jax import lax
import jax.numpy as jnp


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention."""

  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  out_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.zeros_init()
  )
  use_bias: bool = True
  attention_fn: Callable[..., Array] = dot_product_attention
  decode: bool = False
  normalize_qk: bool = False
  rope: bool = False
  qkv_dot_general: DotGeneralT = lax.dot_general
  out_dot_general: DotGeneralT = lax.dot_general
  qkv_dot_general_cls: Any = None
  out_dot_general_cls: Any = None

  @overload
  def __call__(
      self,
      inputs_q: Array,
      inputs_k: Optional[Array] = None,
      inputs_v: Optional[Array] = None,
      *,
      mask: Optional[Array] = None,
      deterministic: Optional[bool] = None,
  ):
    ...

  @overload
  def __call__(
      self,
      inputs_q: Array,
      *,
      inputs_kv: Array = None,
      mask: Optional[Array] = None,
      deterministic: Optional[bool] = None,
  ):
    ...

  @compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_k: Optional[Array] = None,
      inputs_v: Optional[Array] = None,
      *,
      inputs_kv: Optional[Array] = None,
      mask: Optional[Array] = None,
      deterministic: Optional[bool] = None,
  ):
    if inputs_kv is not None:
      if inputs_k is not None or inputs_v is not None:
        raise ValueError(
            'If either `inputs_k` or `inputs_v` is not None, `inputs_kv` must'
            ' be None. If `inputs_kv` is not None, both `inputs_k` and'
            ' `inputs_v` must be None. We recommend using `inputs_k` and'
            ' `inputs_v` args, since `inputs_kv` will be deprecated soon. See'
            ' https://github.com/google/flax/discussions/3389 for more'
            ' information.'
        )
      inputs_k = inputs_v = inputs_kv
      warnings.warn(
          'The inputs_kv arg will be deprecated soon. '
          'Use inputs_k and inputs_v instead. See '
          'https://github.com/google/flax/discussions/3389 '
          'for more information.',
          DeprecationWarning,
      )
    else:
      if inputs_k is None:
        if inputs_v is not None:
          raise ValueError(
              '`inputs_k` cannot be None if `inputs_v` is not None. To have'
              ' both `inputs_k` and `inputs_v` be the same value, pass in the'
              ' value to `inputs_k` and leave `inputs_v` as None.'
          )
        inputs_k = inputs_q
      if inputs_v is None:
        inputs_v = inputs_k
      elif inputs_v.shape[-1] == inputs_v.shape[-2]:
        warnings.warn(
            f'You are passing an array of shape {inputs_v.shape} '
            'to the `inputs_v` arg, when you may have intended '
            'to pass it to the `mask` arg. As of Flax version '
            '0.7.4, the function signature of '
            "MultiHeadDotProductAttention's `__call__` method "
            'has changed to `__call__(inputs_q, inputs_k=None, '
            'inputs_v=None, *, inputs_kv=None, mask=None, '
            'deterministic=None)`. Use the kwarg `mask` instead. '
            'See https://github.com/google/flax/discussions/3389 '
            'and read the docstring for more information.',
            DeprecationWarning,
        )

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        f'Memory dimension ({qkv_features}) must be divisible by number of'
        f' heads ({self.num_heads}).'
    )
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        DenseGeneral,
        axis=-1,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
        dot_general=self.qkv_dot_general,
        dot_general_cls=self.qkv_dot_general_cls,
    )
    query, key, value = (
        dense(name='query')(inputs_q),
        dense(name='key')(inputs_k),
        dense(name='value')(inputs_v),
    )

    if self.normalize_qk:
      query = RMSNormWithBias(name='query_ln')(query)
      key = RMSNormWithBias(name='key_ln')(key)

    dropout_rng = None
    if self.dropout_rate > 0.0:
      m_deterministic = merge_param(
          'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    if self.rope:
      num_pos = int(jnp.sqrt(query.shape[1]))
      x_pos = jnp.arange(0, num_pos)
      y_pos = jnp.arange(0, num_pos)
      x_pos, y_pos = jnp.meshgrid(x_pos, y_pos)
      x_pos = jnp.reshape(x_pos, -1).astype(jnp.float32)
      y_pos = jnp.reshape(y_pos, -1).astype(jnp.float32)

      query = rope(query, [x_pos, y_pos])
      key = rope(key, [x_pos, y_pos])

    x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
    )
    out = DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.out_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        dot_general=self.out_dot_general,
        dot_general_cls=self.out_dot_general_cls,
        name='out',
    )(x)
    return out


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: Optional[int] = None
  dropout: float = 0.0
  dtype: str = 'float32'

  @nn.compact
  def __call__(self, x, cond, deterministic=True):
    _, _, d = x.shape
    x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype)(x)
    x = nn.gelu(x)

    if cond is not None:
      scale = nn.Dense(features=x.shape[-1], use_bias=True, dtype=self.dtype)(
          cond
      )
      shift = nn.Dense(features=x.shape[-1], use_bias=True, dtype=self.dtype)(
          cond
      )
      x *= scale + 1.0
      x += shift

    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, dtype=self.dtype, kernel_init=nn.initializers.zeros)(x)
    return x


class RMSNormWithBias(nn.Module):
  """RMSNorm with learnable bias."""

  @nn.compact
  def __call__(self, x):
    x = nn.RMSNorm()(x)
    bias = self.param(
        'param', nn.initializers.zeros_init(), x.shape[-1], jnp.float32
    )
    return x + bias


def rope(
    x: Array, positions: list[Array], max_rotary_wavelength: int = 10_000
) -> Array:
  """Applies RoPE to x along the second axis."""
  assert x.shape[-1] % (2 * len(positions)) == 0
  num_ft = x.shape[-1] // len(positions)
  freq_exponents = (2.0 / num_ft) * jnp.arange(num_ft // 2)
  inv_freq = jnp.pi / (max_rotary_wavelength**freq_exponents)

  result = []
  xs = jnp.split(x, 2 * len(positions), axis=-1)
  for pos in positions:
    t = pos[..., None, None] * inv_freq
    sin, cos = jnp.sin(t), jnp.cos(t)
    x1, x2, *xs = xs
    result += [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
  return jnp.concatenate(result, axis=-1)
