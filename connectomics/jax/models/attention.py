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
"""Attention layers for n-dimensional spatial inputs.

Currently, contains pixel-based, patch-based, axis-based attention mechanisms
with typical defaults and options used for vision transformers, such as position
biases and learnable positional embeddings.
"""

from collections.abc import Callable
import functools
from absl import logging
import einops
from flax import linen as nn
from flax.linen import initializers
import jax.numpy as jnp
from scenic.model_lib.layers import attention_layers as scenic_attn

Array = jnp.ndarray


class PositionalEmbedding(nn.Module):
  """Adds learnable positional embeddings to [b, ..., l, d] inputs."""

  @nn.compact
  def __call__(self, x):
    *_, l, d = x.shape
    initializer = initializers.normal(stddev=d**-0.5)
    pos_embed = self.param('pos_embed', initializer, (l, d))
    return x + pos_embed


class Attention(nn.Module):
  """Multi-head attention customized from scenic.

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    qkv_features: Dimension of the key, query, and value.
    dropout: Dropout rate.
    positional_embed: Whether to add positional embeddings.
    relative_attention_bias: Whether to use relative attention bias.
  """

  num_heads: int = 32
  qkv_features: int | None = None
  dropout: float = 0.0
  positional_embed: bool = True
  relative_attention_bias: bool = True
  seq_shard_fn: Callable[[Array], Array] = lambda x: x

  def to_seq(self, x: Array) -> tuple[Array, tuple[int, ...]]:
    """Reshape input to sequence and return spatial shape."""
    return x, x.shape[1:-1]

  def from_seq(self, x: Array, spatial_shape: tuple[int, ...]) -> Array:
    """Reshape input to spatial and return output."""
    return x

  def get_attn_shape(self, spatial_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Get n-dimensional shape that attention is applied to."""
    return spatial_shape

  @nn.compact
  def __call__(self, x: Array, train: bool = False) -> jnp.ndarray:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` or for self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      x: Inputs of shape `[bs, ..., features]`.
      train: Whether the model is in train mode.

    Returns:
      Output of shape `[bs, ..., features]`.
    """
    x, spatial_shape = self.to_seq(x)
    x = self.seq_shard_fn(x)
    out_features = x.shape[-1]
    qkv_features = self.qkv_features or x.shape[-1]
    assert (
        qkv_features % self.num_heads == 0
    ), 'Memory dimension must be divisible by number of heads.'
    head_dim = qkv_features // self.num_heads

    if self.positional_embed:
      x = PositionalEmbedding(name='pos_embed')(x)

    # Project inputs_q to multi-headed q/k/v with dimensions
    #  [..., l, num_heads, num_features_per_head].
    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
    )
    query, key, value = (
        dense(name='query')(x),
        dense(name='key')(x),
        dense(name='value')(x),
    )
    query = nn.LayerNorm(name='query_ln', use_bias=False)(query)
    key = nn.LayerNorm(name='key_ln', use_bias=False)(key)

    if self.relative_attention_bias:
      attention_bias = scenic_attn.RelativeAttentionBias(
          self.num_heads, self.get_attn_shape(spatial_shape)
      )()
    else:
      attention_bias = None

    if train and self.dropout > 0:
      dropout_rng = self.make_rng('dropout')
    else:
      dropout_rng = None

    x = scenic_attn.dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rate=self.dropout,
        dropout_rng=dropout_rng,
        deterministic=not train,
    )

    # Back to the original inputs dimensions.
    out = nn.DenseGeneral(
        features=out_features, axis=(-2, -1), use_bias=True, name='out'
    )(x)

    return self.from_seq(out, spatial_shape)


class VoxelAttention(Attention):
  """Multi-head attention with voxels as sequence elements."""

  def to_seq(self, x: Array) -> tuple[Array, tuple[int, ...]]:
    b, *_, c = x.shape
    return x.reshape(b, -1, c), x.shape[1:-1]

  def from_seq(self, x: Array, spatial_shape: tuple[int, ...]) -> Array:
    b, *_, c = x.shape
    return x.reshape(b, *spatial_shape, c)


class EinopAttention(Attention):
  """Multi-head attention with einops patterns."""

  def _get_pattern(
      self, spatial_shape: tuple[int, ...]
  ) -> tuple[str, str, dict[str, int]]:
    raise NotImplementedError()

  def to_seq(self, x: Array) -> tuple[Array, tuple[int, ...]]:
    spatial_shape = x.shape[1:-1]
    in_pattern, out_pattern, axes = self._get_pattern(spatial_shape)
    pattern = in_pattern + ' -> ' + out_pattern
    logging.info(
        'Volume to sequence pattern %r for shape %r', pattern, spatial_shape
    )
    x = einops.rearrange(x, pattern, **axes)
    return x, spatial_shape

  def from_seq(self, x: Array, spatial_shape: tuple[int, ...]) -> Array:
    in_pattern, out_pattern, axes = self._get_pattern(spatial_shape)
    pattern = out_pattern + ' -> ' + in_pattern
    x = einops.rearrange(x, pattern, **axes)
    return x

  def get_attn_shape(self, spatial_shape: tuple[int, ...]) -> tuple[int, ...]:
    raise NotImplementedError()


class PatchAttention(EinopAttention):
  """Multi-head attention between contiguous patches of voxels.

  For 2D and with p{i} denoting patch sizes and d{i} patched dimensions,
  the pattern is 'b (d1 p1) (d2 p2) c -> b (d1 d2) (p1 p2 c)' so we have
  d1 x d2 sequence items with p1 p2 c feature dimension.
  """

  patch_sizes: tuple[int, ...] = (8,)

  def _get_pattern(
      self, spatial_shape: tuple[int, ...]
  ) -> tuple[str, str, dict[str, int]]:
    spatial_in, spatial_out, feature_out = '', '', ''
    axes = dict()
    if len(spatial_shape) != len(self.patch_sizes):
      raise ValueError('spatial_shape and patch_sizes must have same length')
    for i, (dim, patch_size) in enumerate(zip(spatial_shape, self.patch_sizes)):
      spatial_in += f'(d{i+1} p{i+1}) '
      spatial_out += f'd{i+1} '
      feature_out += f'p{i+1} '
      axes[f'p{i+1}'] = patch_size
      axes[f'd{i+1}'] = dim // patch_size
    in_pattern = f'b {spatial_in}c'
    out_pattern = f'b ({spatial_out}) ({feature_out}c)'
    return in_pattern, out_pattern, axes

  def get_attn_shape(self, spatial_shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(d // p for d, p in zip(spatial_shape, self.patch_sizes))


class BlockAttention(EinopAttention):
  """Multi-head attention within contiguous patches of voxels.

  For 2D and with p{i} denoting patch sizes and d{i} patched dimensions,
  the pattern is 'b (d1 p1) (d2 p2) c -> b d1 d2 (p1 p2) c' so we have
  p1 x p2 sequence items with c feature dimension (d1, d2 treated as batch).
  """

  patch_sizes: tuple[int, ...] = (8,)

  def _get_pattern(
      self, spatial_shape: tuple[int, ...]
  ) -> tuple[str, str, dict[str, int]]:
    spatial_in, spatial_out, feature_out = '', '', ''
    axes = dict()
    if len(spatial_shape) != len(self.patch_sizes):
      raise ValueError('spatial_shape and patch_sizes must have same length')
    for i, (dim, patch_size) in enumerate(zip(spatial_shape, self.patch_sizes)):
      spatial_in += f'(d{i+1} p{i+1}) '
      spatial_out += f'd{i+1} '
      feature_out += f'p{i+1} '
      axes[f'p{i+1}'] = patch_size
      axes[f'd{i+1}'] = dim // patch_size
    in_pattern = f'b {spatial_in}c'
    out_pattern = f'b {spatial_out} ({feature_out}) c'
    logging.info(
        'block attention patterns %r %r and axes %r',
        in_pattern,
        out_pattern,
        axes,
    )
    return in_pattern, out_pattern, axes

  def get_attn_shape(self, spatial_shape: tuple[int, ...]) -> tuple[int, ...]:
    return self.patch_sizes


class GridAttention(EinopAttention):
  """Multi-head attention within strided grid spaced across input volume.

  For 2D and with p{i} denoting patch sizes and d{i} patched dimensions,
  the pattern is 'b (p1 d1) (p2 d2) c -> b d1 d2 (p1 p2) c' so we have
  p1 x p2 sequence items with c feature dimension (d1, d2 treated as batch).
  """

  patch_sizes: tuple[int, ...] = (8,)

  def _get_pattern(
      self, spatial_shape: tuple[int, ...]
  ) -> tuple[str, str, dict[str, int]]:
    spatial_in, spatial_out, feature_out = '', '', ''
    axes = dict()
    if len(spatial_shape) != len(self.patch_sizes):
      raise ValueError('spatial_shape and patch_sizes must have same length')
    for i, (dim, patch_size) in enumerate(zip(spatial_shape, self.patch_sizes)):
      spatial_in += f'(p{i+1} d{i+1}) '
      spatial_out += f'd{i+1} '
      feature_out += f'p{i+1} '
      axes[f'p{i+1}'] = patch_size
      axes[f'd{i+1}'] = dim // patch_size
    in_pattern = f'b {spatial_in}c'
    out_pattern = f'b {spatial_out} ({feature_out}) c'
    return in_pattern, out_pattern, axes

  def get_attn_shape(self, spatial_shape: tuple[int, ...]) -> tuple[int, ...]:
    return self.patch_sizes
