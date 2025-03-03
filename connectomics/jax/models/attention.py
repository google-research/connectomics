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
from typing import Sequence

from absl import logging
import einops
from flax import linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray
Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class PositionalEmbedding(nn.Module):
  """Adds learnable positional embeddings to [b, ..., l, d] inputs."""

  @nn.compact
  def __call__(self, x):
    *_, l, d = x.shape
    initializer = initializers.normal(stddev=d**-0.5)
    pos_embed = self.param('pos_embed', initializer, (l, d))
    return x + pos_embed


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x


class RelativeAttentionBias(nn.Module):
  """Provides learnable NxN relative attention bias.

  Forked from scenic (https://github.com/google-research/scenic).

  Attributes:
    num_heads: Number of heads for which to provide relative attention.
    nd_shape: Shape for which to provided relative attention bias. For instance,
      for images we we would provide a 2D shape. Note that batch and feature
      dimensions should be excluded here.
    initializer: Initializer for the bias.
  """

  num_heads: int
  nd_shape: Sequence[int]
  initializer: Initializer = nn.initializers.zeros

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    """Creates relative attention bias that factorizes over dimensions.

    length = prod(nd_shape)

    Returns:
      Bias of shape `[num_heads, length, length]`.
    """
    length = np.prod(self.nd_shape)
    tile = 1
    biases = []
    for i, l in enumerate(self.nd_shape):
      # Relative attention in every dimension separately.
      if l > 1:
        new_bias = self.relative_attn_bias(l, self.num_heads, f'bias_{i}')
        repeat = length // (tile * l)
        if repeat > 1:
          new_bias = new_bias[:, :, jnp.newaxis, :, jnp.newaxis]
          new_bias = jnp.tile(new_bias, [1, tile, repeat, tile, repeat])
          new_bias = jnp.reshape(new_bias, [self.num_heads, length, length])
        elif tile > 1:
          new_bias = jnp.tile(new_bias, [1, tile, tile])
        tile *= l
        biases.append(new_bias)

    return sum(biases)

  def relative_attn_bias(self, length, num_heads, name):
    """Computes attention bias based on relative positions.

    Content-based relative position attention bias was used in:
      https://arxiv.org/pdf/1803.02155.
    Non-content-based relative position attention bias was used in:
      https://arxiv.org/abs/1606.01933.

    Args:
      length: Length of self-attention window for relative attention.
      num_heads: Number of attention heads.
      name: Name of the parameter to be created.

    Returns:
      A `[num_heads, length, length]` tensor with queries.
    """
    # Actually we need only 2 * length - 1 relative positions, but we need at
    # least another entry as padding for relative shift of each row to the right
    num_rel_pos = 2 * length

    rel_bias = self.param(name, self.initializer, (self.num_heads, num_rel_pos))

    # Now we have to shift in order to compute relative biases.
    # Example: length = 3
    # Say we want:  [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
    # Start: [[-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3]]
    # We linearize: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3]
    # We slice: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0]
    # We reshape: [[-2, -1, 0, 1, 2], [3, -2, -1, 0, 1], [2, 3, -2, -1, 0]]
    # We slice: [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
    # Tadaaa!

    # [heads, length * num_rel_pos]
    rel_bias = jnp.tile(rel_bias, [1, length])

    # [heads, length * (num_rel_pos - 1)]
    num_rel_pos -= 1
    rel_bias = rel_bias[..., : length * num_rel_pos]

    # [heads, length, num_rel_pos - 1]
    # Now every row is shifted by 1 to the right.
    rel_bias = rel_bias.reshape(num_heads, length, num_rel_pos)

    # [heads, length, length]
    # Slice the overlapping elements from start.
    rel_bias = rel_bias[..., num_rel_pos - length :]

    return rel_bias


def _attention_dropout(
    attn_weights: jnp.ndarray,
    *,
    rate: float,
    broadcast: bool = True,
    dropout_rng: jnp.ndarray,
) -> jnp.ndarray:
  """Applies dropout on attention weights.

  This always applies the dropout. There is no `deterministic` parameter.

  Forked from scenic (https://github.com/google-research/scenic).

  Arguments:
    attn_weights: Attention weights.
    rate: The dropout rate. (_not_ the keep rate!)
    broadcast: Whether to broadcast on first and second last axis.
    dropout_rng: RNG.

  Returns:
    Weights after dropout.
  """
  keep_prob = 1.0 - rate
  if broadcast:
    # Dropout is broadcast across the batch+head+non-attention dimension.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[0] = 1  # Broadcast batch.
    dropout_shape[-2] = 1  # Broadcast heads.
    keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
  else:
    keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
  multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
      keep_prob, dtype=attn_weights.dtype
  )
  return attn_weights * multiplier


def dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    bias: jnp.ndarray | None = None,
    bias_kv: jnp.ndarray | None = None,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.1,
    dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision | None = None,
    deterministic: bool,
    dropout_rng: jnp.ndarray | None = None,
    capture_attention_weights: bool = True,
) -> jnp.ndarray:
  # pylint: disable=g-doc-args
  """Computes the dot-product attention given query, key and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Forked from scenic (https://github.com/google-research/scenic).

  Args:
    query: Queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: Keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: Values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    bias: Bias for the attention weights. This should be broadcastable to the
      shape: `[batch..., num_heads, q_length, kv_length]` This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
    bias_kv: Attention bias defined for keys only which has shape `[batch...,
      kv_length]`. Can be used for masking elements in k/v.
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    dropout_rate: Dropout rate.
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    deterministic: Deterministic or not (to apply dropout).
    dropout_rng: Optional JAX PRNGKey to be used for dropout.
    capture_attention_weights: Whether to add an identity layer to tag the
      attention weights to be used for capturing them using Flax
      capture_intermediate, e.g. for visualization. Note that if this is set to
      True, this function can be only called within a Flax module.

  Returns:
    Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
  """
  # pylint: enable=g-doc-args
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
      query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
      query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Calculate attention matrix.
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision
  )

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  if bias_kv is not None:
    bias_kv = bias_kv[..., jnp.newaxis, jnp.newaxis, :]
    attn_weights += bias_kv

  # Normalize the attention weights.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if capture_attention_weights:
    # Tag the intermediate weights for logging/visualization.
    attn_weights = IdentityLayer(name='attn_weights')(attn_weights)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.0:
    if dropout_rng is None:
      raise ValueError('Did not provide `rng` to dot_product_attention().')
    attn_weights = _attention_dropout(
        attn_weights,
        rate=dropout_rate,
        broadcast=broadcast_dropout,
        dropout_rng=dropout_rng,
    )

  # Return weighted sum over values for each query position.
  return jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  )


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
      attention_bias = RelativeAttentionBias(
          self.num_heads, self.get_attn_shape(spatial_shape)
      )()
    else:
      attention_bias = None

    if train and self.dropout > 0:
      dropout_rng = self.make_rng('dropout')
    else:
      dropout_rng = None

    x = dot_product_attention(
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
