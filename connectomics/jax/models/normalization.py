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
"""Normalization modules for Flax."""

import functools
from typing import Any, Callable, Optional

import flax.linen as nn
from flax.linen.dtypes import canonicalize_dtype  # pylint: disable=g-importing-member
from flax.linen.module import Module, compact  # pylint: disable=g-importing-member,g-multiple-import
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize  # pylint: disable=g-importing-member,g-multiple-import
from flax.linen.normalization import Array, Axes, Dtype, PRNGKey, Shape   # pylint: disable=g-importing-member,g-multiple-import
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


class NoOp(Module):
  """NoOp."""

  @compact
  def __call__(self, x):
    return x


def norm_layer_from_str(input_string: str, train: Optional[bool] = None) -> Any:
  """Gets normalization layer from string."""
  kwargs = {}
  if not input_string or input_string == 'NoOp':
    layer = NoOp
  elif input_string.startswith('BatchNorm'):
    layer = nn.BatchNorm
    if train is not None:
      kwargs['use_running_average'] = not train
    if '(' in input_string:
      kwargs['momentum'] = float(input_string.replace(
          'BatchNorm(', '').replace(')', ''))
  elif input_string == 'InstanceNorm':
    layer = nn.GroupNorm
    kwargs['group_size'] = 1
  elif input_string == 'ReversibleInstanceNorm':
    layer = ReversibleInstanceNorm
  elif hasattr(nn, input_string):
    layer = getattr(nn, input_string)
  else:
    raise ValueError('normalization layer not found as part of flax.linen.')
  return functools.partial(layer, **kwargs)


def _denormalize(
    mdl: Module,
    x: Array,
    mean: Array,
    var: Array,
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Dtype,
    param_dtype: Dtype,
    epsilon: float,
    use_bias: bool,
    use_scale: bool,
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array],
):
  """Denormalizes the input of a normalization layer with optional learned scale and bias.

  Arguments:
    mdl: Module to apply the denormalization in (normalization params will
      reside in this module).
    x: The input.
    mean: Mean to use for denormalization.
    var: Variance to use for denormalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    param_dtype: The dtype of the parameters.
    epsilon: Denormalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.

  Returns:
    The denormalized input.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])

  y = x
  args = [x]
  if use_bias:
    bias = mdl.param(
        'bias', bias_init, reduced_feature_shape, param_dtype
    ).reshape(feature_shape)
    y -= bias
    args.append(bias)
  var = jnp.expand_dims(var, reduction_axes)
  mul = lax.sqrt(var + epsilon)
  if use_scale:
    scale = mdl.param(
        'scale', scale_init, reduced_feature_shape, param_dtype
    ).reshape(feature_shape)
    mul /= scale
    args.append(scale)
  y *= mul
  y += jnp.expand_dims(mean, reduction_axes)
  dtype = canonicalize_dtype(*args, dtype=dtype)
  return jnp.asarray(y, dtype)


class ReversibleInstanceNorm(Module):
  """Reversible instance normalization (https://openreview.net/forum?id=cGDAkQo1C0p).

  Usage example:
    rev_in = ReversibleInstanceNorm()
    x, stats = rev_in(x)  # x is normalized
    # ...
    y, _ = rev_in(x, stats)  # x is denormalized using stats
    return y

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  epsilon: float = 1e-6
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Any = None
  use_fast_variance: bool = True

  @compact
  def __call__(self, x, stats=None):
    """Applies (reversible) instance normalization on the input.

    Args:
      x: the inputs
      stats: statistics, if passed, inputs are denormalized.

    Returns:
      (De)normalized inputs (the same shape as inputs) and stats.
    """
    reduction_axes = list(range(1, x.ndim - 1)) + [-1]
    feature_axes = (-1,)

    if stats is None:
      transform_fn = _normalize
      mean, var = _compute_stats(
          x.reshape(x.shape + (1,)),
          reduction_axes,
          self.dtype,
          self.axis_name,
          self.axis_index_groups,
          use_fast_variance=self.use_fast_variance,
      )
      stats = {'mean': mean, 'var': var}
    else:
      transform_fn = _denormalize

    return transform_fn(
        self,
        x,
        stats['mean'],
        stats['var'],
        reduction_axes[:-1],
        feature_axes,
        self.dtype,
        self.param_dtype,
        self.epsilon,
        self.use_bias,
        self.use_scale,
        self.bias_init,
        self.scale_init,
    ), stats
