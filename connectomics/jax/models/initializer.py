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
"""Initializer functions."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def constant_init(dim, dtype=jnp.float_):
  """Initializes weights to `1 / shape[dim]`."""

  def init(unused_key, shape, dtype=dtype):
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return 1. / shape[dim] * jnp.full(shape, 1., dtype=dtype)

  return init


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
    max_len: maximum possible length for the input.
    min_scale: float: minimum frequency-scale in sine grating.
    max_scale: float: maximum frequency-scale in sine grating.

  Returns:
    output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, : d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


def init_fn_from_str(input_string: str):
  """Gets init function from string."""
  if input_string == 'constant':
    return constant_init
  elif input_string.startswith('normal('):
    std = input_string.replace('normal(', '').replace(')', '')
    return nn.initializers.normal(float(std))
  elif input_string == 'sinusoidal':
    return sinusoidal_init
  elif hasattr(nn, input_string):
    return getattr(nn, input_string)
  else:
    raise ValueError('init function not found as part of flax.linen.')
