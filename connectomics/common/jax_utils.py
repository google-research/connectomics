# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Jax utilities.
"""

import functools
import string
from typing import Optional

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def correlate_location(
    array: jnp.ndarray,
    loc: tuple[int, ...],
    axis: int = -1,
    method: str = 'einsum',
    nan_to_num: bool = False,
) -> jnp.ndarray:
  """Cross-correlates an N-dimensional array with a location inside it.

  Cross-correlates two arrays along a given axis, where the second array is
  a subset of the first, e.g., `array[loc[1], loc[2], loc[3], :]` if `axis=-1`,
  with `loc` being the location.

  Args:
    array: Input array.
    loc: Location within array. Must be of size N-1, i.e., leaving out `axis`.
    axis: Axis along which to cross-correlate.
    method: Method used to calculate cross-correlations, `scipy` or `einsum`.
    nan_to_num: If set, replaces NaNs and infs in output using `nan_to_num`.

  Returns:
    (N-1)-dimensional array of cross-correlations along `axis` in [-1, +1].
  """
  num_dim = array.ndim
  axis = axis if axis > -1 else num_dim + axis
  assert 0 <= axis < num_dim, 'Invalid axis argument.'
  assert len(loc) == num_dim - 1, 'Invalid length of loc argument.'

  array -= jnp.mean(array, axis=axis, keepdims=True)
  array /= jnp.std(array, axis=axis, keepdims=True)

  start_indices = list(loc)
  start_indices.insert(axis, 0)
  slice_size = [1] * len(loc)
  slice_size.insert(axis, array.shape[axis])
  in2 = jax.lax.dynamic_slice(array, start_indices, slice_size)

  if method == 'scipy':
    out = jax.scipy.signal.correlate(
        in1=array, in2=in2, mode='valid',
        method='auto').squeeze(axis=axis)
  elif method == 'einsum':
    assert num_dim < 27, 'Dimensionality must be smaller than 27.'
    label_in1 = string.ascii_lowercase[:num_dim-1]
    label_in1 = f'{label_in1[:axis]}~{label_in1[axis:]}'
    label_in2 = string.ascii_uppercase[:num_dim-1]
    label_in2 = f'{label_in2[:axis]}~{label_in2[axis:]}'
    label_res = string.ascii_lowercase[:num_dim-1]
    expr = f'{label_in1},{label_in2}->{label_res}'
    out = jnp.einsum(expr, array, in2)
  else:
    raise ValueError(f'method must be `scipy` or `einsum`, not `{method}`.')

  if nan_to_num:
    out = jnp.nan_to_num(out)

  return out / array.shape[axis]


def correlate_center(
    array: jnp.ndarray,
    axis: int = -1,
    method: str = 'einsum'
) -> jnp.ndarray:
  """Cross-correlates an N-dimensional array against its center location.

  Cross-correlates two arrays along a given axis, where the second array is
  a subset of the first, e.g., `array[loc[1], loc[2], loc[3], :]` if `axis=-1`,
  with `loc` being the center location.

  Args:
    array: Input array.
    axis: Axis along which to cross-correlate.
    method: Method used to calculate cross-correlations, `scipy` or `einsum`.

  Returns:
    (N-1)-dimensional array of cross-correlations along `axis` in [-1, +1].
  """
  num_dim = array.ndim
  axis = axis if axis > -1 else num_dim + axis
  return correlate_location(
      array=array,
      loc=tuple([array.shape[d] // 2 for d in range(num_dim) if d != axis]),
      axis=axis,
      method=method)


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def distances_location(
    array: jnp.ndarray,
    loc: tuple[int, ...],
    weights: Optional[tuple[float, ...]] = None,
    squared: bool = False,
) -> jnp.ndarray:
  """Calculates (weighted) l2-distances to location within array.

  Args:
    array: Input array.
    loc: Location within array.
    weights: Optional weights per dimension of array.
    squared: If True, returns squared distances.

  Returns:
    Array of (weighted, squared) l2-distances to location.
  """
  num_dim = array.ndim
  assert num_dim < 27, 'Dimensionality must be smaller than 27.'
  weighted_squared_dist = jnp.einsum(
      '{dims}~,~->{dims}'.format(dims=string.ascii_lowercase[:num_dim]),
      (jnp.moveaxis(jnp.indices(array.shape), 0, -1) - jnp.array(loc))**2,
      weights if weights is not None else jnp.ones((num_dim,)))
  return weighted_squared_dist if squared else jnp.sqrt(weighted_squared_dist)


def parse_device_str(device_idx_str: str) -> jax.Device:
  """Gets Device from device[:idx] formatted string, e.g., `cpu` or `gpu:1`."""
  parts = device_idx_str.split(':')
  if len(parts) not in (1, 2):
    raise ValueError('Invalid input string, expecting device[:idx] format.')

  device_name, device_idx = parts[0], 0 if len(parts) == 1 else int(parts[1])
  devices = jax.devices(device_name)
  if device_idx > len(devices) - 1:
    raise ValueError(
        f'Device index out of range given available {device_name} devices.')
  return devices[device_idx]
