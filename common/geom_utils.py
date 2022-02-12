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
"""Geometric utilities."""

import numbers
from typing import Optional, Sequence
import numpy as np


def integral_image(val: np.ndarray) -> np.ndarray:
  """Computes a summed volume table of 'val'."""
  pads = []
  ii = val
  for axis in range(len(val.shape)):
    ii = ii.cumsum(axis=axis)
    pads.append([1, 0])

  return np.pad(ii, pads, mode='constant')


def query_integral_image(summed_volume_table: np.ndarray,
                         diam: Sequence[int],
                         stride: Optional[Sequence[int]] = None) -> np.ndarray:
  """Queries a summed volume table.

  Operates in 'VALID' mode, i.e. only computes the sums for voxels where the
  full diam // 2 context is available.

  Args:
    summed_volume_table: 2d or 3d integral image ([z]yx)
    diam: diameter ([z,] y, x tuple) of the area within which to compute sums
    stride: optional [z], y, x stride

  Returns:
    sum of all values within a diam // 2 radius (under L1 metric) of every voxel
    in the array from which 'svt' was built.
  """
  if stride is not None:
    assert len(diam) == len(stride)
  else:
    stride = [1] * len(diam)

  svt = summed_volume_table
  if svt.ndim == 3:
    return (svt[diam[0]::stride[0], diam[1]::stride[1], diam[2]::stride[2]] -
            svt[diam[0]::stride[0], diam[1]::stride[1], :-diam[2]:stride[2]] -
            svt[diam[0]::stride[0], :-diam[1]:stride[1], diam[2]::stride[2]] -
            svt[:-diam[0]:stride[0], diam[1]::stride[1], diam[2]::stride[2]] +
            svt[:-diam[0]:stride[0], :-diam[1]:stride[1], diam[2]::stride[2]] +
            svt[:-diam[0]:stride[0], diam[1]::stride[1], :-diam[2]:stride[2]] +
            svt[diam[0]::stride[0], :-diam[1]:stride[1], :-diam[2]:stride[2]] -
            svt[:-diam[0]:stride[0], :-diam[1]:stride[1], :-diam[2]:stride[2]])
  elif svt.ndim == 2:
    return (svt[diam[0]::stride[0], diam[1]::stride[1]] -
            svt[diam[0]::stride[0], :-diam[1]:stride[1]] -
            svt[:-diam[0]:stride[0], diam[1]::stride[1]] +
            svt[:-diam[0]:stride[0], :-diam[1]:stride[1]])
  else:
    raise NotImplementedError(
        'Only 2 and 3-dimensional integral images are supported.')


def point_query_integral_image(summed_volume_table: np.ndarray,
                               start: Sequence[int],
                               end: Sequence[int]) -> numbers.Number:
  """Returns the sum corresponding to start:end.

  Args:
    summed_volume_table: 2d or 3d integral image ([z]yx)
    start: xy[z] point
    end: xy[z] point

  Returns:
    The sum corresponding to start:end within the array from which svt was
    constructed.
  """
  svt = summed_volume_table
  if svt.ndim == 3:
    return (svt[end[2], end[1], end[0]] - svt[end[2], end[1], start[0]] -
            svt[end[2], start[1], end[0]] - svt[start[2], end[1], end[0]] +
            svt[start[2], start[1], end[0]] + svt[start[2], end[1], start[0]] +
            svt[end[2], start[1], start[0]] - svt[start[2], start[1], start[0]])
  elif svt.ndim == 2:
    return (svt[end[1], end[0]] - svt[start[1], end[0]] -
            svt[end[1], start[0]] + svt[start[1], start[0]])
  else:
    raise NotImplementedError(
        'Only 2 and 3-dimensional integral images are supported.')
