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

from __future__ import annotations

import numbers
from typing import Optional, Sequence

from connectomics.common import bounding_box
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


def downsample_area(
    svt: np.ndarray,
    box: bounding_box.BoundingBox,
    scale: np.ndarray,
    dtype: np.dtype,
    mask_svt: Optional[np.ndarray] = None,
) -> tuple[bounding_box.BoundingBoxBase, np.ndarray]:
  """Downsamples data by area-averaging.

  Args:
    svt: [z, y, x] summed-volume table for the data to downsample
    box: bounding box from which the source data was originates
    scale: xyz downsampling factors
    dtype: data type for the output array
    mask_svt: [z, y, x] summed-volume table for the mask

  Returns:
    bounding box for the downsampled subvolume, downsampled subvolume as a
    [1, z', y', x']-shaped array
  """
  # The offset needs to be such that the first input voxel to the svt is
  # evenly divisible by scale.
  off = box.start - box.start // scale * scale
  off[off > 0] = (scale - off)[off > 0]

  scale_zyx = scale[::-1]
  scale_vol = np.prod(scale)

  # The value computed in ret below corresponds to an output cropped by
  # scale - 1 on each side prior to striding, but we need the output to
  # be cropped by the context expected by the processor.
  ret = query_integral_image(
      svt[off[2] :, off[1] :, off[0] :], diam=scale_zyx, stride=scale_zyx
  )
  if mask_svt is None:
    ret = ret / scale_vol
  else:
    missing = query_integral_image(
        mask_svt[off[2] :, off[1] :, off[0] :], diam=scale_zyx, stride=scale_zyx
    )
    norm = np.clip(scale_vol - missing, 1, None)
    ret = ret / norm
    ret[missing == scale_vol] = np.nan

  ret = np.round(ret).astype(dtype)
  out_box = bounding_box.BoundingBox(
      start=(box.start + off) // scale, size=ret.shape[::-1]
  )
  ret = ret[np.newaxis, ...]
  return out_box, ret
