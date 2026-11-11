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
from typing import Sequence

from connectomics.common import bounding_box
import numpy as np


def integral_image(val: np.ndarray) -> np.ndarray:
  """Computes a summed volume table of 'val'."""
  out_dtype = np.empty(0, dtype=val.dtype).cumsum().dtype
  out = np.zeros(tuple(s + 1 for s in val.shape), dtype=out_dtype)
  if val.size == 0:
    return out
  interior = out[tuple(slice(1, None) for _ in val.shape)]
  active_axes = [axis for axis, s in enumerate(val.shape) if s > 1]
  if not active_axes:
    np.copyto(interior, val)
    return out
  np.cumsum(val, axis=active_axes[0], out=interior)
  for axis in active_axes[1:]:
    np.cumsum(interior, axis=axis, out=interior)
  return out


def query_integral_image(summed_volume_table: np.ndarray,
                         diam: Sequence[int],
                         stride: Sequence[int] | None = None) -> np.ndarray:
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
    d0, d1, d2 = diam[0], diam[1], diam[2]
    s0, s1, s2 = stride[0], stride[1], stride[2]
    if (
        d0 == s0
        and d1 == s1
        and d2 == s2
        and svt.shape[0] > d0
        and svt.shape[1] > d1
        and svt.shape[2] > d2
    ):
      n0 = (svt.shape[0] - 1) // d0
      n1 = (svt.shape[1] - 1) // d1
      n2 = (svt.shape[2] - 1) // d2
      grid = svt[: n0 * d0 + 1 : d0, : n1 * d1 + 1 : d1, : n2 * d2 + 1 : d2]
      return np.diff(np.diff(np.diff(grid, axis=0), axis=1), axis=2)
    return (svt[d0::s0, d1::s1, d2::s2] -
            svt[d0::s0, d1::s1, :-d2:s2] -
            svt[d0::s0, :-d1:s1, d2::s2] -
            svt[:-d0:s0, d1::s1, d2::s2] +
            svt[:-d0:s0, :-d1:s1, d2::s2] +
            svt[:-d0:s0, d1::s1, :-d2:s2] +
            svt[d0::s0, :-d1:s1, :-d2:s2] -
            svt[:-d0:s0, :-d1:s1, :-d2:s2])
  elif svt.ndim == 2:
    d0, d1 = diam[0], diam[1]
    s0, s1 = stride[0], stride[1]
    if d0 == s0 and d1 == s1 and svt.shape[0] > d0 and svt.shape[1] > d1:
      n0 = (svt.shape[0] - 1) // d0
      n1 = (svt.shape[1] - 1) // d1
      grid = svt[: n0 * d0 + 1 : d0, : n1 * d1 + 1 : d1]
      return np.diff(np.diff(grid, axis=0), axis=1)
    return (svt[d0::s0, d1::s1] -
            svt[d0::s0, :-d1:s1] -
            svt[:-d0:s0, d1::s1] +
            svt[:-d0:s0, :-d1:s1])
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


def _svt_block_boundaries(offset: int, limit: int, step: int) -> np.ndarray:
  """Returns 1D SVT boundary indices for blocks starting at offset up to limit."""
  if offset >= limit:
    return np.empty(0, dtype=np.intp)
  starts = np.arange(offset, limit, step, dtype=np.intp)
  return np.append(starts, limit)


def _query_svt_blocks(
    svt: np.ndarray,
    sel_z: slice | np.ndarray,
    sel_y: slice | np.ndarray,
    sel_x: slice | np.ndarray,
) -> np.ndarray:
  """Queries 3D SVT block sums at given per-axis boundary slices or indices."""
  grid = svt[sel_z, :, :][:, sel_y, :][:, :, sel_x]
  return np.diff(np.diff(np.diff(grid, axis=0), axis=1), axis=2)


def downsample_area(
    svt: np.ndarray,
    box: bounding_box.BoundingBox,
    scale: np.ndarray,
    dtype: np.dtype,
    mask_svt: np.ndarray | None = None,
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
  off_zyx = off[::-1]
  scale_vol = int(np.prod(scale))

  limits = np.array(svt.shape) - 1
  effective_shape = limits - off_zyx
  if np.any(effective_shape <= 0):
    out_shape = tuple(
        max(0, int((eff + s - 1) // s))
        for eff, s in zip(effective_shape, scale_zyx)
    )
    ret = np.empty((1,) + out_shape, dtype=dtype)
    out_box = bounding_box.BoundingBox(
        start=(box.start + off) // scale, size=out_shape[::-1]  # pyrefly: ignore[bad-argument-type]
    )
    return out_box, ret

  remainder = effective_shape % scale_zyx
  indices = [
      _svt_block_boundaries(int(off_zyx[a]), int(limits[a]), int(scale_zyx[a]))
      for a in range(3)
  ]
  selectors = [
      slice(int(off_zyx[a]), None, int(scale_zyx[a]))
      if remainder[a] == 0
      else indices[a]
      for a in range(3)
  ]

  ret = _query_svt_blocks(svt, selectors[0], selectors[1], selectors[2])
  has_remainder = bool(np.any(remainder > 0))
  if mask_svt is None and not has_remainder:
    ret = ret / scale_vol
  else:
    dz = np.diff(indices[0])
    dy = np.diff(indices[1])
    dx = np.diff(indices[2])
    counts = (
        dz[:, np.newaxis, np.newaxis]
        * dy[np.newaxis, :, np.newaxis]
        * dx[np.newaxis, np.newaxis, :]
    )
    if mask_svt is None:
      ret = ret / counts
    else:
      missing = _query_svt_blocks(
          mask_svt, selectors[0], selectors[1], selectors[2]
      )
      valid = counts - missing
      norm = np.clip(valid, 1, None)
      ret = ret / norm
      if np.issubdtype(dtype, np.floating):
        ret[valid <= 0] = np.nan
      else:
        ret[valid <= 0] = 0

  ret = np.round(ret, out=ret).astype(dtype)
  out_box = bounding_box.BoundingBox(
      start=(box.start + off) // scale, size=ret.shape[::-1]  # pyrefly: ignore[bad-argument-type]
  )
  ret = ret[np.newaxis, ...]
  return out_box, ret


def _select_sum_dtype(input_dtype: np.dtype, max_block_vol: int) -> np.dtype:
  """Selects the narrowest accumulator dtype that cannot overflow."""
  if input_dtype == np.uint8:
    if max_block_vol <= 1:
      return np.dtype(np.uint8)
    if max_block_vol <= 255:
      return np.dtype(np.uint16)
    if max_block_vol <= 16_000_000:
      return np.dtype(np.uint32)
    return np.dtype(np.uint64)
  if input_dtype == np.uint32:
    if max_block_vol <= 1:
      return np.dtype(np.uint32)
    return np.dtype(np.uint64)
  if input_dtype == np.float32:
    # Sums are accumulated in float64 to match the precision of the float64
    # summed-volume table path.
    if max_block_vol <= 1:
      return np.dtype(np.float32)
    return np.dtype(np.float64)
  return np.dtype(np.float64)


def _select_count_dtype(max_block_vol: int) -> np.dtype:
  """Selects the narrowest integer dtype for counting voxels in a block."""
  if max_block_vol <= 255:
    return np.dtype(np.uint8)
  if max_block_vol <= 65535:
    return np.dtype(np.uint16)
  if max_block_vol <= 4_000_000_000:
    return np.dtype(np.uint32)
  return np.dtype(np.int64)


def _block_lengths_1d(length: int, step: int) -> np.ndarray:
  """Returns 1D array of block sizes for an axis of `length` partitioned by `step`."""
  if length <= 0:
    return np.empty(0, dtype=np.int64)
  m, r = divmod(length, step)
  out_n = m + (1 if r > 0 else 0)
  out = np.full(out_n, step, dtype=np.int64)
  if r > 0:
    out[-1] = r
  return out


def _reduce_axis_1d(
    arr: np.ndarray, step: int, dtype: np.dtype
) -> np.ndarray:
  """Sums non-overlapping blocks of length `step` along a 1D array."""
  if step == 1:
    return arr if arr.dtype == dtype else arr.astype(dtype)
  n = arr.shape[0]
  if n == 0:
    return np.empty(0, dtype=dtype)
  return np.add.reduceat(
      arr, np.arange(0, n, step, dtype=np.intp), axis=0, dtype=dtype
  )


def _reduce_axis_3d(
    arr: np.ndarray, step: int, axis: int, dtype: np.dtype
) -> np.ndarray:
  """Sums non-overlapping blocks of length `step` along `axis` of a 3D array."""
  if step == 1:
    return arr if arr.dtype == dtype else arr.astype(dtype)
  n = arr.shape[axis]
  if n == 0:
    out_shape = list(arr.shape)
    out_shape[axis] = 0
    return np.empty(out_shape, dtype=dtype)
  if step > 8:
    return np.add.reduceat(
        arr, np.arange(0, n, step, dtype=np.intp), axis=axis, dtype=dtype
    )

  m, r = divmod(n, step)
  out_n = m + (1 if r > 0 else 0)
  out_shape = list(arr.shape)
  out_shape[axis] = out_n
  out = np.empty(out_shape, dtype=dtype)
  end = m * step

  if axis == 0:
    if m > 0:
      np.add(
          arr[0:end:step, :, :],
          arr[1:end:step, :, :],
          out=out[:m, :, :],
          dtype=dtype,
      )
      for k in range(2, step):
        np.add(out[:m, :, :], arr[k:end:step, :, :], out=out[:m, :, :])
    if r > 0:
      arr[end:, :, :].sum(axis=0, dtype=dtype, out=out[m, :, :])
  elif axis == 1:
    if m > 0:
      np.add(
          arr[:, 0:end:step, :],
          arr[:, 1:end:step, :],
          out=out[:, :m, :],
          dtype=dtype,
      )
      for k in range(2, step):
        np.add(out[:, :m, :], arr[:, k:end:step, :], out=out[:, :m, :])
    if r > 0:
      arr[:, end:, :].sum(axis=1, dtype=dtype, out=out[:, m, :])
  else:
    if m > 0:
      np.add(
          arr[:, :, 0:end:step],
          arr[:, :, 1:end:step],
          out=out[:, :, :m],
          dtype=dtype,
      )
      for k in range(2, step):
        np.add(out[:, :, :m], arr[:, :, k:end:step], out=out[:, :, :m])
    if r > 0:
      arr[:, :, end:].sum(axis=2, dtype=dtype, out=out[:, :, m])
  return out


def _reduce_block_3d(
    arr: np.ndarray,
    steps_zyx: Sequence[int] | np.ndarray,
    input_dtype: np.dtype,
    base_vol: int = 1,
    is_count: bool = False,
) -> np.ndarray:
  """Separably sums 3D blocks of shape `steps_zyx` along Z, Y, and X."""
  vol = base_vol
  out = arr
  for axis in range(3):
    step = int(steps_zyx[axis])
    vol *= step
    if is_count:
      step_dtype = _select_count_dtype(vol)
    else:
      step_dtype = _select_sum_dtype(input_dtype, vol)
    out = _reduce_axis_3d(out, step, axis, step_dtype)
  return out


def _divide_and_round_constant(
    sums: np.ndarray, scale_vol: int, dtype: np.dtype
) -> np.ndarray:
  """Divides `sums` by scalar `scale_vol` and rounds to nearest even in `dtype`."""
  if (
      dtype in (np.uint8, np.uint32)
      and scale_vol > 0
      and (scale_vol & (scale_vol - 1)) == 0
  ):
    if scale_vol == 1:
      return sums.astype(dtype)
    k = scale_vol.bit_length() - 1
    half = 1 << (k - 1)
    q = np.right_shift(sums, k)
    q &= 1
    if half > 1:
      q += half - 1
    q += sums
    q >>= k
    return q.astype(dtype)
  avg = sums / scale_vol
  return np.round(avg, out=avg).astype(dtype)


def _finalize_block_average(
    sums: np.ndarray,
    scale_vol: int,
    dtype: np.dtype,
    dz: np.ndarray | None,
    dy: np.ndarray | None,
    dx: np.ndarray | None,
    counts_3d: np.ndarray | None,
    sz: int,
    sy: int,
    sx: int,
) -> np.ndarray:
  """Computes rounded block averages from block sums and valid counts."""
  if sums.size == 0:
    return np.empty(sums.shape, dtype=dtype)

  if counts_3d is not None:
    norm = np.clip(counts_3d, 1, None)
    avg = sums / norm
    if np.issubdtype(dtype, np.floating):
      avg[counts_3d == 0] = np.nan
    else:
      avg[counts_3d == 0] = 0
    return np.round(avg, out=avg).astype(dtype)

  assert dz is not None and dy is not None and dx is not None
  rem_z = int(dz[-1]) != sz
  rem_y = int(dy[-1]) != sy
  rem_x = int(dx[-1]) != sx
  if not (rem_z or rem_y or rem_x):
    return _divide_and_round_constant(sums, scale_vol, dtype)

  out = _divide_and_round_constant(sums, scale_vol, dtype)
  if rem_z:
    c_z = (
        dz[-1:, np.newaxis, np.newaxis]
        * dy[np.newaxis, :, np.newaxis]
        * dx[np.newaxis, np.newaxis, :]
    )
    avg_z = sums[-1:, :, :] / c_z
    out[-1:, :, :] = np.round(avg_z, out=avg_z).astype(dtype)
  if rem_y:
    c_y = (
        dz[:, np.newaxis, np.newaxis]
        * dy[np.newaxis, -1:, np.newaxis]
        * dx[np.newaxis, np.newaxis, :]
    )
    avg_y = sums[:, -1:, :] / c_y
    out[:, -1:, :] = np.round(avg_y, out=avg_y).astype(dtype)
  if rem_x:
    c_x = (
        dz[:, np.newaxis, np.newaxis]
        * dy[np.newaxis, :, np.newaxis]
        * dx[np.newaxis, np.newaxis, -1:]
    )
    avg_x = sums[:, :, -1:] / c_x
    out[:, :, -1:] = np.round(avg_x, out=avg_x).astype(dtype)
  return out


def downsample_area_scales(
    data: np.ndarray,
    box: bounding_box.BoundingBox,
    scales: Sequence[Sequence[int] | np.ndarray],
    mask: np.ndarray | None = None,
    mask_already_applied: bool = False,
) -> list[tuple[bounding_box.BoundingBoxBase, np.ndarray]]:
  """Downsamples a [C, Z, Y, X] array by area-averaging over one or more scales.

  Uses separable 3D block reduction with minimal-width accumulators and a
  hierarchical multi-scale pyramid when coarser scales are integer multiples
  of finer scales.

  Args:
    data: [C, Z, Y, X] input array (uint8, uint32, or float32).
    box: bounding box from which the source data originates.
    scales: sequence of xyz downsampling factor sequences/arrays.
    mask: optional [Z, Y, X] boolean mask where True indicates excluded voxels.
    mask_already_applied: if True, masked voxels in `data` are already set to 0.

  Returns:
    List of (out_box, downsampled_array) tuples in the same order as `scales`,
    where each `downsampled_array` has shape [C, z', y', x'] and `data.dtype`.
  """
  dtype = data.dtype
  if dtype not in (np.uint8, np.uint32, np.float32):
    raise NotImplementedError(f'Unexpected input type: {dtype!r}')

  scale_arrays = [np.asarray(s) for s in scales]
  num_scales = len(scale_arrays)
  if num_scales == 0:
    return []

  offsets = []
  for scale in scale_arrays:
    off = box.start - box.start // scale * scale
    off[off > 0] = (scale - off)[off > 0]
    offsets.append(off)

  order = sorted(range(num_scales), key=lambda i: int(np.prod(scale_arrays[i])))
  parents: list[int | None] = [None] * num_scales
  remaining_children = [0] * num_scales
  for pos, i in enumerate(order):
    best_parent = None
    best_vol = -1
    for j in order[:pos]:
      if np.all(scale_arrays[i] % scale_arrays[j] == 0):
        vol_j = int(np.prod(scale_arrays[j]))
        if vol_j > best_vol:
          best_vol = vol_j
          best_parent = j
    parents[i] = best_parent
    if best_parent is not None:
      remaining_children[best_parent] += 1

  num_channels = data.shape[0]
  per_scale_channels: list[list[np.ndarray]] = [
      [] for _ in range(num_scales)
  ]
  out_boxes: list[bounding_box.BoundingBoxBase | None] = [
      None
  ] * num_scales

  shared_valid_mask = (~mask) if mask is not None else None

  for chan_idx in range(num_channels):
    chan = data[chan_idx]
    if dtype == np.float32:
      has_nans = bool(chan.size > 0 and np.isnan(np.sum(chan)))
      nans = None
      if has_nans:
        nans = np.isnan(chan)
        has_nans = bool(np.any(nans))
      if has_nans:
        assert nans is not None
        clean_chan = np.where(nans, np.float32(0.0), chan)
        if mask is not None:
          if not mask_already_applied:
            clean_chan[mask] = 0.0
          valid_mask = ~(nans | mask)
        else:
          valid_mask = ~nans
      else:
        if mask is not None and not mask_already_applied:
          clean_chan = np.where(mask, np.float32(0.0), chan)
        else:
          clean_chan = chan
        valid_mask = shared_valid_mask
    else:
      if mask is not None and not mask_already_applied:
        clean_chan = np.where(mask, dtype.type(0), chan)
      else:
        clean_chan = chan
      valid_mask = shared_valid_mask

    rem_children = list(remaining_children)
    saved_sums: dict[int, np.ndarray] = {}
    saved_counts_3d: dict[int, np.ndarray] = {}
    saved_counts_1d: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for i in order:
      scale = scale_arrays[i]
      scale_zyx = scale[::-1]
      off = offsets[i]
      off_zyx = off[::-1]
      scale_vol = int(np.prod(scale))
      p = parents[i]

      if p is None:
        sliced = clean_chan[off_zyx[0] :, off_zyx[1] :, off_zyx[2] :]
        sums_i = _reduce_block_3d(
            sliced, scale_zyx, dtype, base_vol=1, is_count=False
        )
        if valid_mask is not None:
          valid_sliced = valid_mask[off_zyx[0] :, off_zyx[1] :, off_zyx[2] :]
          counts_3d_i = _reduce_block_3d(
              valid_sliced, scale_zyx, dtype, base_vol=1, is_count=True
          )
          dz_i = dy_i = dx_i = None
        else:
          counts_3d_i = None
          dz_i = _block_lengths_1d(sliced.shape[0], int(scale_zyx[0]))
          dy_i = _block_lengths_1d(sliced.shape[1], int(scale_zyx[1]))
          dx_i = _block_lengths_1d(sliced.shape[2], int(scale_zyx[2]))
      else:
        scale_p = scale_arrays[p]
        off_p = offsets[p]
        rel_scale = scale // scale_p
        rel_scale_zyx = rel_scale[::-1]
        rel_off = (off - off_p) // scale_p
        rel_off_zyx = rel_off[::-1]
        base_vol = int(np.prod(scale_p))

        sums_p = saved_sums[p]
        sliced_sums = sums_p[
            rel_off_zyx[0] :, rel_off_zyx[1] :, rel_off_zyx[2] :
        ]
        sums_i = _reduce_block_3d(
            sliced_sums,
            rel_scale_zyx,
            dtype,
            base_vol=base_vol,
            is_count=False,
        )
        if valid_mask is not None:
          counts_p = saved_counts_3d[p]
          sliced_counts = counts_p[
              rel_off_zyx[0] :, rel_off_zyx[1] :, rel_off_zyx[2] :
          ]
          counts_3d_i = _reduce_block_3d(
              sliced_counts,
              rel_scale_zyx,
              dtype,
              base_vol=base_vol,
              is_count=True,
          )
          dz_i = dy_i = dx_i = None
        else:
          counts_3d_i = None
          dz_p, dy_p, dx_p = saved_counts_1d[p]
          dz_i = _reduce_axis_1d(
              dz_p[rel_off_zyx[0] :], int(rel_scale_zyx[0]), np.dtype(np.int64)
          )
          dy_i = _reduce_axis_1d(
              dy_p[rel_off_zyx[1] :], int(rel_scale_zyx[1]), np.dtype(np.int64)
          )
          dx_i = _reduce_axis_1d(
              dx_p[rel_off_zyx[2] :], int(rel_scale_zyx[2]), np.dtype(np.int64)
          )

        rem_children[p] -= 1
        if rem_children[p] == 0:
          del saved_sums[p]
          if valid_mask is not None:
            del saved_counts_3d[p]
          else:
            del saved_counts_1d[p]

      ret_i = _finalize_block_average(
          sums_i,
          scale_vol,
          dtype,
          dz_i,
          dy_i,
          dx_i,
          counts_3d_i,
          int(scale_zyx[0]),
          int(scale_zyx[1]),
          int(scale_zyx[2]),
      )
      per_scale_channels[i].append(ret_i[np.newaxis, ...])
      if out_boxes[i] is None:
        out_boxes[i] = bounding_box.BoundingBox(
            start=(box.start + off) // scale, size=ret_i.shape[::-1]  # pyrefly: ignore[bad-argument-type]
        )

      if rem_children[i] > 0:
        saved_sums[i] = sums_i
        if valid_mask is not None:
          assert counts_3d_i is not None
          saved_counts_3d[i] = counts_3d_i
        else:
          assert dz_i is not None and dy_i is not None and dx_i is not None
          saved_counts_1d[i] = (dz_i, dy_i, dx_i)

  results: list[tuple[bounding_box.BoundingBoxBase, np.ndarray]] = []
  for i in range(num_scales):
    out_box = out_boxes[i]
    if num_channels == 0:
      # No channels were processed; derive the output geometry directly.
      off_zyx = offsets[i][::-1]
      scale_zyx = scale_arrays[i][::-1]
      out_shape = tuple(
          len(_block_lengths_1d(data.shape[1 + a] - int(off_zyx[a]),
                                int(scale_zyx[a])))
          for a in range(3)
      )
      out_box = bounding_box.BoundingBox(
          start=(box.start + offsets[i]) // scale_arrays[i],
          size=out_shape[::-1],  # pyrefly: ignore[bad-argument-type]
      )
      results.append((out_box, np.empty((0,) + out_shape, dtype=dtype)))
      continue
    assert out_box is not None
    if num_channels == 1:
      merged = per_scale_channels[i][0]
    else:
      merged = np.concatenate(per_scale_channels[i], axis=0)
    results.append((out_box, merged))
  return results
