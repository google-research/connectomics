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

from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.common import geom_utils
import numpy as np


class DownsampleAreaTest(absltest.TestCase):

  def test_uniform_data_exact(self):
    """A constant volume should downsample to the same constant."""
    shape = (6, 8, 10)
    val = 42
    data = np.full(shape, val, dtype=np.int64)
    scale = np.array([2, 2, 2])
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=shape[::-1])

    svt = geom_utils.integral_image(data)
    _, out = geom_utils.downsample_area(svt, box, scale, np.dtype(np.uint8))
    np.testing.assert_array_equal(out, val)

  def test_uniform_data_with_remainder(self):
    """A constant volume with remainders should still downsample to the same constant."""
    shape = (5, 7, 9)
    val = 100
    data = np.full(shape, val, dtype=np.int64)
    scale = np.array([2, 2, 2])
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=shape[::-1])

    svt = geom_utils.integral_image(data)
    _, out = geom_utils.downsample_area(svt, box, scale, np.dtype(np.uint8))
    np.testing.assert_array_equal(out, val)

  def test_output_box_coordinates(self):
    """Output box start is correctly aligned to the downsampled grid."""
    shape = (5, 7, 9)
    data = np.ones(shape, dtype=np.int64)
    scale = np.array([2, 2, 2])
    box = bounding_box.BoundingBox(start=(10, 20, 30), size=shape[::-1])

    svt = geom_utils.integral_image(data)
    out_box, out = geom_utils.downsample_area(
        svt, box, scale, np.dtype(np.uint8)
    )

    # The output box start should be in downsampled coordinates:
    # (box.start + off) // scale
    np.testing.assert_array_equal(out_box.start, [5, 10, 15])
    np.testing.assert_array_equal(out_box.size, np.array(out.shape[1:][::-1]))

  def test_downsample_area_scales_matches_svt(self):
    """Direct multi-scale pyramid matches SVT-based downsample_area exactly."""
    rng = np.random.default_rng(123)
    for shape in [(16, 20, 24), (13, 17, 19), (1, 33, 35)]:
      for start in [(0, 0, 0), (3, 5, 7)]:
        box = bounding_box.BoundingBox(start=start, size=shape[::-1])
        scales = [
            np.array([2, 2, 2]),
            np.array([4, 4, 4]),
            np.array([2, 2, 1]),
            np.array([4, 4, 1]),
            np.array([3, 5, 2]),
        ]
        mask = rng.random(shape) < 0.15
        mask_svt = geom_utils.integral_image(mask.astype(np.int64))

        for dtype in (np.uint8, np.uint32):
          data = rng.integers(0, 250, size=(2,) + shape, dtype=dtype)
          for use_mask in (False, True):
            cur_mask = mask if use_mask else None
            cur_mask_svt = mask_svt if use_mask else None
            work = data.copy()
            if use_mask:
              work[:, mask] = 0
            results = geom_utils.downsample_area_scales(
                work,
                box,
                scales,
                mask=cur_mask,
                mask_already_applied=use_mask,
            )
            svts = [
                geom_utils.integral_image(work[c].astype(np.int64))
                for c in range(work.shape[0])
            ]
            for (out_box, out_arr), scale in zip(results, scales):
              expected_chans = []
              expected_box = None
              for svt in svts:
                expected_box, exp_c = geom_utils.downsample_area(
                    svt, box, scale, np.dtype(dtype), cur_mask_svt
                )
                expected_chans.append(exp_c)
              expected = np.concatenate(expected_chans, axis=0)
              self.assertEqual(out_box, expected_box)
              np.testing.assert_array_equal(out_arr, expected)

        # Also test float32 with NaNs and mask.
        fdata = rng.uniform(0.0, 100.0, size=(1,) + shape).astype(np.float32)
        fdata[0, rng.random(shape) < 0.1] = np.nan
        for use_mask in (False, True):
          cur_mask = mask if use_mask else None
          work = fdata.copy()
          if use_mask:
            work[:, mask] = 0
          results = geom_utils.downsample_area_scales(
              work,
              box,
              scales,
              mask=cur_mask,
              mask_already_applied=use_mask,
          )
          clean = np.nan_to_num(work[0].astype(np.float64))
          svt = geom_utils.integral_image(clean)
          nans = np.isnan(fdata[0])
          if use_mask:
            nans = nans | mask
          f_mask_svt = (
              geom_utils.integral_image(nans.astype(np.int64))
              if np.any(nans)
              else None
          )
          for (out_box, out_arr), scale in zip(results, scales):
            expected_box, expected = geom_utils.downsample_area(
                svt, box, scale, np.dtype(np.float32), f_mask_svt
            )
            self.assertEqual(out_box, expected_box)
            np.testing.assert_allclose(
                out_arr, expected, rtol=1e-5, atol=1e-5, equal_nan=True
            )

  def test_downsample_area_masked_integer_block_is_zero(self):
    """Fully masked blocks yield 0 for integer outputs even if data is unzeroed."""
    data = np.full((4, 4, 4), 200, dtype=np.int64)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[:2, :2, :2] = True
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=(4, 4, 4))
    _, out = geom_utils.downsample_area(
        geom_utils.integral_image(data),
        box,
        np.array([2, 2, 2]),
        np.dtype(np.uint8),
        geom_utils.integral_image(mask.astype(np.int64)),
    )
    self.assertEqual(out[0, 0, 0, 0], 0)
    self.assertEqual(out[0, 1, 1, 1], 200)


def _reference_downsample(
    data: np.ndarray,
    box: bounding_box.BoundingBox,
    scale: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[bounding_box.BoundingBox, np.ndarray]:
  """Brute-force block-mean reference for downsample_area_scales.

  Masked and NaN voxels are excluded regardless of their stored values. Fully
  excluded blocks are NaN for floating outputs and 0 for integer outputs.

  Args:
    data: [C, Z, Y, X] input array.
    box: bounding box of the input.
    scale: xyz downsampling factors.
    mask: optional [Z, Y, X] mask of excluded voxels.

  Returns:
    Expected (out_box, [C, z', y', x'] array).
  """
  off = box.start - box.start // scale * scale
  off[off > 0] = (scale - off)[off > 0]
  sz, sy, sx = (int(s) for s in scale[::-1])
  oz, oy, ox = (int(o) for o in off[::-1])
  _, nz, ny, nx = data.shape
  out_shape = tuple(
      max(0, -(-(n - o) // s))
      for n, o, s in zip((nz, ny, nx), (oz, oy, ox), (sz, sy, sx))
  )
  out = np.zeros((data.shape[0],) + out_shape, dtype=data.dtype)
  is_float = np.issubdtype(data.dtype, np.floating)
  for c in range(data.shape[0]):
    for k, j, i in np.ndindex(*out_shape):
      zs = slice(oz + k * sz, oz + (k + 1) * sz)
      ys = slice(oy + j * sy, oy + (j + 1) * sy)
      xs = slice(ox + i * sx, ox + (i + 1) * sx)
      block = data[c, zs, ys, xs].astype(np.float64)
      valid = ~np.isnan(block)
      if mask is not None:
        valid &= ~mask[zs, ys, xs]
      n = int(valid.sum())
      if n == 0:
        out[c, k, j, i] = np.nan if is_float else 0
      else:
        out[c, k, j, i] = np.round(block[valid].sum() / n)
  out_box = bounding_box.BoundingBox(
      start=((box.start + off) // scale).tolist(), size=out_shape[::-1]
  )
  return out_box, out


class DownsampleAreaScalesTest(absltest.TestCase):

  def _check(self, data, box, scales, mask=None, mask_already_applied=False):
    results = geom_utils.downsample_area_scales(
        data, box, scales, mask=mask, mask_already_applied=mask_already_applied
    )
    self.assertLen(results, len(scales))
    for (out_box, out_arr), scale in zip(results, scales):
      exp_box, expected = _reference_downsample(
          data, box, np.asarray(scale), mask
      )
      self.assertEqual(out_box, exp_box, msg=f'scale={scale}')
      self.assertEqual(out_arr.dtype, data.dtype)
      np.testing.assert_array_equal(out_arr, expected, err_msg=f'{scale=}')

  def test_uint8_max_values_wide_accumulators(self):
    """Block volumes >255 (uint32 sums) and steps >8 (reduceat path)."""
    rng = np.random.default_rng(0)
    shape = (3, 40, 37)
    box = bounding_box.BoundingBox(start=(1, 2, 0), size=shape[::-1])
    scales = [
        np.array([16, 16, 1]),
        np.array([17, 15, 1]),
        np.array([32, 32, 2]),
        np.array([9, 10, 3]),
    ]
    full = np.full((1,) + shape, 255, dtype=np.uint8)
    self._check(full, box, scales)
    rand = rng.integers(200, 256, size=(2,) + shape, dtype=np.uint8)
    self._check(rand, box, scales)
    mask = rng.random(shape) < 0.2
    self._check(rand, box, scales, mask=mask)

  def test_uint32_large_values(self):
    rng = np.random.default_rng(1)
    shape = (4, 33, 34)
    box = bounding_box.BoundingBox(start=(0, 3, 1), size=shape[::-1])
    scales = [np.array([2, 2, 1]), np.array([16, 16, 2]), np.array([11, 9, 4])]
    info = np.iinfo(np.uint32)
    full = np.full((1,) + shape, info.max, dtype=np.uint32)
    self._check(full, box, scales)
    rand = rng.integers(
        info.max - 1000, info.max, size=(1,) + shape, dtype=np.uint32,
        endpoint=True,
    )
    self._check(rand, box, scales)
    mask = rng.random(shape) < 0.3
    self._check(rand, box, scales, mask=mask)

  def test_float32_large_magnitude_matches_float64(self):
    """Float32 sums are exact in float64 but would round in float32."""
    rng = np.random.default_rng(2)
    shape = (8, 24, 26)
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=shape[::-1])
    scales = [
        np.array([2, 2, 2]),
        np.array([4, 4, 1]),
        np.array([8, 8, 8]),
        np.array([16, 16, 4]),
    ]
    # Multiples of 1/16 up to 2**20 are exact in float32, and their block sums
    # are exact in float64 but not in float32.
    data = (
        rng.integers(0, 2**24, size=(1,) + shape).astype(np.float32) / 16
    ).astype(np.float32)
    self._check(data, box, scales)
    data[0, rng.random(shape) < 0.1] = np.nan
    self._check(data, box, scales)
    mask = rng.random(shape) < 0.2
    self._check(data, box, scales, mask=mask)

  def test_mask_not_applied(self):
    """Unzeroed data under the mask is excluded when mask_already_applied=False."""
    rng = np.random.default_rng(3)
    shape = (6, 13, 14)
    box = bounding_box.BoundingBox(start=(1, 1, 1), size=shape[::-1])
    scales = [np.array([2, 2, 2]), np.array([4, 4, 2]), np.array([3, 3, 1])]
    mask = rng.random(shape) < 0.3
    mask[:2, :4, :4] = True  # Guarantee some fully masked blocks.
    for dtype in (np.uint8, np.uint32):
      data = rng.integers(1, 250, size=(2,) + shape, dtype=dtype)
      self._check(data, box, scales, mask=mask, mask_already_applied=False)
    fdata = rng.uniform(1.0, 100.0, size=(1,) + shape).astype(np.float32)
    self._check(fdata, box, scales, mask=mask, mask_already_applied=False)
    fdata[0, rng.random(shape) < 0.1] = np.nan
    self._check(fdata, box, scales, mask=mask, mask_already_applied=False)

  def test_duplicate_and_unsorted_scales(self):
    rng = np.random.default_rng(4)
    shape = (9, 17, 18)
    box = bounding_box.BoundingBox(start=(3, 2, 1), size=shape[::-1])
    scales = [
        np.array([8, 8, 4]),
        np.array([2, 2, 1]),
        np.array([4, 4, 2]),
        np.array([2, 2, 1]),
        np.array([8, 8, 4]),
        np.array([4, 4, 1]),
    ]
    data = rng.integers(0, 256, size=(2,) + shape, dtype=np.uint8)
    self._check(data, box, scales)
    mask = rng.random(shape) < 0.2
    self._check(data, box, scales, mask=mask)

  def test_empty_inputs(self):
    """Offsets >= size, zero-length axes, and zero channels give empty output."""
    scales = [np.array([4, 4, 4]), np.array([8, 8, 8]), np.array([2, 2, 1])]
    # Offset >= size along z: start z=1 with scale 8 needs offset 7 > 5.
    shape = (5, 16, 16)
    box = bounding_box.BoundingBox(start=(0, 0, 1), size=shape[::-1])
    data = np.ones((1,) + shape, dtype=np.uint8)
    self._check(data, box, scales)
    self._check(data, box, scales, mask=np.zeros(shape, dtype=bool))

    # Zero-length spatial axis.
    shape = (0, 8, 8)
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=shape[::-1])
    for dtype in (np.uint8, np.uint32, np.float32):
      data = np.ones((2,) + shape, dtype=dtype)
      self._check(data, box, scales)

    # Zero channels.
    shape = (8, 9, 10)
    box = bounding_box.BoundingBox(start=(1, 0, 0), size=shape[::-1])
    data = np.ones((0,) + shape, dtype=np.uint8)
    self._check(data, box, scales)
    self._check(data, box, scales, mask=np.zeros(shape, dtype=bool))


class IntegralImageTest(absltest.TestCase):

  def test_2d(self):
    data = np.array([[1, 2], [3, 4]])
    ii = geom_utils.integral_image(data)
    # Top-left origin should be padded with zeros.
    self.assertEqual(ii[0, 0], 0)
    self.assertEqual(ii[0, 1], 0)
    self.assertEqual(ii[1, 0], 0)
    # Full sum.
    self.assertEqual(ii[2, 2], 10)

  def test_3d(self):
    data = np.ones((2, 3, 4), dtype=int)
    ii = geom_utils.integral_image(data)
    self.assertEqual(ii[-1, -1, -1], 24)


class QueryIntegralImageTest(absltest.TestCase):

  def test_2d_uniform(self):
    data = np.ones((4, 6), dtype=int)
    ii = geom_utils.integral_image(data)
    result = geom_utils.query_integral_image(ii, diam=[2, 3])
    # Each 2x3 block sums to 6.
    np.testing.assert_array_equal(result, 6)

  def test_3d_with_stride(self):
    data = np.ones((4, 6, 8), dtype=int)
    ii = geom_utils.integral_image(data)
    result = geom_utils.query_integral_image(
        ii, diam=[2, 2, 2], stride=[2, 2, 2]
    )
    np.testing.assert_array_equal(result, 8)


if __name__ == '__main__':
  absltest.main()
