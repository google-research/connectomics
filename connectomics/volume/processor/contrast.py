# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
"""Processors for contrast adjustment."""

from connectomics.common import geom_utils
from connectomics.volume import mask as mask_lib
from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
import numpy as np
from scipy import ndimage
import skimage.exposure


class PlaneProcessor(subvolume_processor.SubvolumeProcessor):
  """Abstract base class for plane-wise Processors."""

  def __init__(
      self, plane='yx', mask_configs: str | mask_lib.MaskConfigs | None = None
  ):
    """Constructor.

    Args:
      plane: 2-char string describing the plane within which to compute the
        transformation; one of: (yx, zy, zx)
      mask_configs: Optional mask where positive values indicate regions of the
        image that should not contribute to contrast estimation.
    """
    super().__init__()
    assert len(set(plane)) == 2
    assert not set(plane) - set('xyz')
    self._plane = plane
    if mask_configs is not None and isinstance(mask_configs, str):
      mask_configs = self._get_mask_configs(mask_configs)

    self._mask_configs = mask_configs

  def process_plane(self, image2d: np.ndarray):
    raise NotImplementedError()

  def process(self, subvol: subvolume.Subvolume) -> subvolume.SubvolumeOrMany:
    box = subvol.bbox
    input_ndarray = subvol.data
    dims = 'czyx'
    other = dims.index(list(set('xyz') - set(self._plane))[0])
    plane = [dims.index(x) for x in self._plane]

    mask = None
    if self._mask_configs is not None:
      mask = self._build_mask(self._mask_configs, box)

    desired = [0, other, plane[0], plane[1]]
    transposed = np.ascontiguousarray(np.transpose(input_ndarray, desired))
    if mask is not None:
      mask = np.ascontiguousarray(np.transpose(mask, np.array(desired[1:]) - 1))

    output_dtype = self.output_type(input_ndarray.dtype)
    if output_dtype == input_ndarray.dtype:
      output = transposed  # Process in place.
    else:
      output = np.zeros_like(transposed, dtype=output_dtype)

    rng = np.random.default_rng(42)
    for c in range(transposed.shape[0]):
      for z in range(transposed.shape[1]):
        m = orig = None
        inp = transposed[c, z, ...]

        # Fill masked areas with pixels randomly resampled from unmasked areas.
        # The contrast normalization methods we use do not rely on the spatial
        # structure of the image, just the local histogram.
        if mask is not None:
          inp = inp.copy()
          m = mask[z, ...]
          if np.any(m) and not np.all(m):
            fill = rng.choice(inp[~m], size=np.sum(m), replace=True)
            orig = inp[m]
            inp[m] = fill

        output[c, z, ...] = self.process_plane(inp)

        if orig is not None:
          output[c, z, ...][m] = orig

    output = np.transpose(output, np.argsort(desired))
    return self.crop_box_and_data(box, output)


class CLAHE(PlaneProcessor):
  """Applies CLAHE plane-wise."""

  crop_at_borders = False

  def __init__(
      self,
      plane='yx',
      kernel_size=None,
      clip_limit=0.01,
      clip_min=None,
      clip_max=None,
      invert=False,
      mask_configs: str | mask_lib.MaskConfigs | None = None,
  ):
    """Constructor.

    Args:
      plane: Forwarded to PlaneProcessor.
      kernel_size: Forwarded to equalize_adapthist.
      clip_limit: Forwarded to equalize_adapthist.
      clip_min: Minimum value to retain in the input to CLAHE.
      clip_max: Maximum value to retain in the input to CLAHE.
      invert: Whether to invert the CLAHE result.
      mask_configs: Optional mask where positive values indicate regions of the
        image that should not contribute to contrast estimation.
    """
    super(CLAHE, self).__init__(plane, mask_configs)
    self._kernel_size = kernel_size
    self._clip_limit = clip_limit
    self._invert = invert
    self._clip_max = clip_max
    self._clip_min = clip_min

  def output_type(self, input_type):
    return np.uint8

  def process_plane(self, image2d: np.ndarray) -> np.ndarray:
    if len(set(np.unique(image2d))) == 1:
      return image2d

    if self._clip_min is not None or self._clip_max is not None:
      c_min = self._clip_min if self._clip_min is not None else -np.inf
      c_max = self._clip_max if self._clip_max is not None else np.inf
      image2d = np.clip(image2d, c_min, c_max)

    clahed = skimage.exposure.equalize_adapthist(
        image2d, kernel_size=self._kernel_size, clip_limit=self._clip_limit
    )
    if self._invert:
      clahed = 1.0 - clahed
    return (clahed * 255).astype(np.uint8)


class LCN(PlaneProcessor):
  """Applies Local Contrast Normalization plane-wise."""

  crop_at_borders = False

  def __init__(self, plane='yx', disk_radius=100):
    super(LCN, self).__init__(plane)
    self._selem = skimage.morphology.disk(disk_radius)

  def process_plane(self, image2d):
    return skimage.filters.rank.equalize(image2d, footprint=self._selem)


class SectionStd(PlaneProcessor):
  """Computes standard deviation plane-wise.

  Image statistics are computed in a moving window of size 2*block_radius + 1.
  """

  crop_at_borders = False

  def __init__(self, plane='yx', block_radius=20):
    super(SectionStd, self).__init__(plane)
    self._block_r = block_radius

  def context(self):
    pl = set(self._plane)
    if pl == set('xy'):
      ctx = (self._block_r, self._block_r, 0)
    elif pl == set('zx'):
      ctx = (self._block_r, 0, self._block_r)
    else:
      ctx = (0, self._block_r, self._block_r)

    return ctx, ctx

  def _get_mean_and_std(self, image_f64: np.ndarray) -> tuple[float, float]:
    """Computes mean and std within a pixel-centered block."""
    block_shape = (self._block_r * 2 + 1, self._block_r * 2 + 1)
    total = geom_utils.query_integral_image(
        geom_utils.integral_image(image_f64), block_shape
    )
    total_sq = geom_utils.query_integral_image(
        geom_utils.integral_image(np.square(image_f64)), block_shape
    )

    area = np.prod(block_shape)
    mean = total / area

    var = 1.0 / (area - 1) * total_sq - 1.0 / (area**2 - area) * total**2
    std = np.sqrt(var)
    std[var < 0] = 0.0
    return mean, std

  def process_plane(self, image2d: np.ndarray) -> np.ndarray:
    _, std = self._get_mean_and_std(image2d.astype(np.float64) / 255.0)
    # Max stdev of the normalized image is 0.5.
    return np.pad(
        np.clip(std / 0.5 * 255, 0, 255).astype(np.uint8),
        [[self._block_r, self._block_r], [self._block_r, self._block_r]],
        mode='constant',
    )


class VarianceOfLaplacian(SectionStd):
  """Computes the stddev of the Laplacian of the Gaussian-filtered input.

  This is useful for detecting defocused areas in EM datasets. Low values
  correlate with out of focus areas (e.g. < 55 with the default settings,
  though the specific value is dataset- and resolution-dependent -- tune
  the threshold by visually browsing the output volume).
  """

  crop_at_borders = False

  def __init__(self, plane='yx', block_radius=10, sigma=1.0, scale=64):
    super(VarianceOfLaplacian, self).__init__(plane, block_radius)
    self._sigma = sigma
    self._scale = scale

  def process_plane(self, image2d: np.ndarray) -> np.ndarray:
    glp = ndimage.gaussian_laplace(
        image2d.astype(np.float64), sigma=self._sigma
    )
    _, std = self._get_mean_and_std(glp)
    return np.pad(
        np.clip(std / self._scale * 255, 0, 255).astype(np.uint8),
        [[self._block_r, self._block_r], [self._block_r, self._block_r]],
        mode='constant',
    )


class CLLCN(SectionStd):
  """Applies Contrast Limited LCN plane-wise.

  Implementation follows:
    https://github.com/saalfeldlab/hot-knife/blob/surface-fitting/src/main/java/org/janelia/saalfeldlab/hotknife/ops/CLLCN.java
  """

  crop_at_borders = False

  def __init__(
      self,
      plane='yx',
      block_radius=1023,
      mean_factor=3.0,
      limit=10,
      gamma=0.5,
  ):
    super(CLLCN, self).__init__(plane, block_radius)
    self._mean_factor = mean_factor
    self._limit = limit
    self._gamma = gamma

  def process_plane(self, image2d: np.ndarray) -> np.ndarray:
    def _limit(x):
      grad_1p = self._gamma ** (1.0 / (1.0 - self._gamma))
      ret = (
          (x + grad_1p - self._limit) ** self._gamma
          + self._limit  #
          - grad_1p**self._gamma
      )
      return np.select([x >= self._limit], [ret], x)

    image_f64 = image2d.astype(np.float64) / 255.0
    img_range = 1.0  # image values are [0..img_range]
    mean, std = self._get_mean_and_std(image_f64)

    # Maps [mean - lim(mean_fac * std), mean + lim(mean_fac * std)] -> [0..255].
    d = self._mean_factor * std
    s = _limit(1.0 / d * img_range)
    s[d == 0] = 0
    min_ = np.nan_to_num(mean - img_range / s)

    # Cut image down to the same shape as the output of query_integral_image.
    image_f64 = image_f64[
        self._block_r : -self._block_r, self._block_r : -self._block_r
    ]
    ret = (image_f64 - min_) * s * 0.5

    # Pad back with zeros to compensate data that was removed by integral image
    # querying.
    return np.pad(
        np.clip(ret * 255, 0, 255).astype(np.uint8),
        [[self._block_r, self._block_r], [self._block_r, self._block_r]],
        mode='constant',
    )
