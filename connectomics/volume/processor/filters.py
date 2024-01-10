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
"""Processors for filtering volumetric images."""

from typing import Optional, Sequence

from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
import numpy as np
from scipy import ndimage


SuggestedXyz = subvolume_processor.SuggestedXyz


class HighPass(subvolume_processor.SubvolumeProcessor):
  """Applies a high-pass filter to image data.

  This is useful to make structures more visible at high zoom-out level.

  The method implemented here was originally used by Yuelong Wu @ the Harvard
  Lichtman lab for low-resolution visualization of the h01 dataset.
  """

  def process(self, subvol: subvolume.Subvolume) -> subvolume.SubvolumeOrMany:
    box = subvol.bbox
    input_ndarray = subvol.data
    img = input_ndarray[0, 0, ...]
    hp = img + ndimage.minimum_filter(255 - img, 3)
    return self.crop_box_and_data(box, hp[np.newaxis, np.newaxis, ...])

  def context(self):
    return SuggestedXyz(3, 3, 0), SuggestedXyz(3, 3, 0)

  def subvolume_size(self):
    return SuggestedXyz(1024, 1024, 1)


class ApplyFilter(subvolume_processor.SubvolumeProcessor):
  """Applies an ndimage filter to the volume."""

  def __init__(
      self,
      filter_size: int | Sequence[int],
      filter_type: str,
      mode: str,
      dtype: Optional[str] = None,
  ):
    """Constructor.

    Args:
      filter_size: size of the filter kernel in pixels (int or [z]yx tuple)
      filter_type: one of: median, max, min, uniform
      mode: '2d' (filter will be applied to XY sections) or '3d'
      dtype: dtype for computation and output data; defaults to same as input;
        if specified, input data will be cast to this type prior to filtering
    """
    super().__init__()

    filter_map = {
        'median': ndimage.median_filter,
        'max': ndimage.maximum_filter,
        'min': ndimage.minimum_filter,
        'uniform': ndimage.uniform_filter,
    }
    self._filter_type = filter_type
    self._filter_fn = filter_map[filter_type]
    self._mode = mode
    if dtype is not None:
      self._dtype = np.dtype(dtype).type
    else:
      self._dtype = None

    if isinstance(filter_size, int):
      s = filter_size
      if self._mode == '2d':
        self._filter_size = (1, s, s)
      else:
        self._filter_size = (s, s, s)
    else:
      if self._mode == '2d':
        self._filter_size = (1, filter_size[0], filter_size[1])
      else:
        self._filter_size = filter_size

  @property
  def name_parts(self):
    return (
        self._filter_type,
        'filter',
        '_'.join(str(x) for x in self._filter_size),
        self._mode,
    )

  def output_type(self, input_type):
    if self._dtype is None:
      return input_type
    else:
      return self._dtype

  def context(self):
    flt = np.array(self._filter_size[::-1], dtype=int)
    pre, post = (flt // 2).tolist(), (flt - flt // 2).tolist()
    if self._mode == '2d':
      pre[-1] = post[-1] = 0
    return pre, post

  def process(self, subvol: subvolume.Subvolume) -> subvolume.SubvolumeOrMany:
    box = subvol.bbox
    input_ndarray = subvol.data
    if input_ndarray.shape[0] != 1:
      raise ValueError('Only 1-channel volumes are supported.')

    out_dtype = self.output_type(input_ndarray.dtype)
    input_ndarray = input_ndarray.astype(out_dtype)
    out = np.zeros_like(input_ndarray)
    out[0, ...] = self._filter_fn(input_ndarray[0, ...], self._filter_size)
    return self.crop_box_and_data(box, out)
