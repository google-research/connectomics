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
"""A 4D subvolume."""

from __future__ import annotations

import typing
from typing import Optional, Union

from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.common import box_generator
import numpy as np


def _check_bbox_dims(bbox: bounding_box.BoundingBox, data: np.ndarray):
  bbox_shape = bbox.size[::-1]
  data_shape = data.shape
  if len(bbox_shape) != 4:
    data_shape = data.shape[1:]
  if not np.all(bbox_shape == data_shape):
    raise ValueError(
        f'New bbox does not match data shape: {bbox_shape} vs {data_shape}')


class RelativeSubvolumeIndexer:
  """Index into a subvolume via relative coordinates."""

  _subvol: 'Subvolume'

  def __init__(self, subvol: 'Subvolume'):
    self._subvol = subvol

  def __getitem__(self, ind: array.ArbitrarySlice) -> 'Subvolume':
    slices = array.normalize_index(ind, self._subvol.shape)
    slices = typing.cast(array.CanonicalSlice, slices)
    new_data = self._subvol.data[slices]
    offset = np.array([x.start if x.start is not None else 0 for x in slices])
    return Subvolume(
        new_data,
        self._subvol.bbox.__class__(
            start=self._subvol.start + offset[3:0:-1],
            size=new_data.shape[3:0:-1]))


class AbsoluteSubvolumeIndexer:
  """Index into a subvolume via absolute coordinates."""

  _subvol: 'Subvolume'

  def __init__(self, subvol: 'Subvolume'):
    self._subvol = subvol

  def __getitem__(self, ind: array.ArbitrarySlice) -> 'Subvolume':
    abs_end = (self._subvol.data.shape[0],) + tuple(self._subvol.bbox.end[::-1])
    slices = array.normalize_index(ind, abs_end)
    slices = typing.cast(array.CanonicalSlice, slices)
    adjusted = [slices[0]]
    for i, slc in enumerate(slices[1:]):
      start = slc.start - self._subvol.start[2 - i]
      start = max(start, 0)
      stop = slc.stop - self._subvol.start[2 - i]
      stop = min(stop, self._subvol.bbox.end[2 - i])
      new_slice = slice(start, stop)
      adjusted.append(new_slice)
    return self._subvol[adjusted]  # pytype: disable=unsupported-operands  # dynamic-method-lookup


class Subvolume:
  """A 4D view of subvolume.

  This class is intended to be a "view" into an underlying Volume. Does not copy
  the underlying ndarray; any modifications will propagate to the wrapped data.
  """
  # TODO(timblakely): Do we want to have an ImmutableSubvolume for any reason?
  _data: np.ndarray
  _bbox: bounding_box.BoundingBox

  def __init__(self,
               data_or_subvolume: Union[np.ndarray, 'Subvolume'],
               bbox: Optional[bounding_box.BoundingBox] = None):
    """Create a new Subvolume.

    Args:
      data_or_subvolume: Data to wrap, either an ndarray or an existing
        subvolume.
      bbox: Bounding box for this subvolume. Must match the XYZ size of the
        data.

    Raises:
      ValueError: On invalid combination of constructor arguments.
    """
    if isinstance(data_or_subvolume, Subvolume):
      sv = data_or_subvolume
      if not sv.valid:
        raise ValueError(
            'Attempted to create a subvolume using an invalid subvolume.')
      if bbox is not None:
        raise ValueError('Construction requires either an ndarray and bbox, '
                         'or another subvolume.')
      self._data = sv._data
      self._bbox = sv._bbox
    else:
      data = data_or_subvolume
      if bbox is None:
        raise ValueError(
            'Initializing a subvolume with an ndarray requires a bounding box')
      _check_bbox_dims(bbox, data)
      self._data = data
      self._bbox = bbox

  def __getitem__(self, ind: array.ArbitrarySlice) -> 'Subvolume':
    return RelativeSubvolumeIndexer(self)[ind]

  def __eq__(self, other: Union['Subvolume', np.ndarray, int, float]):
    if isinstance(other, np.ndarray) or isinstance(other, int) or isinstance(
        other, float):
      return np.all(self._data == other)
    return self.bbox == other.bbox and np.all(self._data == other.data)

  @property
  def bbox(self) -> bounding_box.BoundingBox:
    """Bounding box."""
    return self._bbox

  @property
  def data(self) -> np.ndarray:
    """Underlying data."""
    return self._data

  @property
  def shape(self) -> tuple[int, int, int, int]:
    """4d shape of the underlying data in CZYX format."""
    return self._data.shape

  @property
  def start(self) -> np.ndarray:
    """Starting corner of the bounding box."""
    return self._bbox.start

  @property
  def size(self) -> np.ndarray:
    """3d size of the subvolume."""
    return self._bbox.size

  @property
  def index_abs(self) -> AbsoluteSubvolumeIndexer:
    """Index into a subvolume via absolute coordinates."""
    return AbsoluteSubvolumeIndexer(self)

  @property
  def valid(self) -> bool:
    """Return whether this subvolume contains valid data."""
    return len(self.shape) == 4 and np.all(self.size > 0)

  def new_bounding_box(self, bbox: bounding_box.BoundingBox) -> 'Subvolume':
    _check_bbox_dims(bbox, self._data)
    self._bbox = bbox
    return self

  def clip(self, bbox: bounding_box.BoundingBox) -> 'Subvolume':
    return self[bbox.to_slice4d()]

  def clip_abs(self, bbox: bounding_box.BoundingBox) -> 'Subvolume':
    return self.index_abs[bbox.to_slice4d()]

  def merge_with(self,
                 data: Union[np.ndarray, 'Subvolume'],
                 empty_value: Union[int, float] = 0) -> 'Subvolume':
    """Fill data only where values are considered empty.

    If a subvolume is passed, bounding box intersection is performed and only
    overlapping regions are attempted to be filled.

    Args:
      data: Data to fill on empty (ndarray or Subvolume).
      empty_value: Data inside of self.data that should be considered empty.

    Returns:
      self after filling.

    Raises:
      ValueError: If the output shape does not match the input data shape.
    """
    target_sv = self

    if isinstance(data, Subvolume):
      sv: Subvolume = data
      overlapping = self.bbox.intersection(sv.bbox)
      if overlapping is None:
        return self
      if self.bbox == overlapping:
        target_data = sv.data
      else:
        # Need to clip subvolume data to only relevant parts. Since Subvolumes
        # are only views, we can just target a subvolume within self. This
        # avoids creating a copy of the data prior to filling, which can be
        # expensive on larger subvolumes.
        target_sv = self.index_abs[overlapping.to_slice4d()]
        target_data = sv.index_abs[overlapping.to_slice4d()].data
    else:
      target_data = data

    _check_bbox_dims(target_sv.bbox, target_data)
    if not np.all(target_sv.shape == target_data.shape):
      raise ValueError(f'Dimension mismatch in shape: {target_sv.shape} vs '
                       f'{target_data.shape}')
    if np.isnan(empty_value):
      mask = np.isnan(target_sv.data)
    else:
      mask = target_sv.data == empty_value
    if not np.any(mask):
      return self
    target_sv.data[mask] = target_data[mask]

    # Intentionally return self here.
    return self

  def split(self,
            size: np.ndarray,
            origin: Optional[np.ndarray] = None,
            clip_output_subvolumes: bool = False,
            empty_value: Union[int, float] = 0) -> list['Subvolume']:
    """Split a subvolume into smaller, non-overlapping subvolumes.

    Args:
      size: Size of the smaller subvolumes to generate.
      origin: Origin of the grid to split the subvolume in global coordinates.
        If no origin is passed, the beginning of the grid divisions begin at the
        start corner of the subvolume.
      clip_output_subvolumes: If true, final subvolumes will be clipped
        according to self.bbox. If clipping is not applied, empty spaces will be
        filled with empty_value.
      empty_value: Fill value for overlapping, non-clipped subvolumes.

    Returns:
      List of split subvolumes.
    """
    if origin is None:
      # Use this subvolume's start corner.
      start_corner = self.start
    else:
      # Begin at the first origin-aligned-gridded subvolume that comes in
      # contact with this subvolume.
      start_corner = origin + ((self.start - origin) // size * size)
    overlapping_box_limits = bounding_box.BoundingBox(
        start=start_corner, end=self.start + self.size)
    gen = box_generator.BoxGenerator(overlapping_box_limits, size)
    new_subvols = []
    for box in gen.boxes:
      # Handle back edges
      if not clip_output_subvolumes and not np.all(box.size == size):
        box = bounding_box.BoundingBox(box.start, size)
      split_data = self.index_abs[box.to_slice4d()]
      empty_data = np.full([split_data.shape[0]] + list(box.size[::-1]),
                           empty_value)
      new_subvol = Subvolume(empty_data, box)
      new_subvol.merge_with(split_data, empty_value)
      if clip_output_subvolumes:
        new_subvol = new_subvol.clip_abs(self.bbox)
      new_subvols.append(new_subvol)
    return new_subvols


SubvolumeOrMany = Subvolume | list[Subvolume]
