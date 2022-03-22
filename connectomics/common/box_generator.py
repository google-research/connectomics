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
"""Generate overlapping boxes mapped to a coordinate space."""

import bisect
import functools
import itertools
from typing import List, Optional, Sequence, Tuple, Iterable, TypeVar, Union

from connectomics.common import array
from connectomics.common import bounding_box
import numpy as np

S = TypeVar('S', bound='BoxGenerator')
BoxIndexCoordinates = Tuple[int, ...]
IndexedBoundingBox = Tuple[BoxIndexCoordinates, bounding_box.BoundingBoxBase]
BoxIndex = TypeVar('BoxIndex', bound=int)


class BoxGeneratorBase:
  """Base class for BoundingBox generators."""

  @property
  def num_boxes(self) -> int:
    raise NotImplementedError()

  @property
  def box_size(self) -> array.ImmutableArray:
    raise NotImplementedError()

  @property
  def box_overlap(self) -> array.ImmutableArray:
    raise NotImplementedError()

  def generate(self, index: int) -> IndexedBoundingBox:
    raise NotImplementedError()

  def index_to_cropped_box(self, intindex: int) -> bounding_box.BoundingBox:
    raise NotImplementedError()


class BoxGenerator(BoxGeneratorBase):
  """Generates overlapping sub-boxes spanning the input bounding box."""

  def __init__(self,
               outer_box: bounding_box.BoundingBoxBase,
               box_size: Sequence[Union[int, float]],
               box_overlap: Optional[Sequence[Union[int, float]]] = None,
               back_shift_small_boxes: bool = False):
    """Initialize a generator.

    Args:
      outer_box: Volume to subdivide.
      box_size: N-D sequence giving desired 3d size of each sub-box. Smaller
        sub-boxes may be included at the back edge of the volume, but not if
        they are smaller than overlap (in that case they are completely included
        in the preceding box). If an element is None, the entire range available
        within outer_box is used for that dimension.
      box_overlap: N-D sequence giving the overlap between neighboring
        sub-boxes. Must be < box_size.
      back_shift_small_boxes: If True, do not produce undersized boxes at the
        back edge of the outer_box.  Instead, shift the start of these boxes
        back so that they can maintain sub_box_size.  This means that the boxes
        at the back edge will have more overlap than the rest of the boxes.

    Raises:
      ValueError: If box size is incompatible with outer box rank.
      ValueError: If box overlap is incompatible with outer box rank.
      ValueError: If box size is <= overlap.
    """
    # normalize box_size
    box_size = list(box_size)
    squeeze = []
    for i, x in enumerate(box_size):
      if x == 1:
        squeeze.append(i)
      if x is None:
        box_size[i] = outer_box.size[i]

    box_size = np.array(box_size)
    if outer_box.rank != box_size.size:
      raise ValueError('box_size incompatible with outer_box rank (%d vs %d).' %
                       (box_size.size, outer_box.rank))

    # normalize overlap
    if box_overlap is not None:
      box_overlap = list(box_overlap)
      while len(box_overlap) < outer_box.rank:
        box_overlap.append(0)
      box_overlap = np.array(box_overlap)
    else:
      # box_overlap = np.zeros(outer_box.rank, dtype=int)
      box_overlap = np.zeros(outer_box.rank, dtype=box_size.dtype)

    if outer_box.rank != box_overlap.size:
      raise ValueError('box_overlap incompatible with rank (%d vs %d).' %
                       (box_overlap.size, outer_box.rank))

    box_stride = box_size - box_overlap
    if np.any(box_stride <= 0):
      raise ValueError(
          'box_size must be greater than box_overlap: %r versus %r' %
          (box_size, box_overlap))

    # The output_shape is the number of output boxes generated.
    output_shape = -(-(outer_box.size - box_overlap) // box_stride)

    self._outer_box = outer_box
    self._output = bounding_box.BoundingBox(
        start=([0] * output_shape.size), size=output_shape)
    self._squeeze = squeeze
    self._box_size = box_size
    self._box_stride = box_stride
    self._box_overlap = box_overlap
    self._back_shift_small_boxes = back_shift_small_boxes

  def __str__(self) -> str:
    items = [f'num_boxes={self.num_boxes}'
            ] + ['%s=%s' % item for item in vars(self).items()]
    return '%s(%s)' % (type(self).__name__, ', '.join(items))

  def __eq__(self: S, other: S) -> bool:
    for k, v in self.__dict__.items():
      if k not in other.__dict__:
        return False

      if isinstance(v, np.ndarray):
        if not np.all(v == other.__dict__[k]):
          return False
      else:
        if v != other.__dict__[k]:
          return False

    return True

  @property
  def outer_box(self) -> bounding_box.BoundingBoxBase:
    return self._outer_box

  @property
  def output(self) -> bounding_box.BoundingBox:
    return self._output

  @property
  def box_overlap(self) -> array.ImmutableArray:
    return array.ImmutableArray(self._box_overlap)

  @property
  def box_size(self) -> array.ImmutableArray:
    return array.ImmutableArray(self._box_size)

  @property
  def box_stride(self) -> array.ImmutableArray:
    return array.ImmutableArray(self._box_stride)

  @property
  def num_boxes(self) -> int:
    return np.prod(self._output.size)

  @property
  def squeeze(self) -> List[int]:
    return self._squeeze

  @property
  # def start(self) -> array.ImmutableArray:
  def start(self) -> np.ndarray:
    return self._generate(0)[1].start

  @property
  def back_shift_small_boxes(self) -> bool:
    return self._back_shift_small_boxes

  @property
  def boxes(self) -> Iterable[bounding_box.BoundingBoxBase]:
    for i in range(self.num_boxes):
      yield self.generate(i)[1]

  @property
  # def boxes_per_dim(self) -> array.ImmutableArray:
  def boxes_per_dim(self) -> np.ndarray:
    return self._output.size

  def _generate(self, index: BoxIndex) -> IndexedBoundingBox:
    coords = np.unravel_index(index, self._output.size, order='F')
    start = np.maximum(self._outer_box.start,
                       self._outer_box.start + coords * self._box_stride)
    end = np.minimum(start + self._box_size, self._outer_box.end)
    if self._back_shift_small_boxes:
      start = np.maximum(self._outer_box.start, end - self._box_size)
    return coords, self.outer_box.__class__(start=start, end=end)

  # TODO(timblakely): replace usage cases where callers subsequently call
  # np.unravel
  def generate(self, index: BoxIndex) -> IndexedBoundingBox:
    """Generate the mapping for a provided index.

    Args:
      index: index of the box to generate.

    Returns:
      A tuple of the coordinates and the corresponding BoundingBox.
    """
    if index >= self.num_boxes:
      raise ValueError('index must be in range')
    return self._generate(index)

  def index_to_cropped_box(self,
                           index: BoxIndex) -> bounding_box.BoundingBoxBase:
    """Translates a linear index to the corresponding cropped output box.

    In overlapped subvolume processing with non-overlapping outputs, the output
    box defines the subvolume that should be generated as the result of
    processing the given input box.

    Args:
      index: linear index in [0, num_sub_boxes)

    Returns:
      The corresponding output BoundingBox.

    Raises:
      ValueError when an invalid index is passed
    """
    box = self.generate(index)[1]
    is_start, is_end = self.tag_border_locations(index)
    front = self._box_overlap // 2
    back = self._box_overlap - front
    front *= 1 - is_start
    back *= 1 - is_end
    return box.adjusted_by(start=front, end=-back)

  def box_coordinate_to_index(self, point: Sequence[int]) -> BoxIndex:
    point = np.array(point)
    if np.any(point < 0) or np.any(point >= self._output.size):
      raise ValueError(
          'point must be between the origin and the output_shape: %r versus %r'
          % (point, self._output.size))
    return np.ravel_multi_index(point, dims=self._output.size, order='F')

  def offset_to_index(self, index: BoxIndex,
                      offset: Sequence[int]) -> Optional[BoxIndex]:
    """Calculate the index of another box at offset relative to current index.

    This is usually used to calculate the boxes that neighbor the current box.

    Args:
      index: The current flat index from which to calculate the offset index.
      offset: The offset from current index at which to calculate the new index.
        Rank must be the same as the generator.

    Returns:
      The flat index at offset from current index, or None if the given offset
      goes beyond the range of sub-boxes.

    Raises:
      ValueError: If inputs are incorrect.
    """
    if len(offset) != self._outer_box.rank:
      raise ValueError('Offset must have same rank')
    index_box_coord = np.array(offset) + self.generate(index)[0]
    if np.any(index_box_coord < 0) or np.any(
        index_box_coord >= self.boxes_per_dim):
      return None
    return self.box_coordinate_to_index(index_box_coord)

  def spatial_point_to_box_coordinates(
      self, point: Sequence[float]) -> List[BoxIndexCoordinates]:
    """Returns all indexed subvolume corners that intersect with a given point.

    Given an N-D point, returns indexed subvolume grid corners for bounding
    boxes that contain the point.

    Args:
      point: N-D point to query.

    Returns:
      Indexed coordinates for overlapping bounding boxes.
    """

    relative_point = point - self.start
    if np.any(relative_point < 0) or np.any(
        relative_point >= self._outer_box.end):
      return []

    begin_strides = -np.eye(self._outer_box.rank)
    end_strides = np.eye(self._outer_box.rank)

    subvolume_size = self._box_size
    overlap = self._box_overlap
    stride = self._box_stride
    remainder = relative_point % subvolume_size
    lower_grid_coord = (relative_point - remainder) // stride
    all_box_grid_coords = [lower_grid_coord]

    def _make_offset_combos(offsets):
      combos = []
      for i in range(len(offsets)):
        combos = itertools.chain(combos, itertools.combinations(offsets, i + 1))
      new_offsets = functools.reduce(
          lambda a, combo: a + [np.sum(combo, axis=0)], combos, [])
      return [lower_grid_coord + x for x in new_offsets]

    begin_overlap_region = relative_point - lower_grid_coord * stride < overlap
    all_box_grid_coords += _make_offset_combos(
        begin_strides[begin_overlap_region])

    end_overlap_region = relative_point - lower_grid_coord * stride >= stride
    all_box_grid_coords += _make_offset_combos(end_strides[end_overlap_region])

    coords = [x.astype(int) for x in all_box_grid_coords if np.all(x >= 0)]
    return [
        tuple(x)
        for x in coords
        if np.all(x >= 0) and np.all(x < self._output.size)
    ]

  def batch(
      self,
      batch_size: int,
      begin_index: int = 0,
      end_index: Optional[int] = None
  ) -> Iterable[Iterable[bounding_box.BoundingBoxBase]]:
    """Generates iterators for batches of sub-boxes.

    Args:
      batch_size: how many sub-boxes per iterable.
      begin_index: the inclusive beginning numerical index.
      end_index: the exclusive ending numerical index.

    Yields:
      An iterable of sub-boxes for each batch.
    """
    if end_index is None:
      end_index = self.num_boxes
    for i_begin in range(begin_index, end_index, batch_size):
      i_end = min(i_begin + batch_size, end_index)
      yield (self.generate(i)[1] for i in range(i_begin, i_end))

  def tag_border_locations(self,
                           index: BoxIndex) -> Tuple[np.ndarray, np.ndarray]:
    """Checks whether a box touches the border of the BoundingBox.

    Args:
      index: flat index identifying the box to check

    Returns:
      2-tuple of bool N-d ndarrays (dim order: x, y, z,...).
      True if the box touches the border at the start/end (respectively for the
      1st and 2nd element of the tuple) of the bbox along the given dimension.
    """
    coords_xyz = np.array(self._generate(index)[0])
    is_start = coords_xyz == 0
    is_end = coords_xyz == self._output.size - 1
    return is_start, is_end

  def overlapping_subboxes(
      self,
      box: bounding_box.BoundingBox) -> Iterable[bounding_box.BoundingBox]:
    """Yields subbboxes that overlap the given box.

    Args:
      box: Box to query.

    Yields:
      Bounding boxes that overlap.
    """

    def _start_to_box(start):
      full_box = self.outer_box.__class__(start=start, size=self.box_size)
      if self._back_shift_small_boxes:
        shift = np.maximum(full_box.end - self._outer_box.end, 0)
        if shift.any():
          return self.outer_box.__class__(
              start=full_box.start - shift, size=self.box_size)
        return full_box
      else:
        return full_box.intersection(self._outer_box)

    start_xyz = self.start + np.maximum(
        0, (box.start - self.start -
            self._box_size)) // self._box_stride * self._box_stride
    end_xyz = box.end + self._box_size

    for z in range(start_xyz[2], end_xyz[2], self._box_stride[2]):
      for y in range(start_xyz[1], end_xyz[1], self._box_stride[1]):
        for x in range(start_xyz[0], end_xyz[0], self._box_stride[0]):
          sub_box = _start_to_box((x, y, z))
          if sub_box is None or box.intersection(sub_box) is None:
            continue
          yield sub_box


GeneratorIndex = TypeVar('GeneratorIndex', bound=int)
MultiBoxIndex = TypeVar('MultiBoxIndex', bound=int)


class MultiBoxGenerator(BoxGeneratorBase):
  """Wrapper around multiple BoxGenerators.

  Supports generating sub-boxes from multiple outer_boxes with a single linear
  index.
  """

  generators: Sequence[BoxGenerator]

  def __init__(self, outer_boxes: Sequence[bounding_box.BoundingBoxBase], *args,
               **kwargs):
    """Wrapper around multiple BoxGenerators.

    Args:
      outer_boxes: Multiple outer boxes.
      *args: Broadcast to respective generators for input outer_boxes.
      **kwargs: Broadcast to respective generators for input outer_boxes.
    """
    self.generators = [
        BoxGenerator(outer_box, *args, **kwargs) for outer_box in outer_boxes
    ]
    boxes_per_generators = [c.num_boxes for c in self.generators]
    self.prefix_sums = np.cumsum([0] + boxes_per_generators)

  def index_to_generator_index(
      self, multi_box_index: MultiBoxIndex) -> Tuple[GeneratorIndex, BoxIndex]:
    """Determines which outer box the index falls into.

    Args:
      multi_box_index: Index to retrieve, relative to the MultiBoxGenerator.

    Returns:
      The index of the generator that handles the index.

    Raises:
      ValueError: If value is beyond the linear range.
    """
    if multi_box_index < 0 or multi_box_index >= self.num_boxes:
      raise ValueError('Invalid multi_box_index: %d' % multi_box_index)
    generator_index = bisect.bisect_right(self.prefix_sums[1:], multi_box_index)
    index = multi_box_index - self.prefix_sums[generator_index]
    return generator_index, index

  def generate(self, multi_box_index: MultiBoxIndex) -> IndexedBoundingBox:
    """Translates linear index to sub box.

    Outer boxes addressed in order.

    Args:
      multi_box_index: Index to retrieve, relative to the MultiBoxGenerator.

    Returns:
      Tuple containing a per-dimension index tuple (e.g. XYZ) and the box at the
      given index.
    """
    generator_index, index = self.index_to_generator_index(multi_box_index)
    return self.generators[generator_index].generate(index)

  def index_to_cropped_box(
      self, multi_box_index: MultiBoxIndex) -> bounding_box.BoundingBox:
    """Translates linear index to croppped output box.

    Args:
      multi_box_index: Index to retrieve, relative to the MultiBoxGenerator.

    Returns:
      The box at the given index.
    """
    generator_index, index = self.index_to_generator_index(multi_box_index)
    return self.generators[generator_index].index_to_cropped_box(index)

  @property
  def num_boxes(self) -> int:
    """Total number of sub boxes for all outer boxes."""
    return self.prefix_sums[-1]

  def tag_border_locations(
      self, multi_box_index: MultiBoxIndex) -> Tuple[np.ndarray, np.ndarray]:
    """Checks whether a box touches the border of the containing BoundingBox.

    Args:
      multi_box_index: Index to retrieve, relative to the MultiBoxGenerator.

    Returns:
      2-tuple of bool N-d ndarrays (dim order: x, y, z,...).
      True if the box touches the border at the start/end (respectively for the
      1st and 2nd element of the tuple) of the bbox along the given dimension.
    """
    generator_index, index = self.index_to_generator_index(multi_box_index)
    return self.generators[generator_index].tag_border_locations(index)

  @property
  def box_size(self) -> array.ImmutableArray:
    return self.generators[0].box_size

  @property
  def box_overlap(self) -> array.ImmutableArray:
    return self.generators[0].box_overlap
