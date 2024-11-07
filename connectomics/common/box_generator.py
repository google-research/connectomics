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

from __future__ import annotations

import bisect
import dataclasses
import functools
import itertools
import json
from typing import Iterable, Sequence, TypeVar

from connectomics.common import bounding_box
import dataclasses_json
import numpy as np

S = TypeVar('S', bound='BoxGenerator')
BoxIndexCoordinates = tuple[int, ...]
IndexedBoundingBox = tuple[BoxIndexCoordinates, bounding_box.BoundingBoxBase]
BoxIndex = TypeVar('BoxIndex', bound=int)


class BoxGeneratorBase:
  """Base class for BoundingBox generators."""

  @property
  def num_boxes(self) -> int:
    raise NotImplementedError()

  def generate(self, index: int) -> IndexedBoundingBox:
    raise NotImplementedError()

  def index_to_cropped_box(self, intindex: int) -> bounding_box.BoundingBox:
    raise NotImplementedError()

  def __eq__(self: 'BoxGeneratorBase', other: 'BoxGeneratorBase') -> bool:
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

  def to_json(self) -> str:
    raise NotImplementedError()


@dataclasses_json.dataclass_json
@dataclasses.dataclass(eq=False, frozen=True)
class BoxGenerator(BoxGeneratorBase):
  """Generates overlapping sub-boxes spanning the input bounding box."""

  outer_box: bounding_box.BoundingBox
  box_size: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda v: v.tolist(), decoder=np.asarray
      )
  )
  box_overlap: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda v: v.tolist(),
          decoder=np.asarray,
      ),
      default_factory=lambda: np.zeros([]),
  )
  back_shift_small_boxes: bool = dataclasses.field(
      metadata=dataclasses_json.config(field_name='back_shift_small_boxes'),
      default=False,
  )

  squeeze: np.ndarray = dataclasses.field(
      init=False,
      metadata=dataclasses_json.config(
          encoder=lambda v: v.tolist(),
          decoder=np.asarray,
          exclude=dataclasses_json.Exclude.ALWAYS,
      ),
  )
  box_stride: np.ndarray = dataclasses.field(
      init=False,
      metadata=dataclasses_json.config(
          encoder=lambda v: v.tolist(),
          decoder=np.asarray,
          exclude=dataclasses_json.Exclude.ALWAYS,
      ),
  )
  output: bounding_box.BoundingBox = dataclasses.field(
      init=False,
      metadata=dataclasses_json.config(exclude=dataclasses_json.Exclude.ALWAYS),
  )

  def __post_init__(self):
    """Initialize a generator.

    Raises:
      ValueError: If box size is incompatible with outer box rank.
      ValueError: If box overlap is incompatible with outer box rank.
      ValueError: If box size is <= overlap.
    """
    outer_box = self.outer_box
    box_size = self.box_size
    box_overlap = np.asarray(self.box_overlap)
    back_shift_small_boxes = self.back_shift_small_boxes
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
      raise ValueError('box_size incompatible with outer_box rank (%s vs %s).' %
                       (box_size.size, outer_box.rank))

    # normalize overlap
    if box_overlap.ndim != 0:
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

    # Don't allow this to be >= outer_box.size as that will incorrectly result
    # in output_shape 0.
    back_clip = np.minimum(box_overlap, outer_box.size - 1)

    # The output_shape is the number of output boxes generated.
    output_shape = -(-(outer_box.size - back_clip) // box_stride)

    object.__setattr__(self, 'outer_box', outer_box)
    object.__setattr__(
        self,
        'output',
        bounding_box.BoundingBox(
            start=([0] * output_shape.size), size=output_shape
        ),
    )
    object.__setattr__(self, 'squeeze', np.asarray(squeeze))
    object.__setattr__(self, 'box_size', np.asarray(box_size))
    object.__setattr__(self, 'box_stride', np.asarray(box_stride))
    object.__setattr__(self, 'box_overlap', np.asarray(box_overlap))
    object.__setattr__(self, 'back_shift_small_boxes', back_shift_small_boxes)

  def __str__(self) -> str:
    items = [f'num_boxes={self.num_boxes}'
            ] + ['%s=%s' % item for item in vars(self).items()]
    return '%s(%s)' % (type(self).__name__, ', '.join(items))

  @property
  def num_boxes(self) -> int:
    return np.prod(self.output.size)

  @property
  def start(self) -> np.ndarray:
    return self._generate(0)[1].start

  @property
  def boxes(self) -> Iterable[bounding_box.BoundingBoxBase]:
    for i in range(self.num_boxes):
      yield self.generate(i)[1]

  @property
  def boxes_per_dim(self) -> np.ndarray:
    return self.output.size

  def _generate(self, index: BoxIndex) -> IndexedBoundingBox:
    coords = np.unravel_index(index, self.output.size, order='F')
    start = np.maximum(
        self.outer_box.start, self.outer_box.start + coords * self.box_stride
    )
    end = np.minimum(start + self.box_size, self.outer_box.end)
    if self.back_shift_small_boxes:
      start = np.maximum(self.outer_box.start, end - self.box_size)
    is_start, is_end = self.tag_border_locations(index)
    return coords, self.outer_box.__class__(
        start=start, end=end, is_border_start=is_start, is_border_end=is_end)

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
    front = self.box_overlap // 2
    back = self.box_overlap - front
    front *= 1 - is_start
    back *= 1 - is_end
    return box.adjusted_by(start=front, end=-back)

  def box_coordinate_to_index(self, point: Sequence[int]) -> BoxIndex:
    point = np.array(point)
    if np.any(point < 0) or np.any(point >= self.output.size):
      raise ValueError(
          'point must be between the origin and the output_shape: %r versus %r'
          % (point, self.output.size)
      )
    return np.ravel_multi_index(point, dims=self.output.size, order='F')

  def offset_to_index(
      self, index: BoxIndex, offset: Sequence[int]
  ) -> BoxIndex | None:
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
    if len(offset) != self.outer_box.rank:
      raise ValueError('Offset must have same rank')
    index_box_coord = np.array(offset) + self.generate(index)[0]
    if np.any(index_box_coord < 0) or np.any(
        index_box_coord >= self.boxes_per_dim):
      return None
    return self.box_coordinate_to_index(index_box_coord)

  def spatial_point_to_box_coordinates(
      self, point: Sequence[float]
  ) -> list[BoxIndexCoordinates]:
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
        relative_point >= self.outer_box.end
    ):
      return []

    begin_strides = -np.eye(self.outer_box.rank)
    end_strides = np.eye(self.outer_box.rank)

    subvolume_size = self.box_size
    overlap = self.box_overlap
    stride = self.box_stride
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
        if np.all(x >= 0) and np.all(x < self.output.size)
    ]

  def batch(
      self, batch_size: int, begin_index: int = 0, end_index: int | None = None
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

  def tag_border_locations(
      self, index: BoxIndex
  ) -> tuple[np.ndarray, np.ndarray]:
    """Checks whether a box touches the border of the BoundingBox.

    Args:
      index: flat index identifying the box to check

    Returns:
      2-tuple of bool N-d ndarrays (dim order: x, y, z,...).
      True if the box touches the border at the start/end (respectively for the
      1st and 2nd element of the tuple) of the bbox along the given dimension.
    """
    # Can't use _generate here, or it would recurse.
    coords_xyz = np.unravel_index(index, self.output.size, order='F')
    is_start = np.array(coords_xyz) == 0
    is_end = coords_xyz == self.output.size - 1
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
    start_xyz = (
        np.maximum(0, (box.start - self.start - self.box_size))
        // self.box_stride
        * self.box_stride
    )
    end_xyz = box.end + self.box_size

    for z in range(start_xyz[2], end_xyz[2], self.box_stride[2]):
      for y in range(start_xyz[1], end_xyz[1], self.box_stride[1]):
        for x in range(start_xyz[0], end_xyz[0], self.box_stride[0]):
          try:
            idx = self.box_coordinate_to_index((
                x // self.box_stride[0],
                y // self.box_stride[1],
                z // self.box_stride[2],
            ))
          except ValueError:
            continue

          _, sub_box = self._generate(idx)
          if box.intersection(sub_box) is not None:
            yield sub_box


GeneratorIndex = TypeVar('GeneratorIndex', bound=int)
MultiBoxIndex = TypeVar('MultiBoxIndex', bound=int)


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class MultiBoxGenerator(BoxGeneratorBase):
  """Wrapper around multiple BoxGenerators.

  Supports generating sub-boxes from multiple outer_boxes with a single linear
  index.
  """

  outer_boxes: list[bounding_box.BoundingBoxBase]
  box_size: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda v: v.tolist(), decoder=np.asarray
      )
  )
  box_overlap: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda v: v.tolist(),
          decoder=np.asarray,
      ),
      default_factory=lambda: np.zeros([]),
  )
  back_shift_small_boxes: bool = dataclasses.field(
      metadata=dataclasses_json.config(field_name='back_shift_small_boxes'),
      default=False,
  )

  generators: list[BoxGenerator] = dataclasses.field(
      init=False,
      metadata=dataclasses_json.config(exclude=dataclasses_json.Exclude.ALWAYS),
  )
  prefix_sums: np.ndarray = dataclasses.field(
      init=False,
      metadata=dataclasses_json.config(exclude=dataclasses_json.Exclude.ALWAYS),
  )

  def __post_init__(self):
    """Initializes multi-box generator attributes."""
    outer_boxes = self.outer_boxes

    object.__setattr__(
        self,
        'generators',
        [
            BoxGenerator(
                outer_box,
                self.box_size,
                self.box_overlap,
                self.back_shift_small_boxes,
            )
            for outer_box in outer_boxes
        ],
    )
    boxes_per_generators = [c.num_boxes for c in self.generators]
    object.__setattr__(
        self, 'prefix_sums', np.cumsum([0] + boxes_per_generators)
    )
    first_gen = self.generators[0]
    object.__setattr__(self, 'box_size', first_gen.box_size)
    object.__setattr__(self, 'box_overlap', first_gen.box_overlap)
    object.__setattr__(
        self, 'back_shift_small_boxes', first_gen.back_shift_small_boxes
    )
    object.__setattr__(self, 'outer_boxes', list(outer_boxes))

  def index_to_generator_index(
      self, multi_box_index: MultiBoxIndex
  ) -> tuple[GeneratorIndex, BoxIndex]:
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
    return self.generators[generator_index].index_to_cropped_box(index)  # pytype: disable=bad-return-type  # dynamic-method-lookup

  @property
  def num_boxes(self) -> int:
    """Total number of sub boxes for all outer boxes."""
    return self.prefix_sums[-1]

  def tag_border_locations(
      self, multi_box_index: MultiBoxIndex
  ) -> tuple[np.ndarray, np.ndarray]:
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


def from_json(as_json: str) -> BoxGeneratorBase:
  """Deserialize and guess generator type."""
  as_dict = json.loads(as_json)
  if 'outer_box' in as_dict:
    return BoxGenerator.from_dict(as_dict)
  elif 'outer_boxes' in as_dict:
    return MultiBoxGenerator.from_dict(as_dict)
  raise ValueError('Not a known box generator type')
