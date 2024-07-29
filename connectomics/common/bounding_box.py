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
"""Defines the BoundingBox dataclass to describe bounding boxes."""

import dataclasses
import typing
from typing import Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

from connectomics.common import array
from connectomics.common import utils
import dataclasses_json
import numpy as np

T = TypeVar('T', int, float)
S = TypeVar('S', bound='BoundingBoxBase')
FloatSequence = Union[float, Sequence[float]]
BoolSequence = Union[bool, Sequence[bool]]


def limit_encoder(v):
  return [utils.from_np_type(x) for x in v]


# TODO(timblakely): Move is_border_* out of BoundingBox and into BoxGenerator.
@dataclasses_json.dataclass_json
@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class BoundingBoxBase(Generic[T]):
  """BoundingBox encapsulates start/end coordinate pairs of the same length."""
  _start: Tuple[T, ...] = dataclasses.field(
      metadata=dataclasses_json.config(
          field_name='start', encoder=limit_encoder))
  _size: Tuple[T, ...] = dataclasses.field(
      metadata=dataclasses_json.config(
          field_name='size', encoder=limit_encoder))
  # TODO(timblakely): Move these internal.
  _is_border_start: Optional[Tuple[bool, ...]] = dataclasses.field(
      default=None, metadata=dataclasses_json.config(
          field_name='is_border_start'))
  _is_border_end: Optional[Tuple[bool, ...]] = dataclasses.field(
      default=None, metadata=dataclasses_json.config(
          field_name='is_border_end'))

  def __init__(
      self,
      start: Optional[FloatSequence] = None,
      size: Optional[FloatSequence] = None,

      # TODO(timblakely): Move these parameters into the limited upstream
      # usage, as it's not exactly specific to BoundingBox
      is_border_start: Optional[BoolSequence] = None,
      is_border_end: Optional[BoolSequence] = None,

      end: Optional[FloatSequence] = None,
      **kwargs):
    """Initialize a BoundingBox from explicit bounds.

    Exactly two of start, size, and end must be specified.

    Args:
      start: An N-D element sequence specifying the (inclusive) start bound.
      size: An N-D element sequence specifying the size.
      is_border_start: (xyz) Optional N-D element bool sequence specifying
        whether this box is adjacent to the beginning of the containing volume
        along the respective dimension
      is_border_end: (xyz) Optional N-D element bool sequence specifying whether
        this box is adjacent to the end of the containing volume along the
        respective dimension
      end: An N-D element sequence specifying the (exclusive) end bound.
      **kwargs: Fields used during dataclass initialization.

    Raises:
      ValueError: on bad inputs.
    """
    if start is None and '_start' in kwargs:
      start = kwargs['_start']
    if size is None and '_size' in kwargs:
      size = kwargs['_size']
    if (end is not None) + (start is not None) + (size is not None) != 2:
      raise ValueError('Exactly two of start, end, and size must be specified. '
                       f'Got {start=}, {size=}, {end=}.')
    if not array.is_arraylike(start) and not array.is_arraylike(end):
      raise ValueError('At least one of start, end must be a sequence. '
                       f'Got {start=} and {end=}.')

    if array.is_arraylike(start):
      start = np.asarray(start)
    if array.is_arraylike(end):
      end = np.asarray(end)

    if size is not None:
      if array.is_arraylike(size):
        size = np.asarray(size)
      else:
        param = start if start is not None else end
        size = np.ones_like(param) * size
      if start is not None:
        end = start + size
      else:
        start = end - size
    else:
      size = end - start

    object.__setattr__(self, '_start', self._tupleize(start))
    object.__setattr__(self, '_size', self._tupleize(size))

    if len(self.start) != len(self.end) or len(self.end) != len(self.start):
      raise ValueError(
          'BoundingBox.start and BoundingBox.end must be the same length. '
          '%r vs. %r' % (self.start, self.end))

    if is_border_start is not None:
      if len(is_border_start) != self.rank:
        raise ValueError(
            f'is_border_start needs to have exactly {self.rank} items')
      object.__setattr__(self, '_is_border_start',
                         tuple(np.asarray(is_border_start).tolist()))
    else:
      object.__setattr__(self, '_is_border_start', tuple([False] * self.rank))

    if is_border_end is not None:
      if len(is_border_end) != self.rank:
        raise ValueError(
            f'is_border_end needs to have exactly {self.rank} items')
      object.__setattr__(self, '_is_border_end',
                         tuple(np.asarray(is_border_end).tolist()))
    else:
      object.__setattr__(self, '_is_border_end', tuple([False] * self.rank))

  def __eq__(self: S, other: S) -> bool:
    for k, v in self.__dict__.items():
      if k not in other.__dict__:
        return False

      # TODO(timblakely): Do we want to count is_border_* in equality?
      if k.startswith('_is_border_'):
        continue

      if isinstance(v, np.ndarray):
        if not np.all(v == other.__dict__[k]):
          return False
      else:
        if v != other.__dict__[k]:
          return False

    return True

  def __hash__(self: S) -> int:
    return hash((self._start, self._size))

  def _tupleize(self: S, seq: Sequence[float]) -> Tuple[T, ...]:
    """Convert sequence to a correctly-typed tuple.

    Base classes should override this if a specific type is desired.

    Args:
      seq: Sequence of numbers to convert.

    Returns:
      Tuple containing correctly typed values.
    """
    return tuple(seq)

  def _as_ndarray(self: S, seq: Sequence[float]) -> np.ndarray:
    """Convert sequence to a correctly-typed ndarray.

    Base classes should override this if a specific type is desired.

    Args:
      seq: Sequence of numbers to convert.

    Returns:
      Numpy array containing correctly typed values.
    """
    return np.asarray(seq)

  @property
  def rank(self: S) -> int:
    return len(self.start)

  @property
  def start(self: S) -> np.ndarray:
    start = np.asarray(self._start)
    start.setflags(write=False)
    return start

  @property
  def end(self: S) -> np.ndarray:
    end = self.start + self.size
    end.setflags(write=False)
    return end

  @property
  def size(self: S) -> np.ndarray:
    size = np.asarray(self._size)
    size.setflags(write=False)
    return size

  @property
  def is_border_start(self: S) -> np.ndarray:
    is_start = np.asarray(self._is_border_start)
    is_start.setflags(write=False)
    return is_start

  @property
  def is_border_end(self: S) -> np.ndarray:
    is_end = np.asarray(self._is_border_end)
    is_end.setflags(write=False)
    return is_end

  def scale(self: S, scale_factor: FloatSequence) -> S:
    """Returns a new BoundingBox, scaled relative to this one.

    When T is int, takes the floor of the scaling with respect to start
    and the ceiling with respect to size.

    Args:
      scale_factor: The scale factor to use, which may be either numeric or a
        sequence

    Returns:
      A new BoundingBoundingBox down-scaled from the original. The start corner
      is computed by floor-division, but the size is computed by
      ceiling-division.
    """
    if array.is_arraylike(scale_factor):
      if len(scale_factor) != self.rank:
        raise ValueError(f'scale_factor {scale_factor} length does not match '
                         f'rank {self.rank}.')
      scale_factor = np.array(scale_factor, dtype=float)
    start = np.array(self._start, dtype=float) * scale_factor
    size = np.array(self._size, dtype=float) * scale_factor
    if self.start.dtype == int:
      start = np.floor(start).astype(int)
      size = np.ceil(size).astype(int)
    return self.__class__(start=start, size=size)

  def adjusted_by(self: S,
                  start: Optional[Union[T, Sequence[T]]] = None,
                  end: Optional[Union[T, Sequence[T]]] = None) -> S:
    """Adds an offset to the start and/or end bounds of the bounding box.

    Args:
      start: offset added to the start bound
      end: offset added to the end bound

    Returns:
      A new bounding box with adjusted bounds.
    """
    if start is None and end is None:
      return self

    if start is None:
      start = self.start
    else:
      if array.is_arraylike(start) and len(start) != self.rank:
        raise ValueError(f'start {start} length does not match rank '
                         f'{self.rank}.')
      start = self.start + start

    if end is None:
      end = self.end
    else:
      if array.is_arraylike(end) and len(end) != self.rank:
        raise ValueError(f'end {end} length does not match rank {self.rank}.')
      end = self.end + end
    return self.__class__(start=start, end=end)

  def translate(self: S, offset: FloatSequence) -> S:
    """Translates the bounding box by a specified amount.

    Args:
      offset: offset to add to the start

    Returns:
      A new bounding box shifted by the specified vector.
    """
    if array.is_arraylike(offset) and len(offset) != self.rank:
      raise ValueError(f'offset {offset} length does not match rank '
                       f'{self.rank}.')
    start = self.start + offset
    return self.__class__(start=start, size=self.size)

  def intersection(self: S, other: S) -> Optional[S]:
    """Get the intersection with another bounding box, or None.

    Args:
      other: box to intersect.

    Returns:
      The intersection bounding box.

    Raises:
      ValueError: if invalid arguments are specified.
    """
    if self.rank != other.rank:
      raise ValueError(f'self.rank {self.rank} does not match other.rank '
                       f'{other.rank}.')
    start = np.maximum(self.start, other.start)
    end = np.minimum(self.end, other.end)
    if np.any(end <= start):
      return None
    return self.__class__(start=start, end=end)

  def hull(self: S, other: S) -> S:
    """Get the hull (minimum box enclosing) with another bounding box.

    Args:
      other: box to hull.

    Returns:
      The minimum bounding box that contains both boxes.

    Raises:
      ValueError: if invalid arguments are specified.
    """
    if self.rank != other.rank:
      raise ValueError(f'self.rank {self.rank} does not match other.rank '
                       f'{other.rank}.')
    start = np.minimum(self.start, other.start)
    end = np.maximum(self.end, other.end)
    return self.__class__(start=start, end=end)

  def relative(self: S,
               start: Optional[Sequence[float]] = None,
               end: Optional[Sequence[float]] = None,
               size: Optional[Sequence[float]] = None) -> S:
    """Returns a new BoundingBox with the specified bounds relative to self.

    Args:
      start: Specifies the new start bound, relative to self.start.  If not
        specified, the current start bound is kept, unless end and size are both
        specified, in which case it is inferred.
      end: Specifies the new end bound, relative to self.start.  If not
        specified, the current end bound is kept, unless start and size are both
        specified, in which case it is inferred.
      size: In conjunction with start or end (but not both), specifies the new
        size.

    Returns:
      A new bounding box with adjusted bounds, or self if no arguments are
    specified.

    Raises:
      ValueError: if invalid arguments are specified.
    """
    if start is None:
      if end is None:
        if size is not None:
          raise ValueError('size must be specified with either end or start')
        return self
      else:
        if size is None:
          return self.__class__(start=self.start, size=end)
        else:
          start = self.start + end - size
          return self.__class__(start=start, size=size)
    else:
      # start specified.
      if end is None:
        if size is None:
          size = self.size - start
        return self.__class__(start=np.add(self.start, start), size=size)
      else:
        # end specified.
        if size is not None:
          raise ValueError(
              'size must not be specified if both start and end are given')
        return self.__class__(
            start=np.add(self.start, start), size=np.subtract(end, start))

  def to_slice_tuple(self,
                     start_dim: Optional[int] = None,
                     end_dim: Optional[int] = None,
                     order: str = 'c') -> Tuple[slice, ...]:
    """Returns slice in C or Fortran-order (XYZ).

    Args:
      start_dim: Optional beginning dimension to begin slice.
      end_dim: Optional end dimension to begin slice.
      order: C or Fortran order.

    Returns:
      Tuple corresponding to a slice expression akin to np.index_exp.
    """
    start = self.start[start_dim:end_dim]
    end = self.end[start_dim:end_dim]
    extents = tuple(zip(start, end))
    if order[0].lower() == 'c':
      extents = extents[::-1]
    return tuple(slice(s, e, None) for s, e in extents)

  def to_slice3d(self) -> Tuple[slice, slice, slice]:
    """Convenience function for 3d use.

    Returns reverse ZYX order. If the bounding box has fewer than 3 dimensions,
    the remaining dimensions will slice to `:` (`slice(None, None, None)`) for
    remaining dimensions.

    Returns:
      Slice tuple in ZYX order.
    """
    slices = self.to_slice_tuple(0, 3)
    if len(slices) < 3:
      slices = (slice(None, None, None),) * (3 - len(slices)) + slices
    return slices

  def to_slice4d(self) -> Tuple[slice, slice, slice, slice]:
    """Convenience function for 4d use.

    Returns reverse CZYX order. If the bounding box has fewer than 4 dimensions,
    the remaining dimensions will slice to `:` (`slice(None, None, None)`) for
    remaining dimensions.

    Returns:
      Slice tuple in CZYX order.
    """
    slices = self.to_slice_tuple(0, 4)
    if len(slices) < 4:
      slices = (slice(None, None, None),) * (4 - len(slices)) + slices
    return slices

  def encompass(self: S, *boxes: S) -> S:
    """Return a new bounding box that contains this and all other boxes.

    Args:
      *boxes: One or more bounding boxes.

    Returns:
      The minimum bounding box that contains all boxes.

    Raises:
      ValueError: If no boxes are provided.
    """
    if not boxes:
      raise ValueError('At least one bounding box must be specified')
    start = self.start
    end = self.end
    for box in boxes:
      start = np.minimum(start, box.start)
      end = np.maximum(end, box.end)
    return self.__class__(start=start, end=end)

  def __repr__(self):
    return (f'{self.__class__.__name__}(start={self._start}, '
            f'size={self._size}, '
            f'is_border_start={self._is_border_start}, '
            f'is_border_end={self._is_border_end})')


class BoundingBox(BoundingBoxBase[int]):

  def _tupleize(self, seq: Sequence[float]) -> Tuple[int, ...]:
    return tuple(np.array(seq, dtype=int).tolist())

  def _as_ndarray(self, seq: Sequence[float]) -> np.ndarray:
    return np.array(seq, dtype=int)


class FloatBoundingBox(BoundingBoxBase[float]):

  def _tupleize(self, seq: Sequence[float]) -> Tuple[float, ...]:
    return tuple(np.array(seq, dtype=float).tolist())

  def _as_ndarray(self, seq: Sequence[float]) -> np.ndarray:
    return np.array(seq, dtype=float)


def intersections(
    first_box_or_boxes: Union[S, Iterable[S]],
    other_box_or_boxes: Optional[Union[S, Iterable[S]]] = None) -> List[S]:
  """Get intersections between two sequences of boxes.

  Args:
    first_box_or_boxes: A BoundingBox or sequence of BoundingBoxes.
    other_box_or_boxes: An optional BoundingBox or sequence of BoundingBoxes.

  Returns:
    list of intersections between the two sequences. Each element of
    first_box_or_boxes is intersected with each element of other_box_or_boxes,
    and any non-None are added to the list. If other_box_or_boxes is None,
    simply returns list(first_box_or_boxes).
  """
  if isinstance(first_box_or_boxes, BoundingBoxBase):
    first_box_or_boxes = [first_box_or_boxes]

  if other_box_or_boxes is None:
    return list(first_box_or_boxes)

  if isinstance(other_box_or_boxes, BoundingBoxBase):
    other_box_or_boxes = [other_box_or_boxes]

  # Make PyType happy
  first_box_or_boxes = typing.cast(List[S], first_box_or_boxes)
  other_box_or_boxes = typing.cast(List[S], other_box_or_boxes)

  ret = []
  for box0 in first_box_or_boxes:
    for box1 in other_box_or_boxes:
      new_box = box0.intersection(box1)
      if new_box is not None:
        ret.append(new_box)
  return ret


def containing(*boxes: S) -> S:
  """Get the minimum bounding box containing all specified boxes.

  Args:
    *boxes: BoundingBoxes to contain.

  Returns:
    The minimum bounding box that contains all boxes.

  Raises:
    ValueError: if input is empty.
  """

  if not boxes:
    raise ValueError('At least one bounding box must be specified')
  box = boxes[0]
  if len(boxes) == 1:
    return box
  return box.encompass(*boxes[1:])


def from_json(as_json: str) -> BoundingBoxBase:
  """Deserialize and guess bounding box type."""
  bbox = BoundingBoxBase.from_json(as_json)
  if any([utils.is_floatlike(v) for v in bbox.start] +
         [utils.is_floatlike(v) for v in bbox.size]):
    return FloatBoundingBox(bbox.start, bbox.size)
  return BoundingBox(bbox.start, bbox.size)


def from_slices(slices: Sequence[slice],
                order='c',
                inclusive=True) -> BoundingBox:
  """Creates a new BoundingBox from slice extents.

  Args:
    slices: Extents for the limits of the bounding box.
    order: C or Fortran order of slices (CZYX... vs XYZC...).
    inclusive: If the `slice`.start is considered inclusive or not. If not, box
      start/end extents will be slice.start:slice.stop-1.

  Returns:
    New BoundingBox with the given extents.

  Raises:
    ValueError: If any of the extents is not numeric.
  """
  extents = [(s.start, s.stop) for s in slices]
  if np.any(None in np.array(extents)):
    raise ValueError('All slices must be finite.')
  if not inclusive:
    extents = tuple((a, z - 1) for a, z in extents)
  if order[0].lower() == 'c':
    extents = extents[::-1]
  start, stop = zip(*extents)
  return BoundingBox(start, end=stop)
