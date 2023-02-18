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
"""Array types extending numpy arrays.

The classes and types contained in here are primarily a way for PyType to reason
about. NumPy's typing system doesn't support immutability nor constrained
dimension sizes, so we introduce wrappers for common types (3D, 3D+Channels) to
allow for build-time error checking. This is generally preferable to seeing
errors at run-time during long-running pipelines.
"""

from __future__ import annotations

from collections import abc
import numbers
from typing import Any, List, Tuple, TypeVar, Type, Union

from connectomics.common import array_mixins
import numpy as np
import numpy.typing as npt

T = TypeVar('T', int, float)

# TODO(timblakely): Support Ellipses
CoordinateLookup = Union[np.ndarray, List[int], Tuple[int, ...]]
PointLookups = Tuple[Union[int, slice], CoordinateLookup, CoordinateLookup,
                     CoordinateLookup]
ArbitrarySlice = Tuple[Union[int, slice], Union[int, slice], Union[int, slice],
                       Union[int, slice]]
CanonicalSlice = Tuple[slice, slice, slice, slice]
IndexExpOrPointLookups = Union[ArbitrarySlice, PointLookups]
CanonicalSliceOrPointLookups = Union[CanonicalSlice, PointLookups]

ArrayLike = Union[npt.ArrayLike, 'ImmutableArray', 'MutableArray']
Tuple3f = Tuple[float, float, float]
Tuple3i = Tuple[int, int, int]
Tuple4i = Tuple[int, int, int, int]
ArrayLike3d = Union[npt.ArrayLike, 'ImmutableArray', 'MutableArray', Tuple3f,
                    Tuple3i]


def is_point_lookup(ind: IndexExpOrPointLookups) -> bool:
  return len(ind) == 4 and all(
      (isinstance(x, np.ndarray) or isinstance(x, list) or isinstance(x, tuple))
      for x in ind[1:])


def normalize_index(ind: IndexExpOrPointLookups,
                    limits: Tuple4i) -> CanonicalSliceOrPointLookups:
  """Converts a volume indexing expression into the canonical form.

  If the index expression is a point lookup, it is returned unchanged.

  Args:
    ind: indexing expression as passed to []
    limits: maximum limits in the event of open-ended index expression.

  Returns:
    Sequence of slice objects for indexing subvolumes, up to 4d.

  Raises:
    ValueError: when an unsupported indexing expression is passed
  """
  if not isinstance(ind, abc.Sized) or len(ind) > 4:
    raise ValueError(f'Slices must be <= 4D. Ellipses are unsupported. '
                     f'{ind} unsupported.')

  if is_point_lookup(ind):
    # Cast away the type here to make pytype happy. We've checked in the
    # conditional above that all the values for ind[1:] are either an ndarray or
    # a list.
    ind: Any = ind
    return ind

  ind = [process_slice_ind(ind_i, lim_i) for ind_i, lim_i in zip(ind, limits)]
  return tuple(ind)


def process_slice_ind(slice_ind: ArbitrarySlice, limit: int) -> slice:
  """Converts a single slice index to canonical slice form.

  Does not have built-in support for striding. Note that if an underlying
  volume type _does_ support striding, this function should be overridden in
  the child class.

  Args:
    slice_ind: an element of the tuple passed into __getitem__ by the []
      operator.  Should be a slice or int.
    limit: upper limit of the slice range.

  Returns:
    slice in canonical form.

  Raises:
    ValueError: if slice step is given; we don't currently support strided
                slicing.
    ValueError: if ... is given.
  """
  if isinstance(slice_ind, numbers.Integral):
    return slice(slice_ind, slice_ind + 1)
  elif isinstance(slice_ind, type(Ellipsis)):
    raise ValueError("Doesn't currently support '...' slicing.")
  elif not isinstance(slice_ind, slice):
    raise ValueError('Currently supports only slice or int indices for '
                     'slicing.')
  if not (slice_ind.step is None or slice_ind.step == 1):
    raise ValueError("Doesn't currently support strided or reverse slicing.")
  return slice(*slice_ind.indices(limit))


# TODO(timblakely): Make these typed by using Generic[T]
class ImmutableArray(array_mixins.ImmutableArrayMixin, np.ndarray):
  """Strongly typed, immutable NumPy NDArray."""

  def __new__(cls: Type['ImmutableArray'],
              input_array: 'ArrayLike',
              *args,
              zero_copy=False,
              **kwargs) -> 'ImmutableArray':
    if zero_copy:
      obj = np.asanyarray(input_array, *args, **kwargs).view(cls)
    else:
      obj = np.array(input_array, *args, **kwargs).view(cls)
    obj.flags.writeable = False
    return obj

  def __init__(self, *args, zero_copy=False, **kwargs):
    # Needed for mixin construction.
    super().__init__()  # pylint: disable=no-value-for-parameter

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    out = kwargs.get('out', ())

    # Defer to the implementation of the ufunc on unwrapped values.
    inputs = tuple(_to_np_compatible(x) for x in inputs)
    if out:
      kwargs['out'] = tuple(_to_np_compatible(x) for x in out)
    result = getattr(ufunc, method)(*inputs, **kwargs)

    if method == 'at':
      # no return value
      return None

    return MutableArray(result)

  def copy(self, *args, **kwargs) -> 'MutableArray':
    return MutableArray(np.asarray(self).copy(*args, **kwargs))

  def __str__(self):
    return np.ndarray.__repr__(self)


class MutableArray(array_mixins.MutableArrayMixin, ImmutableArray):
  """Strongly typed mutable version of np.ndarray."""

  def __new__(cls: Type['MutableArray'],
              input_array: 'ArrayLike',
              *args,
              zero_copy=False,
              **kwargs) -> 'MutableArray':
    if zero_copy:
      obj = np.asanyarray(input_array, *args, **kwargs).view(cls)
    else:
      obj = np.array(input_array, *args, **kwargs).view(cls)
    obj.flags.writeable = True
    return obj


def is_arraylike(obj):
  # Technically sequences, but this is intended to check for numeric sequences.
  if isinstance(obj, str) or isinstance(obj, bytes):
    return False
  return isinstance(obj, abc.Sequence) or isinstance(
      obj, np.ndarray) or isinstance(obj, ImmutableArray) or isinstance(
          obj, MutableArray)


def _to_np_compatible(array_like) -> np.ndarray:
  if isinstance(array_like, ImmutableArray) or isinstance(
      array_like, MutableArray):
    return np.asarray(array_like)
  return array_like
