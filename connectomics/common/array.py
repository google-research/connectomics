"""Array types extending numpy arrays.

The classes and types contained in here are primarily a way for PyType to reason
about. NumPy's typing system doesn't support immutability nor constrained
dimension sizes, so we introduce wrappers for common types (3D, 3D+Channels) to
allow for build-time error checking. This is generally preferable to seeing
errors at run-time during long-running pipelines.
"""

from collections import abc
import numbers
from typing import Any, Tuple, TypeVar, Type, Union

from connectomics.common import array_mixins
import numpy as np
import numpy.typing as npt

T = TypeVar('T', int, float)

# TODO(timblakely): Support Ellipses
ArbitrarySlice = Union[int, slice]
PointLookup = Union[np.ndarray, list[int]]
PointLookup4d = Tuple[ArbitrarySlice, PointLookup, PointLookup, PointLookup]
ArbitrarySlice4d = Tuple[ArbitrarySlice, ArbitrarySlice, ArbitrarySlice,
                         ArbitrarySlice]
CanonicalSlice4d = Tuple[slice, slice, slice, slice]

ArrayLike = Union[npt.ArrayLike, 'ImmutableArray', 'MutableArray']
Tuple3f = Tuple[float, float, float]
Tuple3i = Tuple[int, int, int]
Tuple4i = Tuple[int, int, int, int]
ArrayLike3d = Union[npt.ArrayLike, 'ImmutableArray', 'MutableArray', Tuple3f,
                    Tuple3i]


def normalize_index(ind: Union[ArbitrarySlice4d, PointLookup4d],
                    limits: Tuple4i) -> Union[CanonicalSlice4d, PointLookup4d]:
  """Converts a volume indexing expression into the canonical form.

  If the index expression is a point lookup, it is returned unchanged.

  Args:
    ind: indexing expression as passed to []
    limits: maximum limits in the event of open-ended index expression.

  Returns:
    4-sequence of slice objects for indexing subvolumes

  Raises:
    ValueError: when an unsupported indexing expression is passed
  """
  if not isinstance(ind, abc.Sized) or len(ind) != 4:
    raise ValueError('Only 4D slices are currently supported. '
                     f'Ellipses are unsupported. {ind} unsupported.')

  if all((isinstance(x, np.ndarray) or isinstance(x, list)) for x in ind[1:]):
    # Cast away the type here to make pytype happy. We've checked in the
    # conditional above that all the values for ind[1:] are either an ndarray or
    # a list.
    ind: Any = ind
    return ind

  ind = [process_slice_ind(ind_i, lim_i) for ind_i, lim_i in zip(ind, limits)]
  return tuple(ind)


def process_slice_ind(slice_ind: ArbitrarySlice, limit: int):
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
