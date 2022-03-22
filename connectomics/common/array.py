"""Array types extending numpy arrays.

The classes and types contained in here are primarily a way for PyType to reason
about. NumPy's typing system doesn't support immutability nor constrained
dimension sizes, so we introduce wrappers for common types (3D, 3D+Channels) to
allow for build-time error checking. This is generally preferable to seeing
errors at run-time during long-running pipelines.
"""

import collections
from typing import Union, Tuple, TypeVar, Type

from connectomics.common import array_mixins
import numpy as np
import numpy.typing as npt

T = TypeVar('T', int, float)


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
  return isinstance(obj, collections.abc.Sequence) or isinstance(
      obj, np.ndarray) or isinstance(obj, ImmutableArray) or isinstance(
          obj, MutableArray)


def _to_np_compatible(array_like) -> np.ndarray:
  if isinstance(array_like, ImmutableArray) or isinstance(
      array_like, MutableArray):
    return np.asarray(array_like)
  return array_like


ArrayLike = Union[npt.ArrayLike, ImmutableArray, MutableArray]
Tuple3f = Tuple[float, float, float]
Tuple3i = Tuple[int, int, int]
ArrayLike3d = Union[npt.ArrayLike, ImmutableArray, MutableArray, Tuple3f,
                    Tuple3i]
