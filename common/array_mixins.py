"""Mutability mixins for numpy numeric and indexing."""

from typing import Any

import numpy as np

# Avoid circular dependency
array_mixins = Any


class InPlaceNumericOpsDisabled:
  """Type marker indicating that in-place numeric methods have been disabled."""
  pass


class IndexingDisabled:
  """Type marker indicating that indexing has been disabled."""
  pass


def _disables_array_ufunc(obj):
  """True when __array_ufunc__ is set to None."""
  try:
    return obj.__array_ufunc__ is None
  except AttributeError:
    return False


def _maybe_run_ufunc(ufunc, self, other, reflexive=False):
  if _disables_array_ufunc(other):
    return NotImplemented
  if reflexive:
    return ufunc(other, self)
  return ufunc(self, other)


class ImmutableArrayMixin:
  """Mixin that only enables immutable numeric methods."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    super().__init__(*args, **kwargs)

  # Comparisons

  def __lt__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.less(self, other)

  def __le__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.less_equal(self, other)

  def __eq__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.equal(self, other)

  def __ne__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.not_equal(self, other)

  def __gt__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.greater(self, other)

  def __ge__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.greater_equal(self, other)

  # Numeric

  def __add__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.add, self, other)

  def __radd__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.add, self, other, reflexive=True)

  __iadd__: InPlaceNumericOpsDisabled

  def __sub__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.subtract, self, other)

  def __rsub__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.subtract, self, other, reflexive=True)

  __isub__: InPlaceNumericOpsDisabled

  def __mul__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.multiply, self, other)

  def __rmul__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.multiply, self, other, reflexive=True)

  __imul__: InPlaceNumericOpsDisabled

  def __matmul__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.matmul, self, other)

  def __rmatmul__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.matmul, self, other, reflexive=True)

  __imatmul__: InPlaceNumericOpsDisabled

  def __truediv__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.true_divide, self, other)

  def __rtruediv__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(
        np.core.umath.true_divide, self, other, reflexive=True)

  __itruediv__: InPlaceNumericOpsDisabled

  def __floordiv__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.floor_divide, self, other)

  def __rfloordiv__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(
        np.core.umath.floor_divide, self, other, reflexive=True)

  __ifloordiv__: InPlaceNumericOpsDisabled

  def __mod__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.remainder, self, other)

  def __rmod__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(
        np.core.umath.remainder, self, other, reflexive=True)

  __imod__: InPlaceNumericOpsDisabled

  def __divmod__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.divmod, self, other)

  def __rdivmod__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.divmod, self, other, reflexive=True)

  def __pow__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.power, self, other)

  def __rpow__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.power, self, other, reflexive=True)

  __ipow__: InPlaceNumericOpsDisabled

  def __lshift__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.left_shift, self, other)

  def __rlshift__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(
        np.core.umath.left_shift, self, other, reflexive=True)

  __ilshift__: InPlaceNumericOpsDisabled

  def __rshift__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.right_shift, self, other)

  def __rrshift__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(
        np.core.umath.right_shift, self, other, reflexive=True)

  __irshift__: InPlaceNumericOpsDisabled

  def __and__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.bitwise_and, self, other)

  def __rand__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(
        np.core.umath.bitwise_and, self, other, reflexive=True)

  __iand__: InPlaceNumericOpsDisabled

  def __xor__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.bitwise_xor, self, other)

  def __rxor__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(
        np.core.umath.bitwise_xor, self, other, reflexive=True)

  __ixor__: InPlaceNumericOpsDisabled

  def __or__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(np.core.umath.bitwise_or, self, other)

  def __ror__(self, other) -> 'array_mixins.MutableArray':
    return _maybe_run_ufunc(
        np.core.umath.bitwise_or, self, other, reflexive=True)

  __ior__: InPlaceNumericOpsDisabled

  # Unary

  def __neg__(self) -> 'array_mixins.MutableArray':
    return np.core.umath.negative(self)

  def __pos__(self) -> 'array_mixins.MutableArray':
    return np.core.umath.positive(self)

  def __abs__(self) -> 'array_mixins.MutableArray':
    return np.core.umath.absolute(self)

  def __invert__(self) -> 'array_mixins.MutableArray':
    return np.core.umath.invert(self)

  # Disable mutable indexing
  __setitem__: IndexingDisabled

  def __array_wrap__(self, out_arr, context=None):
    pass


class MutableArrayMixin:
  """Mixin that enables in-place numeric and indexing methods."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    super().__init__(*args, **kwargs)

  def __iadd__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.add(self, other, out=(self,))

  def __isub__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.subtract(self, other, out=(self,))

  def __imul__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.multiply(self, other, out=(self,))

  def __imatmul__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.matmul(self, other, out=(self,))

  def __itruediv__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.true_divide(self, other, out=(self,))

  def __ifloordiv__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.floor_divide(self, other, out=(self,))

  def __imod__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.remainder(self, other, out=(self,))

  def __ipow__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.power(self, other, out=(self,))

  def __ilshift__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.left_shift(self, other, out=(self,))

  def __irshift__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.right_shift(self, other, out=(self,))

  def __iand__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.bitwise_and(self, other, out=(self,))

  def __ixor__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.bitwise_xor(self, other, out=(self,))

  def __ior__(self, other) -> 'array_mixins.MutableArray':
    return np.core.umath.bitwise_or(self, other, out=(self,))

  def __setitem__(self, index, obj):
    return np.asarray(self).__setitem__(index, obj)
