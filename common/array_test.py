"""Tests for array."""

from absl.testing import absltest
from connectomics.common import array
import numpy as np

ImmutableArray = array.ImmutableArray
MutableArray = array.MutableArray


class ImmutableArrayTest(absltest.TestCase):

  def assertIsTypedCorrectly(self, arr):
    self.assertIsInstance(arr, MutableArray)
    self.assertIsInstance(arr, ImmutableArray)
    self.assertIsInstance(arr, np.ndarray)

  def test_construction(self):
    a = ImmutableArray([1, 2, 3])
    np.testing.assert_array_equal([1, 2, 3], a)

    # Copy by default, same as np.array
    b = ImmutableArray(a)
    self.assertFalse(np.may_share_memory(a, b))

    # Ensure zero-copy is possible
    c = ImmutableArray(a, zero_copy=True)
    self.assertTrue(np.may_share_memory(a, c))

    # Ensure we can downcast
    m = MutableArray([4, 5, 6])
    a = ImmutableArray(m)
    np.testing.assert_array_equal([4, 5, 6], a)
    self.assertFalse(np.may_share_memory(a, b))

    a = ImmutableArray(m, zero_copy=True)
    np.testing.assert_array_equal([4, 5, 6], a)
    self.assertTrue(np.may_share_memory(a, m))

  def test_dtype(self):
    a = ImmutableArray([1, 2, 3], dtype=int)
    np.testing.assert_array_equal([1, 2, 3], a)
    self.assertEqual(int, a.dtype)

    a = ImmutableArray([1., 2.4, 3.8], dtype=float)
    np.testing.assert_array_equal([1., 2.4, 3.8], a)
    self.assertEqual(float, a.dtype)

  def test_tolist(self):
    a = ImmutableArray([1, 2, 3])
    self.assertIsInstance(a.tolist(), list)
    self.assertEqual([1, 2, 3], a.tolist())

  def test_indexing(self):
    a = ImmutableArray([1, 2, 3])
    self.assertEqual(1, a[0])
    self.assertEqual([1, 2, 3], list(a[:3]))

  def test_numpy_ufuncs(self):
    a = ImmutableArray([1, 2, 3])

    out = a + [1, 2, 3]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([2, 4, 6], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a - [4, -1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([-3, 3, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a / [2, 2, 2]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([0.5, 1, 1.5], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))
    self.assertEqual(float, out.dtype)

    out = a // [2, 2, 2]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([0, 1, 1], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))
    self.assertEqual(int, out.dtype)

    out = a * [3, 2, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([3, 4, 0], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))
    self.assertEqual(int, out.dtype)

    out = a * [3., 2.1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([3., 4.2, 0], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))
    self.assertEqual(float, out.dtype)

    out = a & [3, 2, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([1, 2, 0], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a | [2, 1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([3, 3, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a ^ [3, 1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([2, 3, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a << [7, 1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([128, 4, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a >> [2, 1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([0, 1, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a**[3, 2, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([1, 4, 1], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

  def test_copy(self):
    a = ImmutableArray([1, 2, 3])

    b = a.copy()
    self.assertIsTypedCorrectly(b)
    self.assertIsNot(a, b)
    self.assertFalse(np.may_share_memory(a, b))
    np.testing.assert_array_equal(a, b)
    self.assertEqual([1, 2, 3], a.tolist())
    self.assertEqual([1, 2, 3], b.tolist())

    b += [1, 2, 3]
    self.assertEqual([1, 2, 3], a.tolist())
    self.assertEqual([2, 4, 6], b.tolist())

  def test_reflexive_ufuncs(self):
    a = ImmutableArray([17, 22, 38])
    other = np.array([2, 3, 4])
    np.testing.assert_array_equal(a + other, other + a)

    np.testing.assert_array_equal([15, 19, 34], a - other)
    np.testing.assert_array_equal([-15, -19, -34], other - a)

    np.testing.assert_array_equal(other * a, a * other)

    self.assertSequenceAlmostEqual([8.5, 7.3333333, 9.5], a / other)
    self.assertSequenceAlmostEqual([0.11764705, 0.136363636, 0.10526315789],
                                   other / a)


class MutableArrayTest(absltest.TestCase):

  def assertIsTypedCorrectly(self, arr):
    self.assertIsInstance(arr, MutableArray)
    self.assertIsInstance(arr, ImmutableArray)
    self.assertIsInstance(arr, np.ndarray)

  def test_construction(self):
    a = MutableArray([1, 2, 3])
    np.testing.assert_array_equal([1, 2, 3], a)

    # Copy by default, same as np.array
    b = MutableArray(a)
    self.assertFalse(np.may_share_memory(a, b))

    # Ensure zero-copy is possible
    c = MutableArray(a, zero_copy=True)
    self.assertTrue(np.may_share_memory(a, c))

  def test_numpy_ufuncs(self):
    a = MutableArray([1, 2, 3])

    out = a + [1, 2, 3]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([2, 4, 6], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a - [4, -1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([-3, 3, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a / [2, 2, 2]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([0.5, 1, 1.5], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))
    self.assertEqual(float, out.dtype)

    out = a // [2, 2, 2]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([0, 1, 1], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))
    self.assertEqual(int, out.dtype)

    out = a * [3, 2, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([3, 4, 0], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))
    self.assertEqual(int, out.dtype)

    out = a * [3., 2.1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([3., 4.2, 0], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))
    self.assertEqual(float, out.dtype)

    out = a & [3, 2, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([1, 2, 0], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a | [2, 1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([3, 3, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a ^ [3, 1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([2, 3, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a << [7, 1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([128, 4, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a >> [2, 1, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([0, 1, 3], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))

    out = a**[3, 2, 0]
    self.assertIsTypedCorrectly(out)
    self.assertEqual([1, 4, 1], out.tolist())
    self.assertFalse(np.may_share_memory(out, a))


if __name__ == '__main__':
  absltest.main()
