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
"""Untyped tests for array.

Since we're using PyType to enforce mutability constraints ahead-of-time, these
tests ensure that if the type system was somehow disabled there would be a
corresponding runtime error for immutable types.
"""

from typing import Any  # pylint: disable=unused-import

from absl.testing import absltest
from connectomics.common import array  # type: Any

ImmutableArray = array.ImmutableArray


class ImmutableArrayRuntimeErrorTest(absltest.TestCase):

  def test_inplace_add(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a += [4, 5, 6]

  def test_inplace_sub(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a -= [4, 5, 6]

  def test_inplace_mul(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a *= [4, 5, 6]

  def test_inplace_matmul(self):
    a = ImmutableArray([1, 2, 3])
    # In-place matmul is not yet supported, so ensure that this TypeErrors for
    # now. Will need to be updated if Python3 ever supports in-place matmul
    # operations.
    with self.assertRaises((TypeError, ValueError)):
      a @= [4, 5, 6]

  def test_inplace_truediv(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a /= [4, 5, 6]

  def test_inplace_floordiv(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a //= [4, 5, 6]

  def test_inplace_mod(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a %= [4, 5, 6]

  def test_inplace_pow(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a **= [4, 5, 6]

  def test_inplace_lshift(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a <<= [4, 5, 6]

  def test_inplace_rshift(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a >>= [4, 5, 6]

  def test_inplace_and(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a &= [4, 5, 6]

  def test_inplace_xor(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a ^= [4, 5, 6]

  def test_inplace_or(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a |= [4, 5, 6]

  def test_indexing(self):
    a = ImmutableArray([1, 2, 3])
    with self.assertRaises(ValueError):
      a[1] = 7


if __name__ == '__main__':
  absltest.main()
