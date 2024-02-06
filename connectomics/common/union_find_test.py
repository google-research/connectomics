# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for union_find."""

from absl.testing import absltest
from connectomics.common import union_find


class UnionFindTest(absltest.TestCase):

  def test_union_find(self):
    uf = union_find.UnionFind()
    for i in range(10):
      self.assertEqual(i, uf.Find(i))
      self.assertTrue(uf.IsSingleton(i))

    # Merge pairs: (0, 5), (1, 6), ..
    for i in range(5):
      uf.Union(i, i + 5)
      self.assertEqual(uf.Find(i),
                       uf.Find(i + 5))
      self.assertFalse(uf.IsSingleton(i))
      self.assertFalse(uf.IsSingleton(i + 5))
      self.assertNotEqual(uf.Find(i),
                          uf.Find(i + 1))

    # Merge all sets together.
    for i in range(4):
      uf.Union(9, i)
      self.assertEqual(uf.Find(9), uf.Find(i))

    for i in range(10):
      self.assertEqual(uf.Find(0), uf.Find(i))

  def test_bool_cast(self):
    uf = union_find.UnionFind()
    self.assertFalse(uf)

    uf.Find(2)
    self.assertFalse(uf)

    uf.Union(1, 2)
    self.assertTrue(uf)


if __name__ == '__main__':
  absltest.main()
