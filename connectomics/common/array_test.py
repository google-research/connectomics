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
"""Tests for array."""

from absl.testing import absltest
from connectomics.common import array
import numpy as np


class TestNormalizeIndex(absltest.TestCase):

  def test_normalize_index(self):
    limits = (1, 12, 11, 10)
    self.assertEqual((
        slice(0, 1),
        slice(3, 4),
        slice(2, 3),
        slice(1, 2),
    ), array.normalize_index((0, 3, 2, 1), limits))

    self.assertEqual((
        slice(0, 1),
        slice(7, 12, 1),
        slice(6, 2, 1),
        slice(1, 2),
    ), array.normalize_index(np.s_[0, 7:, 6:2, 1], limits))

  def test_point_lookup(self):
    points = (
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    )

    limits = (1, 12, 11, 10)
    index_exp = (0,) + points
    self.assertEqual(index_exp, array.normalize_index(index_exp, limits))


if __name__ == '__main__':
  absltest.main()
