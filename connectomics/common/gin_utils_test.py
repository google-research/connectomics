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
"""Tests for gin utilities."""

from absl.testing import absltest
from connectomics.common import gin_utils


class GinUtilsTest(absltest.TestCase):

  def test_list_to_str(self):
    self.assertEqual(gin_utils.list_to_str(['f', 'o', 'o'], separator=''),
                     'foo')
    self.assertEqual(gin_utils.list_to_str(['f', 'o', 'o'], separator='_'),
                     'f_o_o')
    self.assertEqual(gin_utils.list_to_str(['f', 'o', 'o'], separator='/'),
                     'f/o/o')

  def test_integer_division(self):
    self.assertEqual(gin_utils.integer_division(10, 2), 5)
    self.assertEqual(gin_utils.integer_division(11, 2), 5)


if __name__ == '__main__':
  absltest.main()
