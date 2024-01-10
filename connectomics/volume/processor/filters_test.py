# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
"""Tests for the filters module."""

from absl.testing import absltest
from connectomics.volume.processor import filters

import numpy as np


class ApplyFilterTest(absltest.TestCase):

  def test_context(self):
    processor = filters.ApplyFilter(
        (33, 44), 'max', '2d'
    )
    pre, post = processor.context()

    np.testing.assert_array_equal(pre, [22, 16, 0])
    np.testing.assert_array_equal(post, [22, 17, 0])

    processor = filters.ApplyFilter(
        (1, 33, 44), 'max', '3d'
    )
    pre, post = processor.context()

    np.testing.assert_array_equal(pre, [22, 16, 0])
    np.testing.assert_array_equal(post, [22, 17, 1])



if __name__ == "__main__":
  absltest.main()
