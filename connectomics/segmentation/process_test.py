# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""Tests for segmentation post-processing."""
from absl.testing import absltest
from connectomics.segmentation import process as process_segmentation
import numpy as np


class ProcessSegmentationTest(absltest.TestCase):

  def test_recompute_connected_components(self):
    seg = process_segmentation.recompute_connected_components(
        np.array([0, 1, 0, 2, 0, 3, 1]), offset=10)
    self.assertLen(np.unique(seg), 5)
    self.assertEqual(np.max(seg), 14)


if __name__ == "__main__":
  absltest.main()
