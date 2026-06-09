# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
"""Basic sanity checks for brainstate/alignment/base.py."""

from connectomics.brainstate.alignment import base
import numpy as np
from google3.testing.pybase import googletest


class BaseTest(googletest.TestCase):

  def test_asif_aligner(self):
    """Basic sanity checks."""
    num_train_cells = 42
    num_test_cells = 8
    m1dim = 100
    m2dim = 104
    aligner = base.AsifAligner(
        modality1_train_embeddings=np.random.rand(num_train_cells, m1dim),
        modality2_train_embeddings=np.random.rand(num_train_cells, m2dim))
    self.assertEqual(
        aligner.transform_modality1(
            np.random.rand(num_test_cells, m1dim)).shape,
        (num_test_cells, num_train_cells))
    self.assertEqual(
        aligner.transform_modality2(
            np.random.rand(num_test_cells, m2dim)).shape,
        (num_test_cells, num_train_cells))
    self.assertEqual(
        aligner.backproject_to_modality1(
            np.random.rand(num_test_cells, num_train_cells)).shape,
        (num_test_cells, m1dim))
    self.assertEqual(base.AsifAligner.compute_cross_modal_distance(
        np.random.rand(10, 100), np.random.rand(10, 100)).shape, (10, 10))


if __name__ == "__main__":
  googletest.main()
