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
"""Tests for sampling module."""

from connectomics.brainstate import sampling
import numpy as np
from google3.testing.pybase import googletest


class SamplingTest(googletest.TestCase):

  def test_split_indices_by_labels(self):
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ratios = [0.8]
    rng = np.random.RandomState(22222)
    splits = sampling.split_indices_by_labels(labels, ratios, rng)
    np.testing.assert_array_equal(splits[0], [4, 0, 3, 2, 9, 6, 8, 5])
    np.testing.assert_array_equal(splits[1], [1, 7])

  def test_empty_split(self):
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ratios = [0.8, 0.0]
    rng = np.random.RandomState(22222)
    splits = sampling.split_indices_by_labels(labels, ratios, rng)
    np.testing.assert_array_equal(splits[0], [4, 0, 3, 2, 9, 6, 8, 5])
    np.testing.assert_array_equal(splits[1], [])
    np.testing.assert_array_equal(splits[2], [1, 7])

  def test_split_dataset(self):
    sample_ids = range(10)
    seed = 22222
    train_ratio = 0.7
    valid_ratio = 0.1  # Test 0.2 implicit.
    split = sampling.split_dataset(sample_ids, seed, train_ratio, valid_ratio)
    np.testing.assert_array_equal(split.train_ids, [3, 5, 9, 4, 6, 7, 0])
    np.testing.assert_array_equal(split.valid_ids, [8])
    np.testing.assert_array_equal(split.test_ids, [2, 1])
    np.testing.assert_array_equal(split.train_labels, [0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(split.valid_labels, [0])
    np.testing.assert_array_equal(split.test_labels, [0, 0])

    # Results should be balanced by labels.
    labels = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    split = sampling.split_dataset(
        sample_ids, seed, train_ratio, valid_ratio, labels)
    np.testing.assert_array_equal(split.train_ids, [2, 0, 7, 4, 6, 8])
    np.testing.assert_array_equal(split.valid_ids, [])
    np.testing.assert_array_equal(split.test_ids, [1, 9, 3, 5])
    np.testing.assert_array_equal(split.train_labels, [1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(split.valid_labels, [])
    np.testing.assert_array_equal(split.test_labels, [1, 2, 2, 2])

  def test_upsample(self):
    sample_ids = range(10)
    labels = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    seed = 22222
    train_ratio = 0.7
    valid_ratio = 0.1  # Test 0.2 implicit.
    split = sampling.split_dataset(
        sample_ids, seed, train_ratio, valid_ratio, labels
    )
    upsampled = split.upsampled(upsample_factor=2, dataset_len=10)
    np.testing.assert_array_equal(
        upsampled.train_ids, [2, 0, 7, 4, 6, 8, 12, 10, 17, 14, 16, 18]
    )
    np.testing.assert_array_equal(upsampled.valid_ids, [])
    np.testing.assert_array_equal(
        upsampled.test_ids, [1, 9, 3, 5, 11, 19, 13, 15]
    )
    np.testing.assert_array_equal(
        upsampled.train_labels, [1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2]
    )
    np.testing.assert_array_equal(upsampled.valid_labels, [])
    np.testing.assert_array_equal(
        upsampled.test_labels, [1, 2, 2, 2, 1, 2, 2, 2]
    )

  def test_concatenate_splits(self):
    sample_ids = range(10)
    seed = 22222
    train_ratio = 0.7
    valid_ratio = 0.1  # Test 0.2 implicit.
    split = sampling.split_dataset(sample_ids, seed, train_ratio, valid_ratio)

    labels = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    split2 = sampling.split_dataset(
        sample_ids, seed, train_ratio, valid_ratio, labels
    )

    dataset_lengths = [10, 10]
    concat = sampling.concatenate_splits([split, split2], dataset_lengths)
    np.testing.assert_array_equal(
        concat.train_ids, [3, 5, 9, 4, 6, 7, 0, 12, 10, 17, 14, 16, 18]
    )
    np.testing.assert_array_equal(concat.valid_ids, [8])
    np.testing.assert_array_equal(concat.test_ids, [2, 1, 11, 19, 13, 15])
    np.testing.assert_array_equal(
        concat.train_labels, [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    )
    np.testing.assert_array_equal(concat.valid_labels, [0])
    np.testing.assert_array_equal(concat.test_labels, [0, 0, 1, 2, 2, 2])

  def test_cross_validation_split_dataset(self):
    sample_ids = range(10)
    seed = 22222
    num_splits = 5
    splits = sampling.cross_validation_split_dataset(
        sample_ids, seed, num_splits).sample_id_splits
    np.testing.assert_array_equal(splits[0], [3, 5])
    np.testing.assert_array_equal(splits[1], [9, 4])
    np.testing.assert_array_equal(splits[2], [6, 7])
    np.testing.assert_array_equal(splits[3], [0, 8])
    np.testing.assert_array_equal(splits[4], [2, 1])

    num_splits = 2
    splits = sampling.cross_validation_split_dataset(
        sample_ids, seed, num_splits).sample_id_splits
    np.testing.assert_array_equal(splits[0], [3, 5, 9, 4, 6])
    np.testing.assert_array_equal(splits[1], [7, 0, 8, 2, 1])


if __name__ == "__main__":
  googletest.main()
