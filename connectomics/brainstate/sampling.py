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
"""Utils for selecting consistent dataset splits across experiments."""

from collections.abc import Sequence
import dataclasses
from typing import Self
import numpy as np


@dataclasses.dataclass
class DatasetMultiSplit:
  sample_id_splits: list[np.ndarray]
  label_splits: list[np.ndarray]


@dataclasses.dataclass
class DatasetSplit:
  """Represents split of dataset into train/valid/test for ML experiments."""
  train_ids: np.ndarray
  valid_ids: np.ndarray
  test_ids: np.ndarray
  train_labels: np.ndarray
  valid_labels: np.ndarray
  test_labels: np.ndarray

  def upsampled(self, upsample_factor: int, dataset_len: int) -> Self:
    train_ids, valid_ids, test_ids = [], [], []
    train_labels, valid_labels, test_labels = [], [], []
    for i in range(upsample_factor):
      train_ids.append(self.train_ids + i * dataset_len)
      valid_ids.append(self.valid_ids + i * dataset_len)
      test_ids.append(self.test_ids + i * dataset_len)
      train_labels.append(self.train_labels)
      valid_labels.append(self.valid_labels)
      test_labels.append(self.test_labels)
    return DatasetSplit(
        np.concatenate(train_ids),
        np.concatenate(valid_ids),
        np.concatenate(test_ids),
        np.concatenate(train_labels),
        np.concatenate(valid_labels),
        np.concatenate(test_labels),
    )


def concatenate_splits(
    splits: Sequence[DatasetSplit], dataset_lengths: Sequence[int]
) -> DatasetSplit:
  """Concatenate DatasetSplits by incrementing by previous dataset length."""
  train_ids, valid_ids, test_ids = [], [], []
  train_labels, valid_labels, test_labels = [], [], []
  increment = 0
  for split, l in zip(splits, dataset_lengths):
    train_ids.append(split.train_ids + increment)
    valid_ids.append(split.valid_ids + increment)
    test_ids.append(split.test_ids + increment)
    train_labels.append(split.train_labels)
    valid_labels.append(split.valid_labels)
    test_labels.append(split.test_labels)
    increment += l
  return DatasetSplit(
      np.concatenate(train_ids),
      np.concatenate(valid_ids),
      np.concatenate(test_ids),
      np.concatenate(train_labels),
      np.concatenate(valid_labels),
      np.concatenate(test_labels),
  )


def split_indices_by_labels(
    labels: Sequence[int], ratios: Sequence[float],
    rng: np.random.RandomState) -> list[np.ndarray]:
  """Low-level function to generate arbitrary splits balanced by labels.

  Args:
    labels: The data labels to balance splits by.
    ratios: The ratios of the splits. A final implicit split will be included,
      so e.g. passing ratios=[0.8, 0.1] will result in an 80/10/10 percent
      split. (If ratios adds up to >=1 then the trailing splits will be empty.)
    rng: A np.random.RandomState to use for splitting.

  Returns:
    The indices into labels for each split (total len(ratios) + 1). This can be
  used to index into e.g. example IDs as well.
  """
  split_indices = []
  for label in np.unique(labels):
    label_indices = np.flatnonzero(labels == label)
    rng.shuffle(label_indices)
    # Splits are rounded this way for backward compatibility.
    n = len(label_indices)
    splits = np.cumsum([int(ratio * n) for ratio in ratios])
    split_indices.append(np.split(label_indices, splits))

  return [np.concat(si) for si in zip(*split_indices)]    # Reshape.


def split_dataset_by_ratios(
    sample_ids: Sequence[int], seed: int, ratios: Sequence[float],
    labels: Sequence[int] | None = None,
) -> DatasetMultiSplit:
  """Splits dataset and labels by given ratios, balanced by labels.

  Args:
    sample_ids: IDs to identify examples, e.g. cell ids
    seed: random seed
    ratios: The ratios of the splits. A final implicit split will be included,
      so e.g. passing ratios=[0.8, 0.1] will result in an 80/10/10 percent
      split. (If ratios adds up to >=1 then the trailing splits will be empty.)
    labels: A label array of the same length as sample_ids. When passed, the
      samples for each label are distributed among the splits according to their
      ratios.

  Returns:
    DatasetMultiSplit
  """
  if len(np.unique(sample_ids)) != len(sample_ids):
    raise ValueError("Found repeated sample ids")

  if labels is not None:
    if len(labels) != len(sample_ids):
      raise ValueError("labels must be of the same length as sample_ids")
    labels = np.array(labels, dtype=int)
  else:
    labels = np.zeros(len(sample_ids), dtype=int)

  # Sort by cell id to make samples reproducible even if the samples are passed
  # in a different order
  sample_ids = np.array(sample_ids, dtype=int)
  sample_id_sorting = np.argsort(sample_ids)
  sample_ids = sample_ids[sample_id_sorting]
  labels = labels[sample_id_sorting]
  rng = np.random.RandomState(seed)
  split_indices = split_indices_by_labels(labels, ratios, rng)

  sample_id_splits = [sample_ids[s] for s in split_indices]
  label_splits = [labels[s] for s in split_indices]
  return DatasetMultiSplit(sample_id_splits, label_splits)


def split_dataset(
    sample_ids: Sequence[int], seed: int, train_ratio: float,
    valid_ratio: float = 0, labels: Sequence[int] | None = None,
) -> DatasetSplit:
  """Splits dataset into train / valid / test splits.

  Args:
    sample_ids: IDs to identify examples, e.g. cell ids
    seed: random seed
    train_ratio: ratio of training examples to sample (0-1)
    valid_ratio: ratio of validation examples to sample (0-1)
    labels: Optional label array of the same length as sample_ids. When passed,
      the samples for each label are distributed among the splits according to
      their ratios.

  Returns:
    DatasetSplit
  """
  if train_ratio + valid_ratio > 1:
    raise ValueError(
        "train_ratio and valid_ratio must be <= 1: "
        f"{train_ratio}, {valid_ratio}"
    )
  ratios = train_ratio, valid_ratio
  split = split_dataset_by_ratios(sample_ids, seed, ratios, labels)
  return DatasetSplit(
      train_ids=split.sample_id_splits[0],
      valid_ids=split.sample_id_splits[1],
      test_ids=split.sample_id_splits[2],
      train_labels=split.label_splits[0],
      valid_labels=split.label_splits[1],
      test_labels=split.label_splits[2],
  )


def cross_validation_split_dataset(
    sample_ids: Sequence[int], seed: int, num_splits: int,
    labels: Sequence[int] | None = None) -> DatasetMultiSplit:
  """Splits dataset into num_splits, optionally balanced by labels.

  Args:
    sample_ids: IDs to identify examples, e.g. cell ids
    seed: random seed
    num_splits: The number of splits to produce; typically 5- or 10-fold.
    labels: Optional label array of the same length as sample_ids. When passed,
      the samples for each label are distributed among the splits according to
      their ratios.

  Returns:
    DatasetMultiSplit
  """
  ratios = [1.0 / num_splits] * (num_splits - 1)
  return split_dataset_by_ratios(sample_ids, seed, ratios, labels)
