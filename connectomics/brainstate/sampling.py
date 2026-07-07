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
import hashlib
from typing import Self
import numpy as np


@dataclasses.dataclass
class DatasetMultiSplit:
  """Represents N-way splits, usually equal as used for cross-validations."""
  sample_id_splits: list[np.ndarray]
  label_splits: list[np.ndarray]

  def __str__(self):
    sample_id_split_lens = [len(s) for s in self.sample_id_splits]
    label_split_lens = [len(s) for s in self.label_splits]
    return (
        f'DatasetMultiSplit(sample_id_splits={sample_id_split_lens}, '
        f'label_splits={label_split_lens}, signature={self.signature()})')

  def signature(self):
    md5 = hashlib.md5()
    for s in self.sample_id_splits:
      md5.update(s.tobytes())
    for s in self.label_splits:
      md5.update(s.tobytes())
    return md5.hexdigest()


@dataclasses.dataclass
class DatasetSplit:
  """Represents split of dataset into train/valid/test for ML experiments."""
  train_ids: np.ndarray
  valid_ids: np.ndarray
  test_ids: np.ndarray
  train_labels: np.ndarray
  valid_labels: np.ndarray
  test_labels: np.ndarray

  def __str__(self):
    return (
        f'DatasetSplit(train_ids={len(self.train_ids)}, '
        f'valid_ids={len(self.valid_ids)}, test_ids={len(self.test_ids)}, '
        f'train_labels={len(self.train_labels)}, '
        f'valid_labels={len(self.valid_labels)}, '
        f'test_labels={len(self.test_labels)}, '
        f'signature={self.signature()})')

  def signature(self):
    md5 = hashlib.md5()
    md5.update(self.train_ids.tobytes())
    md5.update(self.test_ids.tobytes())
    md5.update(self.valid_ids.tobytes())
    md5.update(self.train_labels.tobytes())
    md5.update(self.test_labels.tobytes())
    md5.update(self.valid_labels.tobytes())
    return md5.hexdigest()

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
    return DatasetSplit(  # pyrefly: ignore[bad-return]
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
      split. (If ratios adds up to >= 1 then the trailing splits will be empty.)
    labels: A label array of the same length as sample_ids. When passed, the
      samples for each label are distributed among the splits according to their
      ratios.

  Returns:
    DatasetMultiSplit
  """
  if len(np.unique(sample_ids)) != len(sample_ids):
    raise ValueError('Found repeated sample ids')

  if labels is not None:
    if len(labels) != len(sample_ids):
      raise ValueError('labels must be of the same length as sample_ids')
    labels = np.array(labels, dtype=int)  # pyrefly: ignore[bad-assignment]
  else:
    labels = np.zeros(len(sample_ids), dtype=int)  # pyrefly: ignore[bad-assignment]

  # Sort by cell id to make samples reproducible even if the samples are passed
  # in a different order
  sample_ids = np.array(sample_ids, dtype=int)  # pyrefly: ignore[bad-assignment]
  sample_id_sorting = np.argsort(sample_ids)
  sample_ids = sample_ids[sample_id_sorting]  # pyrefly: ignore[bad-index]
  labels = labels[sample_id_sorting]  # pyrefly: ignore[bad-index, unsupported-operation]
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
        'train_ratio and valid_ratio must be <= 1: '
        f'{train_ratio}, {valid_ratio}'
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


def _concat(to_concat):
  return np.concatenate(to_concat) if to_concat else np.array([])


def cross_validation_split_dataset(
    sample_ids: Sequence[int], seed: int, num_splits: int,
    labels: Sequence[int] | None = None, num_splits_for_valid: int = 1,
    num_splits_for_test: int = 1) -> list[DatasetSplit]:
  """Splits dataset into num_splits, optionally balanced by labels.

  Args:
    sample_ids: IDs to identify examples, e.g. cell ids
    seed: random seed
    num_splits: The number of splits to produce; typically 5- or 10-fold.
    labels: Optional label array of the same length as sample_ids. When passed,
      the samples for each label are distributed among the splits according to
      their ratios.
    num_splits_for_valid: How many split buckets to assign to validation set.
      Remaining buckets not assigned to validation or test become train set.
    num_splits_for_test: How many split buckets to assign to test set.

  Returns:
    list[DatasetSplit] where each split has the indicated proportions.
  """
  assert num_splits_for_valid + num_splits_for_test < num_splits
  ratios = [1.0 / num_splits] * (num_splits - 1)
  equal_splits = split_dataset_by_ratios(sample_ids, seed, ratios, labels)

  # Initialize.
  valid_splits = np.arange(0, num_splits_for_valid)
  test_splits = np.arange(0, num_splits_for_test) + num_splits_for_valid
  train_splits = np.arange(
      num_splits_for_valid + num_splits_for_test, num_splits)
  # Build DatasetSplits.
  dataset_splits = []
  for _ in range(num_splits):
    dataset_splits.append(
        DatasetSplit(
            train_ids=_concat(
                [equal_splits.sample_id_splits[s] for s in train_splits]),
            valid_ids=_concat(
                [equal_splits.sample_id_splits[s] for s in valid_splits]),
            test_ids=_concat(
                [equal_splits.sample_id_splits[s] for s in test_splits]),
            train_labels=_concat(
                [equal_splits.label_splits[s] for s in train_splits]),
            valid_labels=_concat(
                [equal_splits.label_splits[s] for s in valid_splits]),
            test_labels=_concat(
                [equal_splits.label_splits[s] for s in test_splits]),
        ))

    # Rotate.
    valid_splits = (valid_splits + 1) % num_splits
    test_splits = (test_splits + 1) % num_splits
    train_splits = (train_splits + 1) % num_splits
  return dataset_splits
