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
"""General utilities."""

from typing import Sequence, Any, Optional
import numpy as np


def upsample_labels(labels: Sequence[int],
                    label_idx: Optional[Sequence[int]] = None) -> np.ndarray:
  """Upsamples dataset such that all classes are balanced.

  Args:
    labels: list or array of labels
    label_idx: list or array of indices. If not provided, indices will be
      created as label_idx=np.arange(len(labels))

  Raises:
    ValueError: if length of labels and label_idx does not match

  Returns:
    resampled_idx: array of resampled indices
  """
  if label_idx is None:
    label_idx = np.arange(len(labels))
  else:
    if len(label_idx) != len(labels):
      raise ValueError(
          "labels and label_idx need to be of same length. "
          f"Currently is {len(labels)} and {len(label_idx)} respectively."
      )
    label_idx = np.asarray(label_idx)

  labels_unique, label_counts = np.unique(labels, return_counts=True)
  max_label_count = np.max(label_counts)

  resampled_idx = []
  for u_class in labels_unique:
    resampled_idx.extend(
        upsample_data(label_idx[labels == u_class], max_label_count))
  return np.array(resampled_idx)


def upsample_data(samples: Sequence[Any],
                  n_samples: int,
                  random_seed: int = 42) -> np.ndarray:
  """Upsamples data to n_samples.

  Args:
    samples: data samples of any kind
    n_samples: number of samples to upsample to. Has to be > len(samples)
    random_seed: random seed

  Raises:
    ValueError: if length of samples exceeds n_samples

  Returns:
    resampled_samples: upsampled samples
  """
  if len(samples) > n_samples:
    raise ValueError(
        "For upsampling there need to be fewer samples than n_samples.")

  if len(samples) == n_samples:
    return np.array(samples)

  resampled_samples = []

  while len(resampled_samples) + len(samples) <= n_samples:
    resampled_samples.extend(samples)

  if len(resampled_samples) < n_samples:
    resampled_samples.extend(
        downsample_data(
            samples,
            n_samples - len(resampled_samples),
            random_seed=random_seed))
  return np.array(resampled_samples)


def downsample_data(samples: Sequence[Any],
                    n_samples: int,
                    random_seed: int = 42) -> np.ndarray:
  """Downsamples data to n_samples by selecting a random subset.

  Args:
    samples: data samples of any kind
    n_samples: number of samples to downsample to. Has to be > len(samples)
    random_seed: random seed

  Raises:
    ValueError: if length of samples does not exceed n_samples

  Returns:
    resampled_samples: downsampled samples
  """
  if len(samples) <= n_samples:
    raise ValueError(
        "For downsampling there need to be more samples than n_samples.")

  random_state = np.random.RandomState(random_seed)
  resampled_samples = random_state.choice(samples, n_samples, replace=False)

  return np.array(resampled_samples)
