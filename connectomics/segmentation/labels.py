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
"""Routines for manipulating numpy arrays of segmentation data."""

from typing import Iterable, List, Tuple

import numpy as np


def relabel(labels: np.ndarray, orig_ids: Iterable[int],
            new_ids: Iterable[int]) -> np.ndarray:
  """Relabels `labels` by mapping `orig_ids` to `new_ids`.

  Args:
    labels: ndarray of segment IDs
    orig_ids: iterable of existing segment IDs
    new_ids: iterable of new segment IDs (len(new_ids) == len(orig_ids))

  Returns:
    int64 ndarray with updated segment IDs
  """
  orig_ids = np.asarray(orig_ids)
  new_ids = np.asarray(new_ids)
  assert orig_ids.size == new_ids.size

  relabel_hashtable = {
      new_id: orig_id for new_id, orig_id in zip(orig_ids, new_ids)
  }
  relabeled = [relabel_hashtable[l] for l in labels.flatten()]
  return np.asarray(relabeled).reshape(labels.shape)


def make_contiguous(
    labels: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
  """Relabels 'labels' so that its ID space is dense.

  If N is the number of unique ids in 'labels', the new IDs will cover the range
  [0..N-1].

  Args:
    labels: ndarray of segment IDs

  Returns:
    tuple of:
      ndarray of dense segment IDs
      list of (old_id, new_id) pairs
  """
  orig_ids = np.unique(np.append(labels, np.uint64(0)))
  new_ids = np.arange(len(orig_ids))
  return relabel(labels, orig_ids, new_ids), list(zip(orig_ids, new_ids))


def are_equivalent(label_a: np.ndarray, label_b: np.ndarray) -> bool:
  """Returns whether two volumes contain equivalent segmentations.

  Segmentations are considered equivalent when there exists a 1:1 map between
  the two arrays -- that is segment shapes and locations have to be exactly
  the same but their IDs can be shuffled.

  Args:
    label_a: numpy array of segments
    label_b: numpy array of segments

  Returns:
    True iff the segmentations 'label_a' and 'label_b' are equivalent.
  """
  if label_a.shape != label_b.shape:
    return False

  a_to_b = {}
  b_to_a = {}

  for a, b in set(zip(label_a.flat, label_b.flat)):
    if a not in a_to_b:
      a_to_b[a] = b
    if b not in b_to_a:
      b_to_a[b] = a

    if a_to_b[a] != b:
      return False

    if b_to_a[b] != a:
      return False

  return True
