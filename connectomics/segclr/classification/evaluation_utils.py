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
"""Utilities for evaluating models."""

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def sample_n(labels: np.ndarray,
             n_samples: int,
             guarantee_n: int = 1,
             secondary_labels: Optional[np.ndarray] = None,
             secondary_label_weights: Optional[Dict[int, float]] = None,
             random_seed: int = 42) -> Sequence[int]:
  """Samples n samples randomly while guaranteeing a min number from each label.

  Args:
    labels: N label array
    n_samples: Number of samples
    guarantee_n: Minimum number of samples from each label. If guarantee exceeds
      the number of labels of a given class, it is corrected down for that
      class.
    secondary_labels: Additional labels that are not predicted but can be used
      to weight the sample selection; e.g. compartment information
    secondary_label_weights: Weights assigned to the secondary labels and
      taken into account when subsampling the data
    random_seed: Random seed

  Returns:
    idx: indices of sampled elements.
  """
  if secondary_label_weights is None:
    secondary_label_weights = {}

  random_state = np.random.RandomState(random_seed)
  if secondary_labels is None:
    secondary_labels = np.ones(len(labels), dtype=int)

  assert len(secondary_labels) == len(labels)

  u_labels = np.unique(labels)

  idx = []
  for u_label in u_labels:
    label_idx = np.where(labels == u_label)[0]

    u_snd_labels = np.unique(secondary_labels[label_idx])

    weights = [secondary_label_weights.get(l, 1.) for l in u_snd_labels]
    weights = np.array(weights) / np.sum(weights)
    for u_snd_label, weight in zip(u_snd_labels, weights):
      u_idx = label_idx[secondary_labels[label_idx] == u_snd_label]
      idx.extend(
          random_state.choice(
              u_idx,
              np.min([int(guarantee_n * weight),
                      len(u_idx)]),
              replace=False))

  possible_idx = np.arange(len(labels))
  possible_idx = possible_idx[~np.isin(possible_idx, idx)]

  if n_samples > len(idx):
    idx.extend(
        random_state.choice(
            possible_idx,
            np.min([len(possible_idx), n_samples - len(idx)]),
            replace=False))

  return idx


def f1_scorer(y_true: Sequence[int], y_pred: Sequence[int],
              label: int) -> float:
  """Helper function to wrap sklearn's f1_score for a given label.

  Args:
    y_true: List of ground truth labels
    y_pred: List of predicted labels
    label: Label to evaluate

  Returns:
    f1 score: F1 Score
  """
  return f1_score(np.asarray(y_true) == label, np.asarray(y_pred) == label)


def precision_scorer(y_true: Sequence[int], y_pred: Sequence[int],
                     label: int) -> float:
  """Helper function to wrap sklearn's precision_score for a given label."""
  return precision_score(
      np.asarray(y_true) == label,
      np.asarray(y_pred) == label)


def recall_scorer(y_true: Sequence[int], y_pred: Sequence[int],
                  label: int) -> float:
  """Helper function to wrap sklearn's recall_score for a given label.

  Args:
    y_true: List of ground truth labels
    y_pred: List of predicted labels
    label: Label to evaluate

  Returns:
    recall Score
  """
  return recall_score(np.asarray(y_true) == label, np.asarray(y_pred) == label)


def evaluate_prediction(labels_test: Sequence[int],
                        pred_test: Sequence[int],
                        incl_per_class: bool = True) -> Dict[str, Any]:
  """Computes evaluation metrics for a prediction.

  Args:
    labels_test: ground truth labels
    pred_test: predicted labels
    incl_per_class: whether to include per class scores

  Returns:
    result_dict: Dictionary with individual results
  """
  result_dict = {}

  labels_test = np.array(labels_test)
  pred_test = np.array(pred_test)

  if incl_per_class:
    for label in np.unique(labels_test):
      result_dict[f"F1(class={label})"] = f1_scorer(labels_test, pred_test,
                                                    label)
      result_dict[f"Precision(class={label})"] = precision_scorer(
          labels_test, pred_test, label)
      result_dict[f"Recall(class={label})"] = recall_scorer(
          labels_test, pred_test, label)

  result_dict["Accuracy"] = np.mean(pred_test == labels_test)
  result_dict["F1"] = f1_score(labels_test, pred_test, average="macro")
  result_dict["Recall"] = recall_score(labels_test, pred_test, average="macro")
  result_dict["Precision"] = precision_score(
      labels_test, pred_test, average="macro")
  return result_dict
