# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Binary ROC AUC regression coverage through collected metric states."""

import numpy as np
import pytest
from scipy import special
from sklearn import metrics as sklearn_metrics
from connectomics.jax import metrics


def evaluate(labels, logits):
  cls = metrics.create_classification_metrics(('negative', 'positive'))
  return cls.from_model_output(
      labels=np.asarray(labels), logits=np.asarray(logits)
  ).compute()


@pytest.mark.parametrize('sign, expected', [(1.0, 1.0), (-1.0, 0.0)])
def test_perfect_and_reversed_rankings(sign, expected):
  labels = np.array([0, 1, 0, 1])
  logits = sign * np.array([[8.0, -8.0], [-8.0, 8.0], [5.0, -5.0], [-5.0, 5.0]])
  assert evaluate(labels, logits)['roc_auc'] == expected


def test_auc_matches_independent_positive_negative_pair_count():
  labels = np.array([0, 1, 0, 1, 1, 0])
  score = np.array([0.2, 0.8, 0.4, 0.4, 0.3, 0.1])
  logits = np.stack([np.log1p(-score), np.log(score)], axis=-1)
  comparisons = [
      float(pos > neg) + 0.5 * float(pos == neg)
      for pos in score[labels == 1]
      for neg in score[labels == 0]
  ]
  actual = evaluate(labels, logits)
  np.testing.assert_allclose(actual['roc_auc'], np.mean(comparisons))
  predictions = logits.argmax(axis=-1)
  p, r, f, _ = sklearn_metrics.precision_recall_fscore_support(
      labels, predictions, labels=[0, 1]
  )
  for i, name in enumerate(['negative', 'positive']):
    assert actual['precision__' + name] == p[i]
    assert actual['recall__' + name] == r[i]
    assert actual['f1__' + name] == f[i]


def test_merging_batches_retains_global_binary_ranking():
  cls = metrics.create_classification_metrics(('negative', 'positive'))
  labels = np.array([0, 1, 0, 1])
  logits = np.array([[2.0, -2.0], [-3.0, 3.0], [1.0, -1.0], [-1.0, 1.0]])
  first = cls.from_model_output(labels=labels[:2], logits=logits[:2])
  second = cls.from_model_output(labels=labels[2:], logits=logits[2:])
  assert first.merge(second).compute()['roc_auc'] == 1.0
  assert (
      first.merge(second).compute()['roc_auc']
      == evaluate(labels, logits)['roc_auc']
  )


def test_swapping_classes_and_logits_preserves_discrimination():
  labels = np.array([0, 1, 1, 0, 1, 0])
  logits = np.array(
      [[2.0, 0.0], [0.0, 4.0], [0.0, 1.0], [2.0, 1.0], [3.0, 2.0], [0.0, 1.0]]
  )
  actual = evaluate(labels, logits)['roc_auc']
  expected = sklearn_metrics.roc_auc_score(
      labels, special.softmax(logits, axis=-1)[:, 1]
  )
  assert actual == expected
  assert evaluate(1 - labels, logits[:, ::-1])['roc_auc'] == expected


def test_tied_scores_remain_chance_level():
  assert evaluate([0, 1, 0, 1], np.zeros((4, 2)))['roc_auc'] == 0.5


def test_multiclass_auc_path_is_unchanged():
  cls = metrics.create_classification_metrics(('a', 'b', 'c'))
  labels = np.array([0, 1, 2, 0, 1, 2])
  logits = np.eye(3)[labels] * 3
  actual = cls.from_model_output(labels=labels, logits=logits).compute()[
      'roc_auc'
  ]
  expected = sklearn_metrics.roc_auc_score(
      labels, special.softmax(logits, axis=-1), multi_class='ovr'
  )
  assert actual == expected == 1.0
