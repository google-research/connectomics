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
"""Module for uncertainty functionality."""

from typing import Tuple
import numpy as np


def calculate_dempster_shafer(logits: np.ndarray) -> np.ndarray:
  """Calculates Dempster Shafer Metric.

  Args:
    logits: N x c array with N elements and c classes. The (adjusted) logits
      from classifier.

  Returns:
    ds_unc: dempster shafer values.
  """
  n_classes = logits.shape[-1]
  ds_unc = n_classes / (n_classes + np.sum(np.exp(logits), axis=-1))

  if len(ds_unc.shape) == 2:
    ds_unc = np.mean(ds_unc, axis=0)

  return ds_unc


def adjust_logits(
    logits: np.ndarray,
    variances: np.ndarray,
    var_scale: float = 3 / np.pi**2) -> Tuple[np.ndarray, np.ndarray]:
  """Adjusts logits with variance estimates.

  Args:
    logits: logits from classifier.
    variances: variances produced by GP layer.
    var_scale: Scaling parameter for variances. Also referred to as lambda. This
      parameter is usually chosen to be either 3/np.pi**2 or np.pi / 8.

  Returns:
    adjusted probas: probabilities derived with softmax from adjusted logits
    adjusted logits: variance adjusted logits
  """
  adj_logits = logits / np.sqrt(1. + var_scale * variances[:, None])

  adj_logits_exp = np.exp(adj_logits)
  adj_probas = adj_logits_exp / np.sum(adj_logits_exp, axis=1)[:, None]
  return adj_probas, adj_logits
