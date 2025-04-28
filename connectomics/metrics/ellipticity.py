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
"""Ellipticity metric."""

from typing import Union

import numpy as np
import scipy.special


def _ellipticity_factor(num_dim: int) -> float:
  """Computes dimensionality-dependent factor required for metric."""
  return scipy.special.gamma(num_dim / 2 + 1) / ((num_dim + 2) * np.pi)**(
      num_dim / 2)


def compute_ellipticity(
    array: np.ndarray,
    label: Union[bool, float, int] = 1,
) -> float:
  """Calculates ellipticity of a labelled object.

  Without discretization, the measure would be constrained within (0, 1],
  equalling 1 iff the investigated set is a circle.

  While this metric is in principle applicable for arbitrary dimensionality,
  this implementation has only been tested for 2D and 3D cases.

  References:
    Misztal & Tabor, 2016: https://dx.doi.org/10.1007/s10851-015-0618-4

  Args:
    array: Array containing the object.
    label: Label identifying the object in the array.

  Returns:
    Ellipticity measure.
  """
  num_dim = array.ndim
  assert num_dim > 1
  coords = np.stack(np.where(array == label), axis=0)
  lambda_n = coords.shape[-1]
  sigma_n = np.cov(coords)
  return _ellipticity_factor(num_dim) * (
      lambda_n / np.sqrt(np.linalg.det(sigma_n)))
