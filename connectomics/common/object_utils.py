# Copyright 2024 The Google Research Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for processing object data."""

from typing import Tuple
import numpy as np
from tf import transformations


def mask_to_points(mask: np.ndarray,
                   voxel_size: Tuple[float, float, float],
                   origin_in_center=True):
  """Converts an object mask to a dense point cloud.

  Args:
    mask: 3-5 dim boolean ndarray representing the object mask
    voxel_size: xyz voxel size in physical units
    origin_in_center: whether the origin of the coordinate system should
        be located in the center of `mask`

  Returns:
    [3, n] array with coordinates of the `True` voxels of mask;
    coordinates are expressed in physical units
  """
  if mask.ndim == 5:
    mask = mask.squeeze(axis=(0, 4))
  elif mask.ndim == 4:
    mask = mask.squeeze(axis=3)
  elif mask.ndim != 3:
    raise ValueError('mask needs to be a 3-5 dimensional array.')

  if origin_in_center:
    r = (np.array(mask.shape) - 1) / 2
  else:
    r = np.array([0, 0, 0])

  z, y, x = np.where(mask)
  return np.array([
      (x - r[2]) * voxel_size[0],  #
      (y - r[1]) * voxel_size[1],  #
      (z - r[0]) * voxel_size[2]
  ])


def compute_orientation(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  eigvals, eigvecs = np.linalg.eig(np.cov(points))
  sorted_idx = np.argsort(eigvals)[::-1]
  return eigvecs[:, sorted_idx], eigvals[sorted_idx]


def compute_rotation_matrix(eigvecs: np.ndarray) -> np.ndarray:
  """Computes a rotation matrix to put the object in a standard orientation.

  Args:
    eigvecs: two xyz vectors ([3, 2] ndarray) corresponding to the
      orthogonal directions of maximum variance, in descending order of
      variance

  Returns:
    3x3 rotation matrix which will reorient the coordinate system so that
    'z' is the direction of maximum variance, and 'y' is the axis of 2nd
    largest variance
  """
  # 1st rotation to make z the axis of maximum variance
  base = np.array([0, 0, 1])
  u = eigvecs[:, 0]
  axis = np.cross(u, base)
  theta = np.arccos(u.dot(base))

  # Cut to 3x3 since we don't need homegeneous coordinates.
  rot = transformations.rotation_matrix(theta, axis)[:3, :3]

  # 2nd rotation to align the 2nd direction of max variance with the y axis
  v = np.matmul(rot, eigvecs[:, 1])
  base = np.array([0, 1, 0])
  axis = np.cross(v, base)
  theta = np.arccos(v.dot(np.array([0, 1, 0])))
  rot2 = transformations.rotation_matrix(theta, axis)[:3, :3]

  # Compose the two rotations.
  return np.matmul(rot2, rot)
