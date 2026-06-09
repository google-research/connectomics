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
"""Baseline cross-modal alignment methods."""

import numpy as np
import sklearn.decomposition
import sklearn.metrics


class CrossModalAligner:
  """Base class for cross-modal aligners."""

  def transform_modality1(self, modality1_embeddings: np.ndarray) -> np.ndarray:
    """Projects modality 1 embeddings into a common space.

    Args:
      modality1_embeddings: num_cells x emb1_dim.

    Returns: num_cells x aligned_emb_dim.
    """
    raise NotImplementedError

  def transform_modality2(self, modality2_embeddings: np.ndarray) -> np.ndarray:
    """Projects modality 2 embeddings into a common space.

    Args:
      modality2_embeddings: num_cells x emb2_dim.

    Returns: num_cells x aligned_emb_dim.
    """
    raise NotImplementedError

  @classmethod
  def compute_cross_modal_distance(
      cls, modality1_relative_encodings: np.ndarray,
      modality2_relative_encodings: np.ndarray) -> np.ndarray:
    """Computes cross-modal distances between aligned embeddings.

    Args:
      modality1_relative_encodings: The aligned embeddings from modality 1.
      modality2_relative_encodings: The aligned embeddings from modality 2.

    Returns:
      A matrix of cross-modal distances between the two sets of embeddings.  In
      practice, this can include further normalization or sharpening as is
      useful.
    """
    raise NotImplementedError

  def backproject_to_modality1(
      self, relative_encodings: np.ndarray) -> np.ndarray:
    """Backprojects aligned embeddings to the original modality 1 space.

    Args:
      relative_encodings: The aligned embeddings from modality 2. This must be
        the output of `transform_modality2` without additional sharpening or
        normalization.

    Returns:
      A matrix of backprojected embeddings in the original modality 1 space.
    """
    raise NotImplementedError


def _zscore_and_sharpen(encodings: np.ndarray, exp: float) -> np.ndarray:
  encodings = encodings - np.mean(encodings, axis=0)[None, :]
  encodings /= np.std(encodings, axis=0)[None, :]
  return encodings**exp


class AsifAligner(CrossModalAligner):
  """Aligns two modalities using the ASIF method.

  Inspired by: http://arxiv.org/abs/2210.01738

  This method projects embeddings into a common space based on pairwise
  distances to anchor points derived from training data.  This simple,
  hand-crafted implementation sets a strong baseline for cross-modal alignment
  and reconstruction on patch-seq data.
  """

  def __init__(
      self, modality1_train_embeddings: np.ndarray,
      modality2_train_embeddings: np.ndarray,
      backproject_exp: float = 4.0):
    """Initializes and trains the alignment.

    Args:
      modality1_train_embeddings: num_train_cells x emb_dim_1.
      modality2_train_embeddings: num_train_cells x emb_dim_2.
      backproject_exp: sharpening exponential factor for weighted average
        backprojection.
    """
    assert (modality1_train_embeddings.shape[0] ==
            modality2_train_embeddings.shape[0])

    # Initial PCA on input embeddings.
    self.pca1 = sklearn.decomposition.PCA()
    self.pca2 = sklearn.decomposition.PCA()
    self.pca1.fit(modality1_train_embeddings)
    self.pca2.fit(modality2_train_embeddings)
    modality1_pcs = self.pca1.transform(modality1_train_embeddings)
    modality2_pcs = self.pca2.transform(modality2_train_embeddings)

    # Z-score to get things closer together.
    self.modality1_mean = np.mean(modality1_pcs, axis=0)[None, :]
    self.modality1_std = np.std(modality1_pcs, axis=0)[None, :]
    self.modality2_mean = np.mean(modality2_pcs, axis=0)[None, :]
    self.modality2_std = np.std(modality2_pcs, axis=0)[None, :]
    modality1_scaled = (
        modality1_pcs - self.modality1_mean) / self.modality1_std
    modality2_scaled = (
        modality2_pcs - self.modality2_mean) / self.modality2_std

    # Norm also helps if using Euclidean distance metric.
    modality1_norm = (
        modality1_scaled / np.linalg.norm(modality1_scaled, axis=1)[:, None])
    modality2_norm = (
        modality2_scaled / np.linalg.norm(modality2_scaled, axis=1)[:, None])
    self.modality1_anchors = modality1_norm
    self.modality2_anchors = modality2_norm

    self.modality1_train_embeddings_original = modality1_train_embeddings
    # Values from 4-10 seem to work well depending on modality.
    self.backproject_exp = backproject_exp

  def transform_modality1(self, modality1_embeddings: np.ndarray) -> np.ndarray:
    modality1_pcs = self.pca1.transform(modality1_embeddings)
    modality1_scaled = (
        modality1_pcs - self.modality1_mean) / self.modality1_std
    modality1_norm = (
        modality1_scaled / np.linalg.norm(modality1_scaled, axis=1)[:, None])
    return sklearn.metrics.pairwise.euclidean_distances(
        modality1_norm, self.modality1_anchors)

  def transform_modality2(self, modality2_embeddings: np.ndarray) -> np.ndarray:
    modality2_pcs = self.pca2.transform(modality2_embeddings)
    modality2_scaled = (
        modality2_pcs - self.modality2_mean) / self.modality2_std
    modality2_norm = (
        modality2_scaled / np.linalg.norm(modality2_scaled, axis=1)[:, None])
    return sklearn.metrics.pairwise.euclidean_distances(
        modality2_norm, self.modality2_anchors)

  @classmethod
  def compute_cross_modal_distance(
      cls, modality1_relative_encodings: np.ndarray,
      modality2_relative_encodings: np.ndarray, exp: float = 1) -> np.ndarray:
    modality1_encodings = _zscore_and_sharpen(modality1_relative_encodings, exp)
    modality2_encodings = _zscore_and_sharpen(modality2_relative_encodings, exp)
    return sklearn.metrics.pairwise.cosine_distances(
        modality1_encodings, modality2_encodings)

  def backproject_to_modality1(
      self, relative_encodings: np.ndarray) -> np.ndarray:
    # Convert distances to weights.
    backproject_weights = 1 / (relative_encodings**self.backproject_exp)
    backproject_weights /= backproject_weights.sum(axis=1)[:, None]

    # Filter out low weights and renormalize.
    thresh = 0.5
    max_weight_per_cell = backproject_weights.max(axis=1)
    for w, m in zip(backproject_weights, max_weight_per_cell):
      w[w < (m * thresh)] = 0
      w /= w.sum()

    return backproject_weights @ self.modality1_train_embeddings_original
