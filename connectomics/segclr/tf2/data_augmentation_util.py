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
"""Augmentation module supporting data.py."""

import collections
from typing import Dict, Sequence, Any
import numpy as np
import tensorflow.compat.v2 as tf


def random_shift(features: Dict[str, Any],
                 max_spatial_shift_augment_vx: int) -> Dict[str, Any]:
  """Applies random shifts to all coordinate features.

  Keys of coordinate features start with 'center'.

  Args:
    features: features dict as eg. loaded from tf.Example
    max_spatial_shift_augment_vx: maximum shift per dimension in voxels

  Returns:
    features: augmented dictionary
  """
  for k in features:
    if k.startswith("center"):
      features[k] += (np.random.rand(3) * 2 - 1) * max_spatial_shift_augment_vx

  return features


@tf.function
def gray_augment_single(data_slice):
  """Helper function for augmentation gray scale values."""
  # Sample parameters for augmentation.
  # Tweaked on some EM samples from h01.
  alpha = 1 + tf.random.uniform(
      [1], minval=-0.25, maxval=0.25, dtype=tf.float32
  )
  c = tf.random.uniform([1], minval=-0.25, maxval=0.25, dtype=tf.float32)
  gamma = tf.pow(
      2.0, tf.random.uniform([1], minval=-1, maxval=1, dtype=tf.float32)
  )
  return tf.pow(tf.clip_by_value(data_slice * alpha + c, 0, 1), gamma)


def gray_augment(
    features: Dict[str, Any], num_pos_examples: int = 2
) -> Dict[str, Any]:
  """Gray value (brightness, contrast) augmentation.

  Args:
    features: features dict as eg. loaded from tf.Example
    num_pos_examples: number of corresponding patches for the same entity

  Returns:
    Transformed data dictionary

  """
  for k, v in features.items():
    if not k.startswith("channel_"):
      continue

    if "em" not in v:
      continue

    # Temporary storage as indexed assignments is not supported by tf.
    features_em = []
    for i_sample in range(num_pos_examples):
      features_em.append(gray_augment_single(v["em"][i_sample]))

    v["em"] = features_em

  return features


def random_flip(features: Dict[str, Any],
                data_keys: Sequence[Sequence[str]],
                num_pos_examples: int = 2) -> Dict[str, Any]:
  """Flipping augmentation.

  Args:
    features: features dict as eg. loaded from tf.Example
    data_keys: keys the augmentation is applied to. Data needs to be of the
      shape [1, x, y, z, ch].
    num_pos_examples: number of corresponding patches for the same entity

  Returns:
    Transformed data dictionary

  """
  # Temporary storage as indexed assignments is not supported by tf.
  features_t = collections.defaultdict(lambda: collections.defaultdict(list))

  for i_sample in range(num_pos_examples):
    # Randomly select axis for flipping.
    flip_axis = np.where(np.random.rand(3) > .5)[0]

    for i_channel, data_key in enumerate(data_keys):
      for k in data_key:
        channel_k = f"channel_{i_channel}"
        aug_sample = tf.reverse(features[channel_k][k][i_sample],
                                axis=flip_axis)
        features_t[channel_k][k].append(aug_sample)

  features.update(features_t)

  return features


def random_permutations(features: Dict[str, Any],
                        data_keys: Sequence[str],
                        num_pos_examples: int = 2) -> Dict[str, Any]:
  """Permutation augmentation.

  Args:
    features: features dict as eg. loaded from tf.Example
    data_keys: keys the augmentation is applied to. Data needs to be of the
      shape [1, x, y, z, ch].
    num_pos_examples: number of corresponding patches for the same entity

  Returns:
    Transformed data dictionary
  """
  # Temporary storage as indexed assignments is not supported by tf.
  features_t = collections.defaultdict(lambda: collections.defaultdict(list))

  for i_sample in range(num_pos_examples):
    # Sample permutations.
    rand_permutations = np.array([np.arange(5)] * 5)
    rand_permutations[:, 1:4] = np.argsort(np.random.rand(5, 3), axis=1) + 1

    for i_channel, data_key in enumerate(data_keys):
      for k in data_key:
        channel_k = f"channel_{i_channel}"
        aug_sample = features[channel_k][k][i_sample]

        for perm in rand_permutations:
          aug_sample = tf.transpose(aug_sample, perm=perm)

        features_t[channel_k][k].append(aug_sample)

  features.update(features_t)

  return features
