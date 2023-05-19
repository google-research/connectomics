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
"""Data input module supporting data.py."""

from typing import Sequence, Any
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile


def expand_pathspec(dataset_path: str) -> list[str]:
  """Helper function for gathering all filenames.

  Args:
    dataset_path: path or regex to precomputed data
      see also: selfsupervised/preprocessing/

  Returns:
    filenames: list of all filenames

  Raises:
    FileNotFoundError: If dataset_path points to a file that does not exist. Or
    if dataset_path regex leads to no results.
  """
  filenames = None
  if "*" in dataset_path:
    filenames = gfile.glob(dataset_path)
  elif "@" in dataset_path:
    filenames = gfile.glob(dataset_path.replace("@", "*"))
  elif gfile.exists(dataset_path):
    filenames = [dataset_path]
  if not filenames:
    raise FileNotFoundError(
        f"Invalid dataset path. No files found: {dataset_path}")
  return filenames


def discover_feature_dict(filenames: Sequence[str]) -> dict[str, Any]:
  """Assembles consensus feature dict from the filenames.

  This is an attempt to relieve the user from the burden to specify the correct
  feature_dict. There are slight differences in the feature_dicts as they are
  created throughout the project. This function attempts to map them to keys
  that work for this package.

  Args:
    filenames: paths to files on disk (not to a sharded file)

  Returns:
    feature_dict: synthesized feature_dict
  """

  feature_lookup_dict = {
      "node_id": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
      "skeleton_id": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
      "skeleton_id-0": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
      "skeleton_id-1": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
      "center": tf.io.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
      "center-0": tf.io.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
      "center-1": tf.io.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
      "vol_name": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
      "label_volume_name": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
      "distance": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
      "dist_to_border-0": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
      "dist_to_border-1": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
      "class_label": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
      "is_train": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
  }

  likely_keys_collection = []
  for filename in filenames:
    dataset = tf.data.TFRecordDataset(filename)
    for record in dataset.take(1):
      feature = tf.train.Example.FromString(record.numpy()).features.feature
    del dataset
    likely_keys_collection.extend(feature.keys())

  likely_keys_unique, likely_keys_counts = np.unique(
      likely_keys_collection, return_counts=True)
  likely_keys = likely_keys_unique[likely_keys_counts == len(filenames)]

  feature_dict = {}
  for feature_key in likely_keys:
    if feature_key in feature_lookup_dict:
      feature_dict[feature_key] = feature_lookup_dict[feature_key]

  return feature_dict
