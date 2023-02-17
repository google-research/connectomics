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
"""Handles training and model storage for model_def."""

import json
from typing import Any, Dict, List, Optional, Tuple

from connectomics.segclr import encoders
from connectomics.segclr.classification import model_def
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile


def train_model(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    train_weights: Optional[np.ndarray] = None,
    valid_data: Optional[np.ndarray] = None,
    valid_labels: Optional[np.ndarray] = None,
    valid_weights: Optional[np.ndarray] = None,
    valid_split: float = 0.1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    verbose: bool = False,
    patience: int = 20,
    training_epochs: int = 10000,
    balance_labels: bool = False,
    random_seed: int = 42,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], tf.keras.Model]:
  """Creates and trains a model.

  The input parameters determine what model is created.

  Args:
    train_data: N x d data array with N elements and d feature dimensions
    train_labels: N data array with integer labels
    train_weights: N data array with float weights. Weights will be normalized.
      If no weights are given, all elements are given equal weight.
    valid_data: Optional. Same as train_data
    valid_labels: Optional but needs to be given if valid_data is not None. Same
      as train_labels.
    valid_weights: Optional. Same as train_weights. Has no effect if no
      valid_data is given.
    valid_split: If no valid_data is given, a part of the training data can be
      used for validation. If no valid_data is given and valid_split==0 all
      training data is used for validation (required for early stopping).
    batch_size: batch size for training
    learning_rate: learning rate for training. Training uses the Adam optimizer.
      See tf.keras.optimizers.Adam for more details for how the learning rate is
      used.
    verbose: prints all training steps and validation results.
    patience: how many steps to wait for early stopping to stop the training.
      The trained model always uses the best (lowest validation score) training
      epoch.
    training_epochs: maximum number of training epochs.
    balance_labels: if True, weights will be computed such that the data is
      balanced. Ignored if train_weights / valid_weights is defined.
    random_seed: The training data is shuffled before passed to the training
      using this seed.
    model_config: See the model_configs.py module for examples. See
      model_def.DeepResNet for documentation.

  Returns:
    train_dat: Training history data
    model_config: The final model config
    train_config: The training config
    model: The trained model.
  """
  train_config = {
      "valid_split": valid_split,
      "batch_size": batch_size,
      "learning_rate": learning_rate,
      "patience": patience,
      "training_epochs": training_epochs,
      "balance_labels": balance_labels,
      "random_seed": random_seed,
  }

  # Define training parameters
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  compile_config = dict(
      loss=loss, metrics=["SparseCategoricalAccuracy"], optimizer=optimizer)
  early_stop_callback = tf.keras.callbacks.EarlyStopping(
      monitor="val_loss", patience=patience, restore_best_weights=True)
  fit_config = dict(
      batch_size=batch_size,
      epochs=training_epochs,
      verbose=verbose,
      shuffle=True,
      validation_split=valid_split,
      callbacks=[early_stop_callback])

  if model_config is None:
    model_config = {}

  model_config["num_classes"] = np.max(train_labels) + 1

  use_gp = model_config.get("num_inducing", 1024) > 0
  use_sn = model_config.get("spec_norm_bound", 6.) > 0

  if use_gp or use_sn:
    if use_gp and use_sn:
      model = model_def.DeepResNetSNGPWithCovReset(**model_config)
    elif use_sn:
      model = model_def.DeepResNetSN(**model_config)
    else:
      model = model_def.DeepResNetGPWithCovReset(**model_config)
  else:
    model = model_def.DeepResNet(**model_config)

  model.compile(**compile_config, run_eagerly=True)

  # Training data
  train_labels = np.asarray(train_labels)
  train_data = np.asarray(train_data)

  if train_weights is None:
    if balance_labels:
      train_weights = np.ones(len(train_labels))
      for label, label_count in zip(
          *np.unique(train_labels, return_counts=True)):
        train_weights[train_labels == label] = 1. / label_count
    else:
      train_weights = np.ones(len(train_labels))

  # Defines validation data
  if valid_data is not None and valid_labels is not None:
    if valid_weights is None:
      if balance_labels:
        valid_weights = np.ones(len(valid_labels))
        for label, label_count in zip(
            *np.unique(valid_labels, return_counts=True)):
          valid_weights[valid_labels == label] = 1. / label_count
      else:
        valid_weights = np.ones(len(valid_data))

    fit_config["validation_data"] = (valid_data, valid_labels,
                                     valid_weights / np.mean(valid_weights))
  elif valid_split == 0:
    # Validate on the entire training set if no validation is given and
    # validation split was set to 0
    fit_config["validation_data"] = (train_data, train_labels,
                                     train_weights / np.mean(train_weights))

  random_state = np.random.RandomState(random_seed)
  train_order = np.arange(len(train_data))
  random_state.shuffle(train_order)

  # Training
  train_history = model.fit(
      train_data[train_order],
      train_labels[train_order],
      **fit_config,
      sample_weight=train_weights[train_order] / np.mean(train_weights))

  return train_history, model_config, train_config, model


def predict_data(
    data: np.ndarray,
    model: tf.keras.Model,
    block_size: int = 50000,
    standardization_mean: float = 0.,
    standardization_std: float = 1.
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Runs prediction on data array.

  Args:
    data: N x d data array with N elements and d feature dimensions
    model: Trained model from model_def.
    block_size: Defines how many elements are predicted at once. Use ~50000 to
      not exceed local memory limits.
    standardization_mean: Standardizes the data to mean=0 with this value.
    standardization_std: Standardizes the data to std=1 with this value.

  Returns:
    variances: GP variances. Empty if not using GP
    probas: classifier probabilities (not adjusted)
    logits: classifier logits
  """

  predict_variances = "GP" in model.__class__.__name__

  probas = []
  logits = []
  variances = []

  data = data.astype(np.float32).copy()
  data -= standardization_mean
  data /= standardization_std

  for data_s in np.array_split(data, (len(data) // block_size) + 1):
    probas.extend(model.probas(data_s).numpy())
    logits.extend(model(data_s).numpy())

    if predict_variances:
      variances.extend(model.variance(data_s).numpy())

  variances = np.array(variances)
  probas = np.array(probas)
  logits = np.array(logits)
  return variances, probas, logits


def save_model(model: tf.keras.Model,
               model_config: Dict[str, Any],
               out_dir: str,
               train_history: Optional[tf.keras.callbacks.History] = None,
               train_config: Optional[Dict[str, Any]] = None,
               standardization_mean: Optional[np.ndarray] = None,
               standardization_std: Optional[np.ndarray] = None,
               emb_model_strs: Optional[List[str]] = None,
               var_scale: Optional[float] = None):
  """Saves a model and additional parameters.

  Args:
    model: Trained model from model_def.
    model_config: model configuration returned by `train_model`,
    out_dir: directory in which the model and its parameters will be stored.
    train_history: training history as compiled by keras
    train_config: The training config
    standardization_mean: numpy array of length equal to the number of embedding
      dimensions containin the original mean values of each dimension
    standardization_std: numpy array of length equal to the number of embedding
      dimensions containin the original std values of each dimension
    emb_model_strs: list of embedding model definitions
    var_scale: Scaling parameter for variances. Also referred to as lambda. This
      parameter is usually chosen to be either 3/np.pi**2 or np.pi / 8.
  """
  gfile.makedirs(out_dir)
  model.save_weights(f"{out_dir}/model_weights")

  with gfile.GFile(f"{out_dir}/model_output_weights.npy", "wb") as f:
    np.save(f, model.classifier.weights[0].numpy())

  configs = {
      "train_config": train_config,
      "model_config": model_config,
      "add_config": {
          "standardization_mean": standardization_mean,
          "standardization_std": standardization_std,
          "sngp_lambda": var_scale,
          "model_class_name": model.__class__.__name__,
          "emb_model_strs": emb_model_strs,
      }
  }

  with gfile.GFile(f"{out_dir}/configs.json", "w") as f:
    json.dump(configs, f, cls=encoders.NumpyEncoder)

  if train_history is not None:
    keras_history = train_history.history
    keras_history["epoch"] = train_history.epoch
    with gfile.GFile(f"{out_dir}/train_history.json", "w") as f:
      json.dump(keras_history, f, cls=encoders.NumpyEncoder)


def load_model(out_dir: str) -> Tuple[tf.keras.Model, Dict[str, Any]]:
  """Loads a model.

  Args:
    out_dir: directory in which the model and its parameters are stored.

  Returns:
    model: the loaded model
    configs: dictionary containing training and model config
  """
  with gfile.GFile(f"{out_dir}/configs.json", "r") as f:
    configs = json.load(f)

  model_class = model_def.__dict__[configs["add_config"]["model_class_name"]]
  model = model_class(**configs["model_config"])
  model.load_weights(f"{out_dir}/model_weights")

  return model, configs
