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
"""Module for running an evaluation."""

import collections
import json
from typing import Any, Dict, List, Optional

from absl import logging
from connectomics.segclr import encoders
from connectomics.segclr.classification import classifier_utils
from connectomics.segclr.classification import evaluation_utils
from connectomics.segclr.classification import model_configs
from connectomics.segclr.classification import model_handler
import numpy as np
from tensorflow.io import gfile


def summarize_result_dicts(
    result_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Computes summary statistics for results.

  Args:
    result_dicts: list of dictionary of results

  Returns:
    summary_dict: Summarized results
  """

  summary_dict = collections.defaultdict(dict)
  for k in result_dicts[0]:
    results = [result_dicts[i_dict][k] for i_dict in range(len(result_dicts))]
    summary_dict["max"][k] = np.max(results, axis=0)
    summary_dict["min"][k] = np.min(results, axis=0)
    summary_dict["mean"][k] = np.mean(results, axis=0)
    summary_dict["std"][k] = np.std(results, axis=0)
    summary_dict["n_repeats"][k] = len(results)
  return dict(summary_dict)


def save_predictions(model_dir: str,
                     variances: np.ndarray,
                     logits: np.ndarray,
                     labels: np.ndarray,
                     test_idx: np.ndarray,
                     add_test_data: Optional[Dict[str, np.ndarray]] = None):
  """Saves test predictions for a trained model.

  Args:
    model_dir: directory where model is stored.
    variances: numpy array of length N of variances for each prediction
    logits: N x c numpy array of the predicted logits for each element and class
    labels: numpy array of length N of ground truth labels for each prediction
    test_idx: numpy array of length M indexing into an array of length N to
      resample the test dataset
    add_test_data: dictionary of numpy arrays of length N storing additional
      data. The keys are used for naming the files like so: test_{data_key}.npy
  """
  with gfile.GFile(f"{model_dir}/test_variances.npy", "wb") as f:
    np.save(f, variances)

  with gfile.GFile(f"{model_dir}/test_logits.npy", "wb") as f:
    np.save(f, logits)

  with gfile.GFile(f"{model_dir}/test_labels.npy", "wb") as f:
    np.save(f, labels)

  with gfile.GFile(f"{model_dir}/test_idx.npy", "wb") as f:
    np.save(f, test_idx)

  if add_test_data is not None:
    for data_key in add_test_data:
      with gfile.GFile(f"{model_dir}/test_{data_key}.npy", "wb") as f:
        np.save(f, add_test_data[data_key])


def evaluate_train_test(labels_train: np.ndarray,
                        labels_test: np.ndarray,
                        embeddings_train: np.ndarray,
                        embeddings_test: np.ndarray,
                        out_path: str,
                        model_name: str,
                        model_id_str: str,
                        standardization_mean: np.ndarray,
                        standardization_std: np.ndarray,
                        emb_model_strs: Optional[List[str]] = None,
                        upsample_train: bool = False,
                        upsample_test: bool = False,
                        add_test_data: Optional[Dict[str, np.ndarray]] = None,
                        n_random_states: int = 3,
                        run_id: int = 0,
                        var_scale: float = 3 / np.pi**2) -> Dict[str, Any]:
  """Trains and evaluates models for a single (sampled) dataset.

  Args:
    labels_train: numpy array of length N with the ground truth labels for the
      training data
    labels_test: numpy array of length N with the ground truth labels for the
      test data
    embeddings_train: N x d numpy array with the embeddings with feature
      dimension d for the training data
    embeddings_test: N x d numpy array with the embeddings with feature
      dimension d for the test data
    out_path: path where to store the model
    model_name: name of the model. See ../classification/model_configs for
      possible model names
    model_id_str: unique identifier for the model in this run. This is
      required because parallel models are being trained and stored in the same
      folder.
    standardization_mean: numpy array of length equal to the number of embedding
      dimensions containin the original mean values of each dimension
    standardization_std: numpy array of length equal to the number of embedding
      dimensions containin the original std values of each dimension
    emb_model_strs: list of embedding model definitions
    upsample_train: Upsamples training data to the same number of items per
      class
    upsample_test: Upsamples test data to the same number of items per class
    add_test_data: dictionary of numpy arrays of length N storing additional
      data. The keys are used for naming the files like so: test_{data_key}.npy
    n_random_states: The number of randomizations to do for this evaluation run
    run_id: The ID of this evaluation run. The ID is used for creating storage
      paths
    var_scale: Scaling parameter for variances. Also referred to as lambda. This
      parameter is usually chosen to be either 3/np.pi**2 or np.pi / 8.

  Returns:
    result_dict: dictionary of results for each randomly initialized evaluation.
  """
  if emb_model_strs is None:
    emb_model_strs = []

  result_dict = {}
  result_dict["n_data"] = len(embeddings_train)
  result_dict["n_data_per_class_train"] = list(
      zip(*np.unique(labels_train, return_counts=True)))
  result_dict["n_data_per_class_test"] = list(
      zip(*np.unique(labels_test, return_counts=True)))
  result_dict["upsample_train"] = upsample_train
  result_dict["upsample_test"] = upsample_test

  if upsample_train:
    train_idx = classifier_utils.upsample_labels(labels_train)
    embeddings_train = embeddings_train[train_idx]
    labels_train = labels_train[train_idx]

  if upsample_test:
    test_idx = classifier_utils.upsample_labels(labels_test)
  else:
    test_idx = np.arange(len(labels_test))

  result_dicts_rand_states = []
  for random_state in range(n_random_states):
    if not model_id_str:
      model_dir = f"{out_path}_I{run_id}_R{random_state}/"
    else:
      model_dir = f"{out_path}_I{run_id}_R{random_state}/{model_id_str}/"
    gfile.makedirs(model_dir)

    model_config = model_configs.__dict__[model_name]

    learning_rate = 1e-3
    if model_config["batch_size"] >= 1024:
      learning_rate = 1e-2
    if model_config["batch_size"] >= 2048:
      learning_rate = 1e-1

    train_config = {"learning_rate": learning_rate}

    logging.info("Train config: %s", train_config)
    train_history, model_config, train_config, model = (
        model_handler.train_model(
            model_config=model_config,
            train_data=embeddings_train,
            train_labels=labels_train,
            valid_split=0.5,
            **train_config,
        )
    )

    model_handler.save_model(
        model,
        model_config,
        model_dir,
        train_history=train_history,
        train_config=train_config,
        standardization_mean=standardization_mean,
        standardization_std=standardization_std,
        emb_model_strs=emb_model_strs,
        var_scale=var_scale)

    variances_test, probas_test, logits_test = model_handler.predict_data(
        embeddings_test, model)

    save_predictions(
        model_dir,
        variances_test,
        logits_test,
        labels_test,
        test_idx,
        add_test_data=add_test_data)

    pred_test = np.argmax(probas_test, axis=1)
    result_dicts_rand_states.append(
        evaluation_utils.evaluate_prediction(labels_test[test_idx],
                                             pred_test[test_idx]))

  for k in result_dicts_rand_states[0]:
    result_dict[k] = np.mean([r[k] for r in result_dicts_rand_states])

  with gfile.GFile(f"{model_dir}/results.json", "w") as f:
    f.write(json.dumps(result_dict, cls=encoders.NumpyEncoder))

  return result_dict


def compute_evaluations(
    data_dict: Dict[str, Any],
    n_samples: int,
    n_runs: int,
    guarantee_n: int,
    out_path: str,
    model_name: str,
    model_id_str: str,
    standardization_mean: np.ndarray,
    standardization_std: np.ndarray,
    emb_model_strs: List[str],
) -> Dict[str, Any]:
  """Manages evaluation.

  Args:
    data_dict: data as returned by loading_utils.build_datasets
    n_samples: number of samples to draw from the training dataset
    n_runs: number of evaluation runs to initialize. In each run, a new
      subsampled training set is drawn.
    guarantee_n: number of samples to be guaranteed to come from each class. If
      guarantee_n exceeds the number of unique samples from a given class, it is
      ignored.
    out_path: Path to where to store results and trained models.
    model_name: name of the model. See ../classification/model_configs for
      possible model names
    model_id_str: unique identifier for the model in this in run. This is
      required because parallel models are beiing trained and stored in the same
      folder.
    standardization_mean: numpy array of length equal to the number of embedding
      dimensions containin the original mean values of each dimension
    standardization_std: numpy array of length equal to the number of embedding
      dimensions containin the original std values of each dimension
    emb_model_strs: list of embedding model definitions

  Returns:
    summarized_dict: Dictionary with the combined results from all runs.
  """
  result_dicts = []

  for i_run in range(n_runs):
    ids = evaluation_utils.sample_n(
        data_dict["train"]["labels"],
        n_samples,
        guarantee_n=guarantee_n,
        random_seed=i_run)

    train_labels = data_dict["train"]["labels"][ids]
    train_embeddings = data_dict["train"]["embeddings"][ids]

    result_dict = evaluate_train_test(
        train_labels,
        data_dict["test"]["labels"],
        train_embeddings,
        data_dict["test"]["embeddings"],
        model_id_str=model_id_str,
        standardization_mean=standardization_mean,
        standardization_std=standardization_std,
        out_path=out_path,
        model_name=model_name,
        emb_model_strs=emb_model_strs,
        run_id=i_run)
    n_data = len(train_labels)
    result_dicts.append(result_dict)

  summarized_dict = summarize_result_dicts(result_dicts)
  summarized_dict["n_data"] = n_data
  return summarized_dict
