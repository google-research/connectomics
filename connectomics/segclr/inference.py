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
"""Simple wrapper to load TF1 SegCLR embedding model to run inference.

The original SegCLR experiments were run using a TensorFlow 1 framework based on
tf.estimator.  We are in the process of updating to TensorFlow 2 and tf.data for
future / public use.
"""

from connectomics.segclr import model
import numpy as np
import tensorflow.compat.v1 as tf
# pylint: disable=g-deprecated-tf-checker
from tensorflow.compat.v1 import estimator as tf_estimator

tf.disable_eager_execution()


class EmbeddingModel:
  """Builds model from TF1 checkpoint and runs inference."""

  def __init__(self, ckpt_path, batch_size=1, eager_load=True):
    self.ckpt_path = ckpt_path
    self.input_data_shape = [batch_size, 129, 129, 129, 1]

    self.model_cache = {}
    tf.keras.backend.clear_session()
    if eager_load:
      self._build_model()

  def _build_model(self):
    """Builds model from TF1 checkpoint."""
    model_fn = model.model_gen(model_dir='', n_data_channels=1, proj_out_dim=16,
                               mode='pred')
    config_proto = tf.ConfigProto(
        log_device_placement=True, allow_soft_placement=True,
        inter_op_parallelism_threads=32)
    run_config = tf_estimator.tpu.RunConfig(session_config=config_proto)
    classifier = tf_estimator.tpu.TPUEstimator(
        model_fn=model_fn, config=run_config, train_batch_size=1,
        params={'model_class': 0, 'model_args': [], 'is_chief': False},
        use_tpu=False)

    feeds = {'data': tf.placeholder(tf.float32, shape=self.input_data_shape)}
    labels = None
    estimator_spec = classifier.model_fn(
        feeds.copy(), labels, tf_estimator.ModeKeys.PREDICT, classifier.config)
    fetch = estimator_spec.predictions

    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, self.ckpt_path)

    self.model_cache = {'feeds': feeds, 'fetch': fetch, 'session': session}

  def _get_model(self):
    if not self.model_cache:
      self._build_model()
    return self.model_cache

  def process(self, input_data):
    """Runs the model on the input_data.

    Args:
      input_data: A np.uint8 array, shape-compatible with self.input_data_shape.
        Input values are assumed to range [0, 255].

    Returns:
      np.float array of batch x L embedding vectors.
    """
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.reshape(input_data, self.input_data_shape)
    model_cache = self._get_model()
    result = model_cache['session'].run(
        model_cache['fetch'],
        feed_dict={model_cache['feeds']['data']: input_data})
    return result['embeddings']
