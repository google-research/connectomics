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
"""Defines generic ResNet model with SNGP addition.

This module was based on https://www.tensorflow.org/tutorials/understanding/sngp
"""

from typing import Optional, Any
import edward2 as ed
import numpy as np
import tensorflow as tf


class DeepResNet(tf.keras.Model):
  """Defines a multi-layer residual network.

  The ResNet modules follow the RN-v2 design with full pre-activation
  """

  def __init__(self,
               num_classes: int,
               num_layers: int = 4,
               num_hidden: int = 32,
               dropout_rate: float = 0.0,
               l2_reg: Optional[float] = None,
               use_bn: bool = False,
               **gp_kwargs):
    """Initializer.

    Args:
      num_classes: number of output classes
      num_layers: number of ResNet modules
      num_hidden: number of hidden units per ResNet Module
      dropout_rate: Dropout rate; no dropout is applied when set to zero. Never
        applied to output layer.
      l2_reg: per layer L2 regulizarization weight. Applies to
        kernel_regularizer in keras layers.Dense
      use_bn: If true, BatchNorm is applied to all layers. Never applied to
        output layer.
      **gp_kwargs: Keyword arguments for the GP module. See the definition of
        ed.layers.RandomFeatureGaussianProcess for more details. See the code
        below for defaults.
    """
    super().__init__()

    # Defines class meta data.
    self.num_hidden = num_hidden
    self.num_layers = num_layers
    self.dropout_rate = dropout_rate
    self.use_bn = use_bn
    self.l2_reg = l2_reg
    self.gp_kwargs = gp_kwargs

    # Defines the hidden layers.
    self.input_layer_train = tf.keras.layers.Dense(
        self.num_hidden, trainable=True)
    self.input_layer_notrain = tf.keras.layers.Dense(
        self.num_hidden, trainable=False)
    self.dense_layers_1 = [
        self.make_dense_layer(activation=None) for _ in range(num_layers)  # pytype: disable=wrong-arg-types  # always-use-return-annotations
    ]
    self.bns_1 = [
        tf.keras.layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, trainable=True)
        for _ in range(num_layers)
    ]
    self.dense_layers_2 = [
        self.make_dense_layer(activation=None) for _ in range(num_layers)  # pytype: disable=wrong-arg-types  # always-use-return-annotations
    ]
    self.bns_2 = [
        tf.keras.layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, trainable=True)
        for _ in range(num_layers)
    ]

    # Defines the output layer. Includes gaussian process if selected
    self.classifier = self.make_output_layer(num_classes)

  def call(
      self, inputs: tf.Tensor, training: bool = True, mask=None
  ) -> tf.Tensor:
    """Computes the resnet hidden representations.

    Args:
      inputs: input data to the network
      training: set to true if training the network.
      mask: not implemented.

    Returns:
      network output
    """
    if mask is not None:
      raise NotImplementedError

    if self.dropout_rate > 0:
      x = tf.keras.layers.Dropout(self.dropout_rate)(inputs, training=training)

    if self.num_layers > 0:
      hidden = self.input_layer_train(inputs)

      # RN block.This follows the RN-v2 design with full pre-activation
      for i in range(self.num_layers):
        # Preactivation
        x = tf.identity(hidden)

        # Dropout and BatchNorm
        if self.dropout_rate > 0:
          x = tf.keras.layers.Dropout(self.dropout_rate)(x, training=training)
        if self.use_bn:
          x = self.bns_1[i](x, training=training)

        # Non-linearity 1 and dense layer 1
        x = tf.keras.layers.ReLU()(x)
        x = self.dense_layers_1[i](x)

        # Dropout and BatchNorm
        if self.dropout_rate > 0:
          x = tf.keras.layers.Dropout(self.dropout_rate)(x, training=training)
        if self.use_bn:
          x = self.bns_2[i](x, training=training)

        # Non-linearity 2 and dense layer 2
        x = tf.keras.layers.ReLU()(x)
        x = self.dense_layers_2[i](x)
        hidden = x + hidden
    else:
      # Pass through for "linear" models
      hidden = inputs

    return self.classifier(hidden)

  def make_dense_layer(self, activation: str = "relu") -> tf.Tensor:
    """Uses the Dense layer as the hidden layer."""
    return tf.keras.layers.Dense(
        self.num_hidden,
        kernel_initializer=None,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        if self.l2_reg else None)

  def make_output_layer(self, num_classes: int) -> tf.Tensor:
    """Uses the Dense layer as the output layer."""
    return tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        if self.l2_reg else None)

  def probas(self, inputs: tf.Tensor) -> tf.Tensor:
    """Applies softmax."""
    logits = self.call(inputs, training=False)
    return tf.nn.softmax(logits, axis=-1)


class ResetCovarianceCallback(tf.keras.callbacks.Callback):
  """Class to insert covariance reset callbacks.

  This is necessary for training the GP net.
  """

  def on_epoch_begin(self, epoch: int, logs: Any = None):
    """Resets covariance matrix at the beginning of the epoch."""
    if epoch > 0:
      self.model.classifier.reset_covariance_matrix()


class DeepResNetSN(DeepResNet):
  """Defines a multi-layer residual network with SNGP layers."""

  def __init__(self, spec_norm_bound: float = 6., **kwargs):
    """Initializer.

    Args:
      spec_norm_bound: Spectral norm bound. See norm_multiplier in
        ed.layers.SpectralNormalization for more details.
      **kwargs: pass through
    """
    self.spec_norm_bound = spec_norm_bound
    super().__init__(**kwargs)

  def make_dense_layer(self,
                       activation: str = "relu",
                       spec_norm_bound: Optional[float] = None
                      ) -> tf.Tensor:
    """Applies spectral normalization to the hidden layer."""

    if spec_norm_bound is None:
      spec_norm_bound = self.spec_norm_bound

    dense_layer = super().make_dense_layer(activation=activation)

    return ed.layers.SpectralNormalization(
        dense_layer, norm_multiplier=spec_norm_bound, iteration=1)


class DeepResNetGP(DeepResNet):
  """Defines a multi-layer residual network with SNGP layers."""

  def __init__(self, spec_norm_bound: float = 6., **kwargs):
    """Initializer.

    Args:
      spec_norm_bound: Spectral norm bound. See norm_multiplier in
        ed.layers.SpectralNormalization for more details.
      **kwargs: pass through
    """

    self.spec_norm_bound = spec_norm_bound
    super().__init__(**kwargs)

  def make_output_layer(
      self, num_classes: int) -> tf.Tensor:
    """Uses Gaussian process as the output layer."""
    kwargs = {
        "use_custom_random_features": True,
        "return_random_features": False,
        "scale_random_features": True,
        "gp_kernel_scale": 2.,
        "gp_cov_momentum": -1,
        "gp_cov_ridge_penalty": 1.,
        "l2_regularization": 0.,
        "normalize_input": True,
        "num_inducing": 1024
    }

    kwargs.update(self.gp_kwargs)

    return ed.layers.RandomFeatureGaussianProcess(num_classes, **kwargs)

  def call(self,
           inputs: tf.Tensor,
           training: bool = True,
           mask=None,
           return_covmat: bool = False) -> tf.Tensor:
    """Gets logits and covariance matrix from GP layer."""
    if mask is not None:
      raise NotImplementedError
    logits, _ = super().call(inputs, training=training)
    return logits

  def predict(self, inputs: tf.Tensor) -> tf.Tensor:
    """Short for call without training."""
    return super().call(inputs, training=False)

  def variance(self, inputs: tf.Tensor) -> tf.Tensor:
    """Computes variances."""
    _, covmat = super().call(inputs, training=False)
    return tf.linalg.diag_part(covmat)

  def adjusted_logits(self,
                      inputs: tf.Tensor,
                      lambda_param: float = np.pi / 8.) -> tf.Tensor:
    """Computes variance adjusted logits."""

    logits, covmat = super().call(inputs, training=False)

    variance = tf.linalg.diag_part(covmat)[:, None]
    return logits / tf.sqrt(1. + lambda_param * variance)

  def adjusted_probas(self,
                      inputs: tf.Tensor,
                      lambda_param: float = np.pi / 8.) -> tf.Tensor:
    """Computes adjusted variance probabilities."""
    return tf.nn.softmax(
        self.adjusted_logits(inputs, lambda_param=lambda_param), axis=-1)


class DeepResNetSNGP(DeepResNetGP):
  """Defines a multi-layer residual network with SNGP layers."""

  def __init__(self, spec_norm_bound: float = 6., **kwargs):
    """Initializer.

    Args:
      spec_norm_bound: Spectral norm bound. See norm_multiplier in
        ed.layers.SpectralNormalization for more details.
      **kwargs: pass through
    """
    self.spec_norm_bound = spec_norm_bound
    super().__init__(**kwargs)

  def make_dense_layer(self,
                       activation: str = "relu",
                       spec_norm_bound: Optional[float] = None) -> tf.Tensor:
    """Applies spectral normalization to the hidden layer."""

    if spec_norm_bound is None:
      spec_norm_bound = self.spec_norm_bound

    dense_layer = super().make_dense_layer(activation=activation)

    return ed.layers.SpectralNormalization(
        dense_layer, norm_multiplier=spec_norm_bound, iteration=1)


class DeepResNetGPWithCovReset(DeepResNetGP):
  """Class to insert covariance reset callbacks + GP."""

  def fit(self, *args: Any, **kwargs: Any) -> Any:
    """Adds ResetCovarianceCallback to model callbacks."""
    callbacks = list(kwargs.get("callbacks", []))
    callbacks.append(ResetCovarianceCallback())
    kwargs["callbacks"] = callbacks

    return super().fit(*args, **kwargs)


class DeepResNetSNGPWithCovReset(DeepResNetSNGP):
  """Class to insert covariance reset callbacks + SNGP."""

  def fit(self, *args: Any, **kwargs: Any) -> Any:
    """Adds ResetCovarianceCallback to model callbacks."""
    callbacks = list(kwargs.get("callbacks", []))
    callbacks.append(ResetCovarianceCallback())
    kwargs["callbacks"] = callbacks

    return super().fit(*args, **kwargs)
