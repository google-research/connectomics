# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Utilities for instantiating models."""

import collections.abc
import inspect
import json
import re
from typing import Any, Type

from absl import logging
from connectomics.common import file
from connectomics.common import import_util
# pylint:disable=unused-import
from connectomics.jax.models import convstack

import flax.linen as nn
import ml_collections

DEFAULT_PKG = 'connectomics.jax.models'


def class_from_name(
    model_class: str, default_packages: str = DEFAULT_PKG
) -> tuple[Type, Type]:  # pylint:disable=g-bare-generic
  model_cls = import_util.import_symbol(
      model_class, default_packages=default_packages
  )
  cfg_cls = (
      inspect.signature(model_cls.__init__).parameters['config'].annotation
  )
  return model_cls, cfg_cls


def get_config_name(config_cls_name: str) -> str:
  """Returns the default ConfigDict field name for a given model class name."""
  # The model is configured by a field, the name of which is the snake
  # case version of the config class.
  return re.sub(r'(?<!^)(?=[A-Z]([^A-Z]|$))', '_', config_cls_name).lower()


def model_from_config(
    config: ml_collections.ConfigDict,
    default_packages: str = DEFAULT_PKG,
) -> nn.Module:
  """Initializes a JAX model from settings in a ConfigDict.

  A typical use case is to instantiate a model for training based on
  settings that can be overridden from the command line.

  Args:
    config: ConfigDict containing a field with the settings for the model; the
      model is expected to be configured with a single dataclass stored in its
      '.config' attribute
    default_packages: module from which to import the model class

  Returns:
    flax model object
  """
  model_cls, cfg_cls = class_from_name(config.model_class, default_packages)
  cfg_field = get_config_name(cfg_cls.__name__)

  logging.info('Using config settings from "%r"', cfg_field)
  model_cfg = getattr(config, cfg_field)
  # By converting the config to a FrozenConfigDict, we ensure that it is
  # hashable. This is e.g. required for static arguments passed to jax.jit.
  model_cfg = ml_collections.config_dict.FrozenConfigDict(model_cfg)
  return model_cls(
      config=cfg_cls(**model_cfg), name=getattr(config, 'model_name', None)
  )


def model_from_name(
    model_class: str,
    model_name: str | None = None,
    default_packages: str = DEFAULT_PKG,
    **kwargs
) -> nn.Module:
  """Initializes a JAX model given a name and its config settings.

  A typical use case is to instantiate a model for inference based on
  settings recorded in a JSON object teogether with the experiment
  that was used to train the model.

  Args:
    model_class: name of the Python class implementing the model.
    model_name: name of the model parameters (passed to the constructor of the
      model class as `name` parameter)
    default_packages: module from which to import 'model_class'
    **kwargs: arguments passed to the configuration object for the model

  Returns:
    flax model object
  """
  model_cls = import_util.import_symbol(
      model_class, default_packages=default_packages
  )
  cfg_cls = (
      inspect.signature(model_cls.__init__).parameters['config'].annotation
  )

  # TODO(mjanusz): Figure out how to make this compatible with callable config
  # values.
  def _skip_arg(name, value, cls):
    """Detects settings which currently cannot be restored."""

    if isinstance(value, str) and (
        value.startswith('function ') or 'unserializable' in value
    ):
      return True

    if (
        getattr(
            inspect.signature(cls).parameters[name].annotation,
            '__origin__',
            None,
        )
        is collections.abc.Callable
    ):
      return True

    return False

  def _value(key, value, cls):
    val_type = inspect.signature(cls).parameters[key].annotation
    if hasattr(val_type, '__dataclass_fields__'):
      value = {
          k: _value(k, v, val_type)
          for k, v in value.items()
          if not _skip_arg(k, v, val_type)
      }
      return val_type(**value)
    else:
      return value

  kwargs = {
      k: _value(k, v, cfg_cls)
      for k, v in kwargs.items()
      if not _skip_arg(k, v, cfg_cls)
  }

  logging.info(
      'Initializing model %r with config %r(%r)', model_cls, cfg_cls, kwargs
  )
  return model_cls(config=cfg_cls(**kwargs), name=model_name)


def model_from_dict_config(
    config: dict[str, Any],
    default_packages: str = DEFAULT_PKG,
) -> nn.Module:
  """Initializes a JAX model from settings in a python dictionary.

  Like model_from_config, but uses a dictionary as configuration.

  Args:
    config: dictionary containing a field with the settings for the model; the
      model is expected to be configured with a single dataclass stored in its
      '.config' attribute
    default_packages: module from which to import the model class

  Returns:
    flax model object
  """

  _, cfg_cls = class_from_name(config['model_class'], default_packages)
  cfg_field = get_config_name(cfg_cls.__name__)
  return model_from_name(
      config['model_class'],
      config.get('model_name'),
      default_packages,
      **config[cfg_field],
  )


def save_config(config: ml_collections.ConfigDict, path: file.PathLike):
  """Saves model config to a file."""
  with file.Path(path).open('wt') as f:
    f.write(config.to_json_best_effort() + '\n')


def load_config(path: file.PathLike) -> ml_collections.ConfigDict:
  """Loads a model config from a file."""
  with file.Path(path).open('rt') as f:
    return ml_collections.ConfigDict(json.loads(f.read()))
