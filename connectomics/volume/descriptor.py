# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Implementation of VolumeDescriptor."""

import dataclasses
import functools
import typing
from typing import Optional, Union

from connectomics.common import file
from connectomics.volume import base
from connectomics.volume import decorator
from connectomics.volume import tensorstore as tsv
import dataclasses_json


@dataclasses_json.dataclass_json
@functools.partial(dataclasses.dataclass, frozen=True, eq=True)
class VolumeDescriptor:
  """De/Serializable description of a volume."""

  # List of python dicts of specs to decorate the volume via
  # decorator.from_specs. E.g.:
  #   '[{"decorator": "ZSub", "args": [{"2198": 2197}]}]'
  # If left unspecified, the undecorated volume is used.
  decorator_specs: list[decorator.DecoratorSpec] = dataclasses.field(
      default_factory=list)

  # Exactly one of `volumeinfo` or `tensorstore_config` must be specified

  # Path of tensorstore config to load, or an instantiated TensorstoreConfig
  # object.
  tensorstore_config: Optional[tsv.TensorstoreConfig] = dataclasses.field(
      default=None,
      metadata=dataclasses_json.config(
          decoder=file.dataclass_loader(tsv.TensorstoreConfig)))

  # Internal use only. Path to the VolumeInfo file.
  volinfo: Optional[str] = None


def load_descriptor(spec: Union[str, VolumeDescriptor]) -> VolumeDescriptor:
  return file.load_dataclass(VolumeDescriptor, spec)


def load_ts_config(
    spec: Union[str, tsv.TensorstoreConfig]) -> tsv.TensorstoreConfig:
  return file.load_dataclass(tsv.TensorstoreConfig, spec)


def open_descriptor(spec: Union[str, VolumeDescriptor]) -> base.BaseVolume:
  """Open a volume from a volume descriptor.

  Args:
    spec: A serialized or fully loaded instance of VolumeDescriptor, or a path
      to a JSON serialized VolumeDescriptor object.

  Returns:
    Loaded volume, including any volume decorators.

  Raises:
    ValueError: If volinfo is present, as it is internal-only.
  """

  spec = load_descriptor(spec)

  if spec.volinfo:
    raise ValueError('volinfo field not supported')

  config = load_ts_config(spec.tensorstore_config)

  config = typing.cast(tsv.TensorstoreConfig, config)
  volume = tsv.TensorstoreVolume(config)
  return decorator.from_specs(volume, spec.decorator_specs)
