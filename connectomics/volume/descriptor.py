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

import copy
import dataclasses
import typing
from typing import Any, Optional, Union

from connectomics.common import file
from connectomics.common import utils
from connectomics.volume import base
from connectomics.volume import tensorstore as tsv
from connectomics.volume import tsv_decorator
import dataclasses_json


@dataclasses.dataclass(frozen=True, eq=True)
class VolumeDescriptor(utils.NPDataClassJsonMixin):
  """De/Serializable description of a volume."""

  # List of python dicts of specs to decorate the volume via
  # decorator.from_specs. E.g.:
  #   '[{"decorator": "ZSub", "args": [{"2198": 2197}]}]'
  # If left unspecified, the undecorated volume is used.
  decorator_specs: list[tsv_decorator.DecoratorSpec] = dataclasses.field(
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
  desc = file.load_dataclass(VolumeDescriptor, spec)
  if desc is None:
    raise ValueError(f'Could not load descriptor: {spec}')
  return desc


def open_descriptor(
    spec: Union[str, VolumeDescriptor],
    context: Optional[dict[str, Any]] = None) -> base.Volume:
  """Open a volume from a volume descriptor.

  Args:
    spec: A serialized or fully loaded instance of VolumeDescriptor, or a path
      to a JSON serialized VolumeDescriptor object.
    context: Optional TensorStore Context specification. Only applicable to
      TensorStore-backed volumes.

  Returns:
    Loaded volume, including any volume decorators.

  Raises:
    ValueError: If volinfo is present, as it is internal-only.
  """
  spec = load_descriptor(spec)

  if spec.volinfo:
    raise ValueError('volinfo field not supported')

  if context:
    if not spec.tensorstore_config:
      raise ValueError(
          'Context can only be applied to TensorStore-backed volumes')
    spec = copy.deepcopy(spec)
    spec.tensorstore_config.spec['context'] = context

  config = tsv.load_ts_config(spec.tensorstore_config)

  config = typing.cast(tsv.TensorstoreConfig, config)
  volume = tsv.TensorstoreVolume(config)
  return tsv_decorator.from_specs(volume, spec.decorator_specs)
