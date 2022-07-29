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
"""Dask plugins supporting subvolume processing."""

import typing

from connectomics.volume import base
from connectomics.volume import descriptor
from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
import dask.distributed as dd


class TSVolume(dd.WorkerPlugin):

  def __init__(self, vol_descriptor: descriptor.VolumeDescriptor):
    self.descriptor = vol_descriptor.to_json(indent=2)

  def setup(self, worker: dd.Worker):
    self.vol = descriptor.open_descriptor(self.descriptor)


class ProcessSubvolumeWorker(dd.WorkerPlugin):
  """ProcessSubvolume worker implemented in Dask."""

  _worker: dd.Worker
  _id: str
  _config: subvolume_processor.ProcessVolumeConfig

  def __init__(self, cluster_id: str,
               config: subvolume_processor.ProcessVolumeConfig):
    self._id = cluster_id
    self._config = config

  @staticmethod
  def name(cluster_id: str) -> str:
    return f'dask_worker_{cluster_id}'

  @staticmethod
  def volume_name(cluster_id: str, name: str) -> str:
    return f'{name}_{cluster_id}'

  def setup(self, worker: dd.Worker):
    self._worker = worker

  def _get_volume(self, name: str) -> base.BaseVolume:
    ts_volume = typing.cast(
        TSVolume,
        self._worker.plugins[ProcessSubvolumeWorker.volume_name(self._id,
                                                                name)])
    return ts_volume.vol

  def process_subvolume(self,
                        subvol: subvolume.Subvolume) -> subvolume.Subvolume:
    processor = subvolume_processor.get_processor(self._config.processor)
    processor.set_effective_subvol_and_overlap(subvol.bbox.size,
                                               processor.overlap())
    result = processor.process(subvol)
    if isinstance(result, list):
      raise ValueError(
          'Dask SubvolumeProcessors must return a single subvolume.')
    assert isinstance(result, base.Subvolume)
    return result
