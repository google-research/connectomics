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
"""Dask-based SubvolumeProcessor."""

import copy
import dataclasses
import os
import typing
from typing import Any, Optional
import uuid

from absl import logging
from connectomics.common import bounding_box
from connectomics.common import file
from connectomics.pipeline.dask import cluster
from connectomics.pipeline.dask import plugins
from connectomics.volume import base
from connectomics.volume import descriptor
from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
from connectomics.volume import tensorstore as tsv
import dask.array as da
from dask.delayed import Delayed
import dask.distributed as dd
import numpy as np

ProcessSubvolumeWorker = plugins.ProcessSubvolumeWorker
TSVolume = plugins.TSVolume


@dataclasses.dataclass
class ProcessorOutput:
  """Calculated output parameters for a processed volume."""
  chunk_size: np.array
  voxel_size: np.array
  bounding_boxes: list[bounding_box.BoundingBoxBase]
  dtype: str

  # Dimensions of the chunks for each channel. For example, a 1x1000^3 4D volume
  # in CZYX format processed at an XYZ subvolume size of 350x300x400 would
  # result in the following chunksizes:
  #    ((1,), (350, 350, 300), (300, 300, 300, 100), (400, 400, 200))
  dask_chunks: tuple[tuple[int]]


class DaskTensorStoreWriter:
  """Dask-compatible wrapper class passable to dask.array.store()."""

  def __init__(self, desc, context: Optional[dict[str, Any]] = None):
    self.out = _open_volume_with_context(desc, context)

  def __setitem__(self, slices: list[slice], value):
    # Ensure clamping if needed; Dask can't natively handle sub-chunk-size
    # chunks that SubvolumeProcessors can potentially output at the edges.
    clipped_slices = []
    for i, slc in enumerate(slices):
      sz = slc.stop - slc.start
      diff = sz - value.shape[i]
      if diff < 0:
        raise ValueError(
            'Unexpectedly large array to write: {value.shape} vs {slices}')
      clipped_slices.append(slice(slc.start, slc.stop - diff, slc.step))
    self.out.write_slices(clipped_slices, value)


class TensorStoreDaskArray:
  """Dask-compatible wrapper around a TensorstoreVolume."""

  def __init__(self, volume: tsv.TensorstoreVolume):
    self._volume = volume
    self.dtype = self._volume.dtype
    self.chunks = self._volume.chunk_size

  def __getitem__(self, item):
    return self._volume.__getitem__(item).data

  def __getattr__(self, attr):
    if attr in self.__dict__:
      return getattr(self, attr)
    if attr == '_volume':
      raise AttributeError
    return getattr(self._volume, attr)


def bbox_from_info(info, index=0):
  info = info[index]
  array_loc = np.array(info['array-location'])[3:0:-1, :]
  return bounding_box.BoundingBox(**dict(zip(['start', 'end'], array_loc.T)))


def process_block(data, block_info, pipeline_id: str):
  worker = dd.get_worker()
  dask_worker = typing.cast(
      ProcessSubvolumeWorker,
      worker.plugins[ProcessSubvolumeWorker.name(pipeline_id)])

  bbox = bbox_from_info(block_info)
  subvol = subvolume.Subvolume(data, bbox)
  return dask_worker.process_subvolume(subvol).data


def _open_volume_with_context(
    desc: descriptor.VolumeDescriptor,
    context: Optional[dict[str, Any]] = None) -> base.BaseVolume:
  if context:
    desc = copy.deepcopy(desc)
    assert desc.tensorstore_config
    desc.tensorstore_config.spec['context'] = context
  return descriptor.open_descriptor(desc)


def _create_volume_if_necessary(desc: descriptor.VolumeDescriptor):
  modified_desc = copy.deepcopy(desc)
  assert modified_desc.tensorstore_config
  modified_desc.tensorstore_config.spec['create'] = True
  modified_desc.tensorstore_config.spec['delete_existing'] = True
  _ = descriptor.open_descriptor(modified_desc)


class DaskRunner:
  """Run a SubvolumeProcessor within a Dask cluster."""

  cluster_config: cluster.DaskClusterConfig

  _cluster: Optional[dd.SpecCluster] = None
  _client: dd.Client

  _process_config: subvolume_processor.ProcessVolumeConfig
  _id: str

  def __init__(self,
               cluster_config: cluster.DaskClusterConfig,
               pipeline_id: Optional[str] = None):
    self._id = str(uuid.uuid4()) if pipeline_id is None else pipeline_id
    logging.info('Initializing pipeline id %s', self._id)
    self.cluster_config = cluster_config

    if cluster_config.local:
      self._cluster, self._client = cluster.local_cluster(cluster_config)
    else:
      assert cluster_config.remote
      self._client = dd.Client(**cluster_config.remote.to_dict())

  @property
  def id(self) -> str:
    return self._id

  def _register_with_cluster(self,
                             config: subvolume_processor.ProcessVolumeConfig):
    """Registers required resources with the remote Dask cluster.

    Args:
      config: Configuration containing the ProcessSubvolumeWorker, intput, and
        output volumes.
    """
    # TODO(timblakely): pass this to the task via functools.partial.
    self._process_config = config
    self._client.register_worker_plugin(
        ProcessSubvolumeWorker(self.id, config),
        name=ProcessSubvolumeWorker.name(self.id))

  @classmethod
  def connect(cls,
              cluster_config: cluster.DaskClusterConfig,
              restart=True) -> 'DaskRunner':
    dask_runner = DaskRunner(cluster_config)
    if restart:
      # TODO(timblakely): Need a workaround for
      # https://github.com/dask/distributed/issues/6455
      dask_runner._client.restart()
    return dask_runner

  def _calculate_processor_output(
      self, config: subvolume_processor.ProcessVolumeConfig) -> ProcessorOutput:
    """Determines the output volume shape/params from a ProcessVolumeConfig.

    Args:
      config: Configuration for the pipeline

    Returns:
      ProcessorOutput containing the various aspects of the expected output
      volume.
    """
    processor = subvolume_processor.get_processor(config.processor)
    processor.set_effective_subvol_and_overlap(config.subvolume_size,
                                               processor.overlap())
    output_box = processor.expected_output_box(
        bounding_box.BoundingBox([0, 0, 0], size=config.subvolume_size))

    input_volume = _open_volume_with_context(config.input_volume,
                                             config.input_ts_context)

    output_channels = processor.num_channels(input_volume.shape[0])
    chunk_size = [output_channels, *output_box.size[::-1]]

    voxel_size = processor.pixelsize(input_volume.voxel_size)

    bboxes = [
        processor.expected_output_box(b) for b in input_volume.bounding_boxes
    ]

    dtype = processor.output_type(input_volume.dtype)

    # TODO(timblakely): support multiple bounding boxes.
    # Calculate expected dask output channels to the output size is correct.
    dask_chunks = []

    for idx, lim in enumerate(input_volume.shape):
      if idx == 0:
        # Channel dimension is always aligned
        dask_chunks.append((output_channels,))
        continue
      chunk_sizes = [chunk_size[idx]]
      remainder = lim % chunk_size[idx]
      if remainder:
        chunk_sizes.append(remainder)
      dask_chunks.append(tuple(chunk_sizes))

    return ProcessorOutput(chunk_size, voxel_size, bboxes, dtype.__name__,
                           dask_chunks)

  def run(self,
          config: subvolume_processor.ProcessVolumeConfig,
          wait=True) -> Optional[Delayed]:
    """Begin processing a volume on a Dask cluster.

    Args:
      config: ProcessVolumeConfig to process.
      wait: Wait for all tasks to complete before returning. If not set, will
        return the dask.Delayed value from the dask.array.store() call.

    Returns:
      The dask.Delayed value from dask.array.store() if `wait` is set.
    """
    processed = self._calculate_processor_output(config)

    input_vol = _open_volume_with_context(config.input_volume,
                                          config.input_ts_context)

    ts_array = TensorStoreDaskArray(input_vol)
    dask_array = da.from_array(ts_array, chunks=tuple(ts_array.chunks))

    # Rechunk according to subvolume_size.

    # Apply the mapping. We need to calculate the chunk size prior since the
    # subvolume processor may change the output type.
    mapped_array = dask_array.map_blocks(
        process_block,
        dtype=input_vol.dtype,
        chunks=processed.chunk_size,
        pipeline_id=self.id)

    # Now that we have the mapped array we can determine the final dimensions of
    # the output volume. Note that it may not match the actual bounding boxes
    # which may be smaller.
    out_desc = descriptor.VolumeDescriptor(
        tensorstore_config=tsv.TensorstoreConfig(
            spec={
                'driver': 'n5',
                'kvstore': {
                    'driver': 'file',
                    'path': config.output_dir,
                },
                'metadata': {
                    'blockSize': processed.chunk_size,
                    'dataType': processed.dtype,
                    'dimensions': mapped_array.shape
                },
            },
            metadata=tsv.TensorstoreMetadata(
                voxel_size=processed.voxel_size,
                bounding_boxes=processed.bounding_boxes)))

    # Create the volume ahead of time so multiple workers don't try to write to
    # attributes.json.
    _create_volume_if_necessary(out_desc)

    self._register_with_cluster(config)

    # Store the volume. We don't need to lock, as our final chunk size is
    # perfectly chunk-aligned to the output.
    result = mapped_array.store(
        DaskTensorStoreWriter(out_desc, config.output_ts_context),
        lock=False,
        compute=wait)

    # Write out the VolumeDescriptor alongside the volume.
    with file.GFile(os.path.join(config.output_dir, 'volume.json'), 'w') as f:
      f.write(out_desc.to_json(indent=2))

    return result
