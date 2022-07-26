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
import os
import typing
from typing import Optional
import uuid

from absl import logging
from connectomics.common import bounding_box
from connectomics.common import box_generator
from connectomics.common import file
from connectomics.common import utils
from connectomics.pipeline.dask import cluster
from connectomics.pipeline.dask import plugins
from connectomics.volume import descriptor
from connectomics.volume import subvolume_processor
from connectomics.volume import tensorstore as tsv
import dask.distributed as dd

ProcessSubvolumeWorker = plugins.ProcessSubvolumeWorker
TSVolume = plugins.TSVolume


def process_bundle(pipeline_id: str, *args, **kwargs):
  worker = dd.get_worker()
  dask_worker = typing.cast(
      ProcessSubvolumeWorker,
      worker.plugins[ProcessSubvolumeWorker.name(pipeline_id)])
  return dask_worker.process_bundle(*args, **kwargs)


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
    self._client.register_worker_plugin(
        TSVolume(config.input_volume),
        name=ProcessSubvolumeWorker.volume_name(self.id, 'input_volume'))
    self._client.register_worker_plugin(
        TSVolume(config.output_volume),
        name=ProcessSubvolumeWorker.volume_name(self.id, 'output_volume'))

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

  def _compute_output_spec(
      self, config: subvolume_processor.ProcessVolumeConfig
  ) -> subvolume_processor.ProcessVolumeConfig:
    """Compute the output chunking based on the input SubvolumeProcessor.

    Args:
      config: ProcessVolumeConfig pipeline configuration.

    Returns:
      Updated new ProcessSubvolumeConfig with correct output_volume.
    """
    computed_config = copy.deepcopy(config)
    processor = subvolume_processor.get_processor(computed_config.processor)
    input_volume = descriptor.open_descriptor(computed_config.input_volume)

    processor.set_effective_subvol_and_overlap(computed_config.subvolume_size,
                                               processor.overlap())

    output_box = processor.expected_output_box(
        bounding_box.BoundingBox([0, 0, 0],
                                 size=computed_config.subvolume_size))

    output_descriptor = computed_config.output_volume
    # Set metadata
    assert output_descriptor.tensorstore_config
    output_voxel_size = processor.pixelsize(input_volume.voxel_size)
    output_chunk_size = output_box.size
    output_descriptor.tensorstore_config.metadata = tsv.TensorstoreMetadata(
        voxel_size=tuple([utils.from_np_type(v) for v in output_voxel_size]),
        bounding_boxes=copy.deepcopy(computed_config.bounding_boxes))
    # Set TS Config
    # TODO(timblakely): Check to make sure it's n5
    output_channels = processor.num_channels(input_volume.shape[0])
    out_spec = output_descriptor.tensorstore_config.spec
    out_volume_name = computed_config.output_dir
    out_spec['kvstore']['path'] = out_volume_name
    out_spec['metadata'] = {
        'blockSize': [output_channels] + list(output_chunk_size[::-1]),
        'dataType':
            utils.canonicalize_dtype_for_ts(
                processor.output_type(input_volume.dtype)),
        'dimensions': [output_channels] +
                      [int(x) for x in input_volume.shape[1:]],
    }
    # Ensure we can JSON serialize it; np.uint64 isn't a valid JSON type :/
    for field in ['blockSize', 'dimensions']:
      out_spec['metadata'][field] = [
          utils.from_np_type(v) for v in out_spec['metadata'][field]
      ]
    attributes_file = os.path.join(out_volume_name, 'attributes.json')
    is_new = False
    if not file.Exists(attributes_file):
      is_new = True
      out_spec['create'] = True
      out_spec['delete_existing'] = True
      logging.info('Volume %s does not exist, creating', out_volume_name)

    # Open the output volume once to make sure it's created if need be,
    # otherwise all workers will try to create it at once and all but one will
    # error out.
    logging.info('Opening output volume %s', out_volume_name)
    temp_desc = descriptor.open_descriptor(output_descriptor)
    del temp_desc

    if is_new:
      del out_spec['create']
      del out_spec['delete_existing']
    logging.info('Computed config: %s', computed_config.to_json(indent=2))
    return computed_config

  def run(self, config: subvolume_processor.ProcessVolumeConfig, wait=True):
    """Begin processing a volume on a Dask cluster.

    Args:
      config: ProcessVolumeConfig to process.
      wait: Wait for all tasks to complete before returning.
    """

    computed_config = self._compute_output_spec(config)
    del config
    self._register_with_cluster(computed_config)

    # TODO(timblakely): supoport back-shift-small-sub-boxes
    generator = box_generator.MultiBoxGenerator(
        computed_config.bounding_boxes,
        box_size=computed_config.subvolume_size,
        box_overlap=computed_config.overlap)
    logging.info('Number of bounding boxes to process: %s', generator.num_boxes)

    tasks = []
    for batch in utils.batch(
        range(generator.num_boxes), computed_config.batch_size):
      bboxes = [generator.generate(b)[1] for b in batch]
      tasks.append(self._client.submit(process_bundle, self.id, bboxes))

    if wait:
      dd.wait(tasks)

    # Write out the VolumeDescriptor alongside the volume.
    with file.GFile(
        os.path.join(computed_config.output_dir, 'volume.json'), 'w') as f:
      f.write(computed_config.output_volume.to_json(indent=2))
