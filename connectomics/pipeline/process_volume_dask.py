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
r"""Process a volume using the Dask framework.

Supports launching local clusters and connecting to remote, already running
clusters (e.g. by launch_local_dask_cluster.py or a Kubernetes cluster running
in the cloud).

Note that currently only TensorStore + n5 compression are supported as output
formats, though we expect to be able to support arbitrary TensorStore output
formats in the near future.

Example usage:

  python3 connectomics/pipeline/process_volume_dask.py \
        --config path/to/process/volume/config.json \
        --dask_config path/to/cluster/config.json
"""

from absl import app
from absl import flags
from absl import logging
from connectomics.common import file
from connectomics.pipeline.dask import cluster
from connectomics.pipeline.dask import runner
from connectomics.volume import subvolume_processor

_CONFIG = flags.DEFINE_string(
    name='config',
    default=None,
    help='JSON ProcessVolumeConfig, or path to JSON config.',
    required=True)
_DASK_CONFIG = flags.DEFINE_string(
    'dask_config',
    None,
    'JSON DaskClusterConfig, or path to JSON config.',
    required=True)
_RESTART_WORKERS = flags.DEFINE_bool(
    'restart_workers', False,
    'Restart workers on initial connection to the cluster. If you are using '
    'a long-running cluster and you are iterating on code, the workers will '
    'not pick up or reload any changes unless they are restarted prior to '
    'executing a pipeline.')


def verify_process_volume_config(
    config: subvolume_processor.ProcessVolumeConfig):
  if config.input_volume.volinfo is not None:
    raise ValueError('Volinfo not supported externally')


def load_configs(
    serialized_config_or_path: str, serialized_dask_config_or_path: str
) -> tuple[subvolume_processor.ProcessVolumeConfig, cluster.DaskClusterConfig]:
  """Load the requested ProcessVolumeConfig and DaskClusterConfig.

  Args:
    serialized_config_or_path: Serialized or file path to a ProcessVolumeConfig.
    serialized_dask_config_or_path: Serialized or file path to a
      DaskClusterConfig.

  Returns:
    A tuple containing the loaded configs.
  """
  config = file.load_dataclass(subvolume_processor.ProcessVolumeConfig,
                               serialized_config_or_path)
  assert config
  verify_process_volume_config(config)

  dask_config = file.load_dataclass(cluster.DaskClusterConfig,
                                    serialized_dask_config_or_path)
  assert dask_config
  return config, dask_config


def main(_) -> None:
  config, dask_config = load_configs(_CONFIG.value, _DASK_CONFIG.value)
  logging.info('Loaded config: %s', config.to_json(indent=2))
  verify_process_volume_config(config)

  dask_processor = runner.DaskRunner.connect(
      dask_config, restart=_RESTART_WORKERS.value)
  dask_processor.run(config)


if __name__ == '__main__':
  app.run(main)
