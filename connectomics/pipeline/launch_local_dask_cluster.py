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
r"""Launch a local dask cluster.

Example usage:

  python3 connectomics/pipeline/launch_local_dask_cluster.py \
      --dask_config path/to/dask_cluster.json
"""

from absl import app
from absl import flags

from connectomics.common import file
from connectomics.pipeline.dask import cluster

_DASK_CONFIG = flags.DEFINE_string(
    'dask_config',
    None,
    'JSON DaskClusterConfig, or path to serialized config.',
    required=True)


def main(_) -> None:
  config = file.load_dataclass(cluster.DaskClusterConfig, _DASK_CONFIG.value)
  if not config.local:
    raise ValueError('Only local clusters currently supported.')
  cluster.local_cluster(config, block=True)


if __name__ == '__main__':
  app.run(main)
