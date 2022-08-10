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
"""Dask pipeline and utility functions."""

import dataclasses
import time
from typing import Optional

from absl import logging
import dask.distributed as dd
import dataclasses_json


@dataclasses.dataclass
class LocalDaskClusterConfig(dataclasses_json.DataClassJsonMixin):
  """Dataclass version of parameters passed to dask.distributed.LocalCluster."""

  # Number of workers in the local cluster. Exact count depends on the
  # underlying Dask configuration (i.e. `'threaded'` vs `'processes'`), but in
  # general will default to the number of available virtual cores.
  n_workers: Optional[int]

  # Number of threads per worker. Exact count depends on the underlying Dask
  # configuration (i.e. `'threaded'` vs `'processes'`), but in general will
  # default to the number of available virtual cores. Note that in combination
  # with `n_workers`, thread count can be intentionally oversubscribed, which
  # can be useful in situations where I/O is a bottleneck.
  threads_per_worker: Optional[int]

  # Name of the cluster.
  name: str = 'local_cluster'

  # Port for workers to communicate with the scheduler.
  scheduler_port: int = 8786

  # Address to bind the scheduler and dashboard (by default) to.
  ip: str = '127.0.0.1'

  # Address:port to launch the monitoring dashboard. If address is not
  # specified, it will be assumed to be the IP address in `ip` (port must still
  # be prefixed with ":").
  dashboard_address: str = ':8787'


@dataclasses.dataclass
class RemoteDaskClusterConfig(dataclasses_json.DataClassJsonMixin):
  """Dataclass configuration for remote Dask clusters."""

  # Address of the remote cluster in the format of addr:port.
  address: str


@dataclasses.dataclass
class DaskClusterConfig(dataclasses_json.DataClassJsonMixin):
  """Dataclass configuration for Dask clusters."""

  # Configuration to launch a local Dask cluster.
  local: Optional[LocalDaskClusterConfig] = None

  # Configuration to connect to a remote dask cluster (including an
  # externally-launched local cluster).
  remote: Optional[RemoteDaskClusterConfig] = None

  # In the event workers need additional python paths, they can be added by
  # listing them here. Paths will be updated when the client connects.
  additional_pythonpaths: Optional[list[str]] = None


def local_cluster(config: DaskClusterConfig,
                  block=False) -> tuple[dd.SpecCluster, dd.Client]:
  """Start a local Dask cluster.

  Args:
    config: Cluster config.
    block: If true, the function will never return and instead sleep forever;
      useful for launching a cluster separately from pipelines.

  Returns:
    tuple of dask.SpecCluster (i.e. the cluster object) and a dask.Client.
  """
  assert config.local
  logging.info('Starting local cluster in %sblocking mode',
               ('' if block else 'non-'))
  cluster = dd.LocalCluster(**config.local.to_dict())
  logging.info('Connecting client to cluster: %s', cluster)
  client = dd.Client(cluster)
  if config.additional_pythonpaths:
    logging.info('Adding additional python paths: %s',
                 config.additional_pythonpaths)

    def _add_paths(paths: list[str]):
      import sys  # pylint: disable=g-import-not-at-top
      for path in paths:
        sys.path.append(path)

    client.run(_add_paths, config.additional_pythonpaths)

  while block:
    time.sleep(1)

  return cluster, client
