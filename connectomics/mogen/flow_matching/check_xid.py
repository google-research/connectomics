# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
"""Check XManager experiment status by XID."""

from absl import app
from absl import flags
from google3.learning.deepmind.xmanager2.client import xmanager_api

FLAGS = flags.FLAGS
flags.DEFINE_integer('xid', 262280444, 'XID')


def main(argv):
  del argv
  client = xmanager_api.XManagerApi()
  wu = client.get_work_unit(FLAGS.xid, 1)
  cell = getattr(wu, 'borg_cell', 'unknown')
  host = getattr(wu, 'borg_hostname', 'unknown')
  print(f'WU 1: cell={cell}, host={host}, status={wu.status}')
  failure = getattr(wu, 'failure_description', 'none')
  print(f'Failure: {failure}')


if __name__ == '__main__':
  app.run(main)
