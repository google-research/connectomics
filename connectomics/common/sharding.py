# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
"""Sharding functions useful for breaking large files into pieces."""
import hashlib


def md5_shard(segment_id: int, num_shards: int, byteorder: str = 'little',
              bytewidth: int = 8) -> int:
  """A simple sharder based on md5 hashing."""
  md5 = hashlib.md5()
  md5.update(segment_id.to_bytes(bytewidth, byteorder))
  return int.from_bytes(md5.digest(), byteorder) % num_shards
