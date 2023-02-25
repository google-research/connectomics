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
"""Reader for CSV ZIP sharded embedding release."""
import os
from typing import List, Mapping, Tuple
import zipfile
from connectomics.common import sharding


class EmbeddingReader:
  """Reader to load and parse embedding data from sharded ZIP archives."""

  def __init__(self, filesystem, zipdir: str, sharder):
    self._filesystem = filesystem  # Provides open(path) method.
    self._zipdir = zipdir
    self._sharder = sharder

  def _get_csv_data(self, seg_id: int) -> str:
    shard = self._sharder(seg_id)
    zip_path = os.path.join(self._zipdir, f'{shard}.zip')
    with self._filesystem.open(zip_path) as f:
      with zipfile.ZipFile(f) as z:
        with z.open(f'{seg_id}.csv') as c:
          return c.read().decode('utf-8')

  def _parse_csv_data(
      self, csv_data: str
  ) -> Mapping[Tuple[float, float, float], List[float]]:
    """Parses CSV rows into mapping from node XYZ coord to embedding vector."""
    embeddings_from_xyz = {}
    for l in csv_data.split('\n'):
      fields = l.split(',')
      # node_id = int(fields[0])  # This is not currently useful for much.
      xyz = tuple(float(f) for f in fields[1:4])
      embedding = [float(f) for f in fields[4:]]
      assert xyz not in embeddings_from_xyz
      embeddings_from_xyz[xyz] = embedding

    return embeddings_from_xyz

  def __getitem__(self, seg_id: int):
    csv_data = self._get_csv_data(seg_id)
    return self._parse_csv_data(csv_data)


# Unfortunately, this round of exports were accidentally run with bytewidth 64.
# Newer exports will be fixed to be bytewidth 8.
DATA_URL_FROM_KEY_BYTEWIDTH64 = dict(
    # H01 human temporal cortex dataset.
    h01='gs://h01-release/data/20220326/c3/embeddings/segclr_csvzips',
    h01_nm_coord=(
        'gs://h01-release/data/20220326/c3/embeddings/segclr_nm_coord_csvzips'
    ),
    h01_agg10um='gs://h01-release/data/20220326/c3/embeddings/segclr_aggregated_10um_csvzips',

    # MICrONS mouse visual cortex dataset.
    microns_v343=(
        'gs://iarpa_microns/minnie/minnie65/embeddings_m343/segclr_csvzips'
    ),
    microns_nm_coord_public_offset_v343='gs://iarpa_microns/minnie/minnie65/embeddings_m343/segclr_nm_coord_public_offset_csvzips',
    microns_v343_agg25um='gs://iarpa_microns/minnie/minnie65/embeddings_m343/segclr_aggregated_25um_csvzips',

    # Older v117 segmentation of MICrONS.
    microns_v117='gs://iarpa_microns/minnie/minnie65/embeddings/segclr_csvzips',
)


def get_reader(key: str, filesystem, num_shards: int = 10_000):
  """Convenience helper to get reader for given dataset key."""
  if key in DATA_URL_FROM_KEY_BYTEWIDTH64:
    url = DATA_URL_FROM_KEY_BYTEWIDTH64[key]
    bytewidth = 64
  else:
    raise ValueError(f'Key not found: {key}')

  def sharder(segment_id: int) -> int:
    return sharding.md5_shard(
        segment_id, num_shards=num_shards, bytewidth=bytewidth)

  return EmbeddingReader(filesystem, url, sharder)
