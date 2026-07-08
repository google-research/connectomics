# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import glob
import json
import os

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from connectomics.segmentation import export_pychunkedgraph_beam
import numpy as np
import tensorstore as ts
import zstandard as zstd

from connectomics.segmentation import pychunkedgraph_proto as export_data_pb2


class ExportPychunkedgraphBeamTest(absltest.TestCase):

  def test_get_pcg_segment_id_base_basic(self):
    self.assertEqual(
        (1 << 56) + (1 << 55),
        export_pychunkedgraph_beam.get_pcg_segment_id_base(1, (1, 0, 0)),
    )
    self.assertEqual(
        (1 << 56) + (1 << 54),
        export_pychunkedgraph_beam.get_pcg_segment_id_base(2, (1, 0, 0)),
    )
    self.assertEqual(
        (1 << 56) + (1 << 54) + (2 << 52) + (3 << 50),
        export_pychunkedgraph_beam.get_pcg_segment_id_base(2, (1, 2, 3)),
    )

  def test_basic2x1x1(self):
    temp_dir = self.create_tempdir().full_path

    array = np.array(
        [
            [1, 2, 2, 3],
            [4, 5, 5, 6],
        ],
        dtype=np.uint64,
    )
    array = array.T[:, :, np.newaxis]

    ts_spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(temp_dir, 'input_zarr/'),
        },
    }
    dataset = ts.open(
        ts_spec, dtype=ts.uint64, shape=array.shape, create=True
    ).result()
    dataset.write(array).result()

    request = export_data_pb2.ExportPychunkedgraphRequest()
    request.input_tensorstore_spec = json.dumps(ts_spec)
    request.pychunkedgraph_block_shape.x = 2
    request.pychunkedgraph_block_shape.y = 2
    request.pychunkedgraph_block_shape.z = 1

    output_kvstore_spec = {
        'driver': 'file',
        'path': os.path.join(temp_dir, 'graph/'),
    }
    request.output_pychunkedgraph_kvstore = json.dumps(output_kvstore_spec)
    dropped_edges_path = os.path.join(temp_dir, 'dropped_edges.csv')
    request.output_dropped_edges_path = dropped_edges_path

    edges = [
        (1, 2, 10.0),
        (4, 6, 11.0),
        (5, 6, 12.0),
        (3, 6, 13.0),
    ]

    with TestPipeline() as p:
      pcoll_edges = p | beam.Create(edges)
      export_pychunkedgraph_beam.process_request(request, pcoll_edges)

    with open(
        os.path.join(temp_dir, 'graph/pychunkedgraph_metadata.json'), 'r'
    ) as f:
      metadata = json.load(f)
      self.assertEqual(metadata['chunk_grid_shape'], [2, 1, 1])
      self.assertEqual(metadata['chunk_shape'], [2, 2, 1])
      self.assertEqual(metadata['volume_shape'], [4, 2, 1])
      self.assertEqual(metadata['spatial_id_bits'], 1)

    block0_segment_base = export_pychunkedgraph_beam.get_pcg_segment_id_base(
        1, (0, 0, 0)
    )
    block1_segment_base = export_pychunkedgraph_beam.get_pcg_segment_id_base(
        1, (1, 0, 0)
    )

    pcg0_1 = block0_segment_base + 0
    pcg0_2 = block0_segment_base + 1
    pcg0_4 = block0_segment_base + 2
    pcg0_5 = block0_segment_base + 3
    pcg1_2 = block1_segment_base + 0
    pcg1_5 = block1_segment_base + 2
    pcg1_6 = block1_segment_base + 3

    def decode_chunk_edges(filepath):
      with open(filepath, 'rb') as f:
        compressed = f.read()
      dctx = zstd.ZstdDecompressor()
      uncompressed = dctx.decompress(compressed)
      msg = (
          export_data_pb2.ExportPychunkedgraphRequest.PychunkedgraphChunkEdgesMsg()
      )
      msg.ParseFromString(uncompressed)

      def decode_msg(edges_msg):
        node_ids1 = np.frombuffer(edges_msg.node_ids1, dtype='<u8')
        node_ids2 = np.frombuffer(edges_msg.node_ids2, dtype='<u8')
        affinities = np.frombuffer(edges_msg.affinities, dtype='<f4')
        return list(zip(node_ids1, node_ids2, affinities))

      return (
          decode_msg(msg.in_chunk),
          decode_msg(msg.cross_chunk),
          decode_msg(msg.between_chunk),
      )

    in_chunk, cross_chunk, between_chunk = decode_chunk_edges(
        os.path.join(temp_dir, 'graph/edges_0_0_0.proto.zst')
    )
    self.assertIn((pcg0_1, pcg0_1, float('inf')), in_chunk)
    self.assertIn((pcg0_1, pcg0_2, 10.0), in_chunk)
    self.assertIn((pcg0_2, pcg1_2, float('inf')), cross_chunk)
    self.assertIn((pcg0_4, pcg1_6, 11.0), between_chunk)

    def decode_components(filepath):
      with open(filepath, 'rb') as f:
        data = f.read()
      msg = (
          export_data_pb2.ExportPychunkedgraphRequest.PychunkedgraphChunkComponentsMsg()
      )
      msg.ParseFromString(data)
      return list(msg.components)

    comps0 = decode_components(
        os.path.join(temp_dir, 'graph/components_0_0_0.proto')
    )
    self.assertEqual(
        comps0, [3, pcg0_1, pcg0_2, pcg1_2, 4, pcg0_4, pcg0_5, pcg1_5, pcg1_6]
    )

  def test_basic2x3x1(self):
    temp_dir = self.create_tempdir().full_path

    array = np.array(
        [
            [1, 2, 2, 3],
            [4, 5, 5, 6],
            [7, 5, 5, 9],
        ],
        dtype=np.uint64,
    )
    array = array.T[:, :, np.newaxis]

    ts_spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(temp_dir, 'input_zarr/'),
        },
    }
    dataset = ts.open(
        ts_spec, dtype=ts.uint64, shape=array.shape, create=True
    ).result()
    dataset.write(array).result()

    request = export_data_pb2.ExportPychunkedgraphRequest()
    request.input_tensorstore_spec = json.dumps(ts_spec)
    request.pychunkedgraph_block_shape.x = 2
    request.pychunkedgraph_block_shape.y = 1
    request.pychunkedgraph_block_shape.z = 1

    output_kvstore_spec = {
        'driver': 'file',
        'path': os.path.join(temp_dir, 'graph/'),
    }
    request.output_pychunkedgraph_kvstore = json.dumps(output_kvstore_spec)
    dropped_edges_path = os.path.join(temp_dir, 'dropped_edges.csv')
    request.output_dropped_edges_path = dropped_edges_path

    edges = [
        (1, 2, 10.0),
        (4, 6, 11.0),
        (5, 6, 12.0),
        (3, 6, 13.0),
        (9, 1, 14.0),
    ]

    with TestPipeline() as p:
      pcoll_edges = p | beam.Create(edges)
      export_pychunkedgraph_beam.process_request(request, pcoll_edges)

    dropped = ''
    for f_path in sorted(glob.glob(dropped_edges_path + '*')):
      with open(f_path, 'r') as f:
        dropped += f.read()
    self.assertIn('1,9', dropped)


if __name__ == '__main__':
  absltest.main()
