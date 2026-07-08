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

"""Apache Beam Python pipeline for exporting PyChunkedGraph ingest data."""

import collections
import json
import math

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import tensorstore as ts
import zstandard as zstd

from apache_beam.io.gcp import bigquery as beam_bq
from connectomics.segmentation import pychunkedgraph_proto as export_data_pb2

_REQUEST_TEXT = flags.DEFINE_string(
    'request_text',
    None,
    'TextFormat ExportPychunkedgraphRequest',
)
_RUNNER = flags.DEFINE_string('runner', None, 'Pipeline runner to use')
_OUTPUT_TENSORSTORE_SPEC = flags.DEFINE_string(
    'output_tensorstore_spec',
    None,
    'JSON TensorStore spec for writing remapped supervoxel volume. '
    'Must be rank 3, dtype uint64, same shape as input volume.',
)


def get_pcg_segment_id_base(
    coord_bits: int, cell_coords: tuple[int, int, int]
) -> int:
  id_base = 1 << (64 - 8)
  for i in range(3):
    id_base |= int(cell_coords[i]) << (64 - 8 - coord_bits * (i + 1))
  return id_base


class GridProcessorDoFn(beam.DoFn):
  """Base DoFn with shared grid computation logic."""

  def __init__(
      self, request_proto: export_data_pb2.ExportPychunkedgraphRequest
  ):
    self.request_proto = request_proto

  def setup(self):
    json_spec = json.loads(self.request_proto.input_tensorstore_spec)
    self.store = ts.open(json_spec, read=True).result()

    shape = self.store.domain.shape
    self.pychunkedgraph_block_shape = np.array([
        self.request_proto.pychunkedgraph_block_shape.x,
        self.request_proto.pychunkedgraph_block_shape.y,
        self.request_proto.pychunkedgraph_block_shape.z,
    ])
    self.volume_shape = np.array(shape[:3])
    self.pychunkedgraph_grid_shape = np.ceil(
        self.volume_shape / self.pychunkedgraph_block_shape
    ).astype(np.int64)

    self.pychunkedgraph_coord_bits = 0
    for i in range(3):
      while (
          1 << self.pychunkedgraph_coord_bits
      ) < self.pychunkedgraph_grid_shape[i]:
        self.pychunkedgraph_coord_bits += 1

  def get_cell_coords_from_cell_index(
      self, cell_index: int
  ) -> tuple[int, int, int]:
    coords = []
    for i in range(3):
      coords.append(int(cell_index % self.pychunkedgraph_grid_shape[i]))
      cell_index //= self.pychunkedgraph_grid_shape[i]
    return tuple(coords)

  def get_cell_index_from_cell_coords(
      self, coords: tuple[int, int, int]
  ) -> int:
    block_id = 0
    stride = 1
    for i in range(3):
      block_id += stride * coords[i]
      stride *= self.pychunkedgraph_grid_shape[i]
    return block_id


class MapSupervoxelsFn(GridProcessorDoFn):
  """Maps supervoxels."""

  def __init__(
      self,
      request_proto: export_data_pb2.ExportPychunkedgraphRequest,
      output_tensorstore_spec: str | None = None,
  ):
    super().__init__(request_proto)
    self.output_tensorstore_spec = output_tensorstore_spec

  def setup(self):
    super().setup()
    if self.output_tensorstore_spec:
      json_spec = json.loads(self.output_tensorstore_spec)
      self.output_store = ts.open(
          json_spec, dtype=ts.uint64, write=True
      ).result()
    else:
      self.output_store = None

  def process(self, input_index: int):
    pychunkedgraph_cell_coords = self.get_cell_coords_from_cell_index(
        input_index
    )

    start_voxel_coords = (
        np.array(pychunkedgraph_cell_coords) * self.pychunkedgraph_block_shape
    )
    voxel_shape = np.minimum(
        self.pychunkedgraph_block_shape, self.volume_shape - start_voxel_coords
    )

    x0, y0, z0 = start_voxel_coords
    dx, dy, dz = voxel_shape
    subvolume_array = (
        self.store[x0 : x0 + dx, y0 : y0 + dy, z0 : z0 + dz].read().result()
    )

    flat_data = subvolume_array.ravel(order='F')
    _, idx = np.unique(flat_data, return_index=True)
    sorted_idx = np.sort(idx)
    unique_ids_with_zero = flat_data[sorted_idx]
    unique_ids = [int(k) for k in unique_ids_with_zero if k != 0]

    if self.output_store is not None:
      base_segment_id = get_pcg_segment_id_base(
          self.pychunkedgraph_coord_bits, pychunkedgraph_cell_coords
      )
      remap = np.zeros_like(subvolume_array)
      for i, orig_id in enumerate(unique_ids):
        remap[subvolume_array == orig_id] = base_segment_id + i
      x0i, y0i, z0i = int(x0), int(y0), int(z0)
      dxi, dyi, dzi = int(dx), int(dy), int(dz)
      self.output_store[
          x0i : x0i + dxi, y0i : y0i + dyi, z0i : z0i + dzi
      ].write(remap).result()

    for k in unique_ids:
      yield beam.pvalue.TaggedOutput('supervoxel_to_block', (k, input_index))

    for neighbor_i in range(27):
      neighbor_offset = [0, 0, 0]
      v = neighbor_i
      for i in range(3):
        neighbor_offset[i] = (v % 3) - 1
        v //= 3

      neighbor_coords = (
          pychunkedgraph_cell_coords[0] + neighbor_offset[0],
          pychunkedgraph_cell_coords[1] + neighbor_offset[1],
          pychunkedgraph_cell_coords[2] + neighbor_offset[2],
      )

      out_of_bounds = False
      for i in range(3):
        if (
            neighbor_coords[i] < 0
            or neighbor_coords[i] >= self.pychunkedgraph_grid_shape[i]
        ):
          out_of_bounds = True
          break

      if out_of_bounds:
        continue

      neighbor_block_id = self.get_cell_index_from_cell_coords(neighbor_coords)
      yield beam.pvalue.TaggedOutput(
          'map_output', (int(neighbor_block_id), (input_index, unique_ids))
      )


class OutputPychunkedgraphDataFn(GridProcessorDoFn):
  """Outputs PyChunkedGraph data."""

  def setup(self):
    super().setup()
    try:
      json_spec = json.loads(self.request_proto.output_pychunkedgraph_kvstore)
    except json.JSONDecodeError:
      json_spec = self.request_proto.output_pychunkedgraph_kvstore
    self.kvstore = ts.KvStore.open(json_spec).result()

  def process(self, element):
    block_id, group = element
    supervoxel_maps = group['supervoxel_maps']
    edges = group['edges']

    block_to_supervoxels = {}
    for map_tuple in supervoxel_maps:
      source_block_id, original_ids = map_tuple
      base = get_pcg_segment_id_base(
          self.pychunkedgraph_coord_bits,
          self.get_cell_coords_from_cell_index(source_block_id),
      )
      mapping = {}
      for i, orig_id in enumerate(original_ids):
        mapping[orig_id] = base + i
      block_to_supervoxels[source_block_id] = mapping

    edges_map = collections.defaultdict(dict)
    for sv_a, sv_b, score in edges:
      if sv_a == sv_b:
        continue
      if sv_b not in edges_map[sv_a] or score > edges_map[sv_a][sv_b]:
        edges_map[sv_a][sv_b] = score
      if sv_a not in edges_map[sv_b] or score > edges_map[sv_b][sv_a]:
        edges_map[sv_b][sv_a] = score

    seen_edges = set()

    def emit_edge(a, b):
      return (min(a, b), max(a, b))

    def add_chunk_edges(source_block_id, target_block_id, keep_order):
      result_edges = []
      result_emits = []
      if (
          source_block_id not in block_to_supervoxels
          or target_block_id not in block_to_supervoxels
      ):
        return result_edges, result_emits

      source_supervoxels = block_to_supervoxels[source_block_id]
      target_supervoxels = block_to_supervoxels[target_block_id]

      for orig_source_sv, pcg_source_sv in source_supervoxels.items():
        if orig_source_sv not in edges_map:
          continue
        for target_orig_sv, score in edges_map[orig_source_sv].items():
          if target_orig_sv not in target_supervoxels:
            continue

          if source_block_id != target_block_id:
            in_chunk_in_neighbor = False
            for other_block_id, other_svs in block_to_supervoxels.items():
              if (
                  other_block_id != source_block_id
                  and orig_source_sv in other_svs
                  and target_orig_sv in other_svs
              ):
                in_chunk_in_neighbor = True
                break
            if in_chunk_in_neighbor:
              continue

          norm_edge = (
              min(orig_source_sv, target_orig_sv),
              max(orig_source_sv, target_orig_sv),
          )
          if norm_edge in seen_edges:
            continue
          seen_edges.add(norm_edge)

          result_emits.append(emit_edge(orig_source_sv, target_orig_sv))
          target_pcg_sv = target_supervoxels[target_orig_sv]

          if not keep_order:
            result_edges.append((
                min(pcg_source_sv, target_pcg_sv),
                max(pcg_source_sv, target_pcg_sv),
                score,
            ))
          else:
            result_edges.append((pcg_source_sv, target_pcg_sv, score))
      return result_edges, result_emits

    def remove_duplicate_edges(edge_list):
      return sorted(list(set(edge_list)))

    def get_in_chunk_edges():
      result, emits = add_chunk_edges(block_id, block_id, False)
      source_supervoxels = block_to_supervoxels[block_id]
      for _, pcg_sv in source_supervoxels.items():
        result.append((pcg_sv, pcg_sv, float('inf')))
      return remove_duplicate_edges(result), emits

    def get_cross_chunk_edges():
      result = []
      source_svs = block_to_supervoxels[block_id]
      for target_block_id, target_svs in block_to_supervoxels.items():
        if target_block_id == block_id:
          continue
        for orig_source_sv, pcg_source_sv in source_svs.items():
          if orig_source_sv in target_svs:
            target_pcg_sv = target_svs[orig_source_sv]
            result.append((pcg_source_sv, target_pcg_sv, float('inf')))
      return remove_duplicate_edges(result)

    def get_between_chunk_edges():
      source_coords = self.get_cell_coords_from_cell_index(block_id)
      nearest_neighbors = []
      next_nearest_neighbors = []

      for target_block_id, _ in block_to_supervoxels.items():
        if target_block_id == block_id:
          continue
        target_coords = self.get_cell_coords_from_cell_index(target_block_id)
        diff = [source_coords[i] - target_coords[i] for i in range(3)]
        dist = (diff[0] != 0) + (diff[1] != 0) + (diff[2] != 0)
        if dist == 1:
          nearest_neighbors.append(target_block_id)
        else:
          next_nearest_neighbors.append(target_block_id)

      result = []
      all_emits = []
      for target_block_id in nearest_neighbors:
        edges, emits = add_chunk_edges(block_id, target_block_id, True)
        result.extend(edges)
        all_emits.extend(emits)

      for target_block_id in next_nearest_neighbors:
        edges, emits = add_chunk_edges(block_id, target_block_id, True)
        result.extend(edges)
        all_emits.extend(emits)

      return remove_duplicate_edges(result), all_emits

    in_chunk_edges, in_emits = get_in_chunk_edges()
    cross_chunk_edges = get_cross_chunk_edges()
    between_chunk_edges, between_emits = get_between_chunk_edges()

    for emit in in_emits:
      yield emit
    for emit in between_emits:
      yield emit

    def make_edges_msg(edges_list):
      msg = export_data_pb2.ExportPychunkedgraphRequest.PychunkedgraphEdgesMsg()
      node_ids1 = []
      node_ids2 = []
      affinities = []
      for a, b, score in edges_list:
        node_ids1.append(a)
        node_ids2.append(b)
        affinities.append(score)
      msg.node_ids1 = np.array(node_ids1, dtype='<u8').tobytes()
      msg.node_ids2 = np.array(node_ids2, dtype='<u8').tobytes()
      msg.affinities = np.array(affinities, dtype='<f4').tobytes()
      return msg

    chunk_edges_msg = (
        export_data_pb2.ExportPychunkedgraphRequest.PychunkedgraphChunkEdgesMsg()
    )
    chunk_edges_msg.in_chunk.CopyFrom(make_edges_msg(in_chunk_edges))
    chunk_edges_msg.cross_chunk.CopyFrom(make_edges_msg(cross_chunk_edges))
    chunk_edges_msg.between_chunk.CopyFrom(make_edges_msg(between_chunk_edges))

    cctx = zstd.ZstdCompressor()
    encoded_edges = cctx.compress(chunk_edges_msg.SerializeToString())

    def get_components():
      seen_segment_ids = set()
      source_svs = block_to_supervoxels[block_id]
      components = []
      for orig_source_sv, _ in source_svs.items():
        if orig_source_sv in seen_segment_ids:
          continue

        queue = collections.deque()
        members = []

        def add_to_component(orig_id, q, m):
          seen_segment_ids.add(orig_id)
          q.append(orig_id)
          for _, other_svs in block_to_supervoxels.items():
            if orig_id in other_svs:
              m.append(other_svs[orig_id])

        add_to_component(orig_source_sv, queue, members)

        while queue:
          orig_sv = queue.popleft()
          if orig_sv not in edges_map:
            continue
          for target_orig_sv in edges_map[orig_sv]:
            if target_orig_sv in seen_segment_ids:
              continue
            add_to_component(target_orig_sv, queue, members)

        members.sort()
        components.append(members)

      components.sort()
      msg = (
          export_data_pb2.ExportPychunkedgraphRequest.PychunkedgraphChunkComponentsMsg()
      )
      flat_components = []
      for members in components:
        flat_components.append(len(members))
        flat_components.extend(members)
      msg.components.extend(flat_components)
      return msg

    encoded_components = get_components().SerializeToString()

    cell_coords = self.get_cell_coords_from_cell_index(block_id)
    edges_path = (
        f'edges_{cell_coords[0]}_{cell_coords[1]}_{cell_coords[2]}.proto.zst'
    )
    components_path = (
        f'components_{cell_coords[0]}_{cell_coords[1]}_{cell_coords[2]}.proto'
    )

    self.kvstore.write(edges_path, encoded_edges).result()
    self.kvstore.write(components_path, encoded_components).result()


def process_request(
    request: export_data_pb2.ExportPychunkedgraphRequest,
    pcoll_chosen_edges: beam.PCollection,
    output_tensorstore_spec: str | None = None,
):
  """Processes the export request."""

  def duplicate_edges(edge):
    a, b, score = edge
    yield (a, (b, score))
    yield (b, (a, score))

  chosen_edge_table = (
      pcoll_chosen_edges
      | 'DuplicateChosenEdges' >> beam.FlatMap(duplicate_edges)
      | 'ReshuffleDuplicatedEdges' >> beam.Reshuffle()
  )

  def normalize_edges(kv):
    a, (b, _) = kv
    if a < b:
      yield (a, b)
    elif a > b:
      yield (b, a)

  normalized_edges = chosen_edge_table | 'NormalizeChosenEdges' >> beam.FlatMap(
      normalize_edges
  )

  json_spec = json.loads(request.input_tensorstore_spec)
  store = ts.open(json_spec, read=True).result()
  volume_shape = np.array(store.domain.shape[:3])
  pychunkedgraph_block_shape = np.array([
      request.pychunkedgraph_block_shape.x,
      request.pychunkedgraph_block_shape.y,
      request.pychunkedgraph_block_shape.z,
  ])
  pychunkedgraph_grid_shape = np.ceil(
      volume_shape / pychunkedgraph_block_shape
  ).astype(np.int64)
  num_blocks = int(np.prod(pychunkedgraph_grid_shape))

  map_supervoxels_result = (
      pcoll_chosen_edges.pipeline
      | 'BlockIntegerRange' >> beam.Create(range(num_blocks))
      | 'MapSupervoxels'
      >> beam.ParDo(
          MapSupervoxelsFn(request, output_tensorstore_spec)
      ).with_outputs(
          'map_output', 'supervoxel_to_block', main='main'
      )
  )

  supervoxel_maps = map_supervoxels_result.map_output
  supervoxel_to_block = map_supervoxels_result.supervoxel_to_block

  def distribute_edges(kv):
    sv_a, group = kv
    blocks = list(group['supervoxel_to_block'])
    if not blocks:
      return
    edges = group['chosen_edge_table']
    for b, score in edges:
      for block_id in blocks:
        yield (block_id, (sv_a, b, score))

  block_to_edges = (
      {
          'supervoxel_to_block': supervoxel_to_block,
          'chosen_edge_table': chosen_edge_table,
      }
      | 'JoinEdgesToSupervoxels' >> beam.CoGroupByKey()
      | 'DistributeEdgesToBlocks' >> beam.FlatMap(distribute_edges)
  )

  emitted_edges = (
      {'supervoxel_maps': supervoxel_maps, 'edges': block_to_edges}
      | 'JoinPerBlockSupervoxelMapsAndEdges' >> beam.CoGroupByKey()
      | 'OutputPychunkedgraphData'
      >> beam.ParDo(OutputPychunkedgraphDataFn(request))
  )

  def format_dropped(kv):
    edge, group = kv
    in_normalized = group['normalized']
    in_emitted = group['emitted']
    if in_normalized and not in_emitted:
      yield f'{edge[0]},{edge[1]}'

  _ = (
      {
          'normalized': (
              normalized_edges | 'NormMap' >> beam.Map(lambda e: (e, 1))
          ),
          'emitted': emitted_edges | 'EmitMap' >> beam.Map(lambda e: (e, 1)),
      }
      | 'JoinForDiff' >> beam.CoGroupByKey()
      | 'FilterDropped' >> beam.FlatMap(format_dropped)
      | 'WriteDroppedEdges'
      >> beam.io.WriteToText(
          request.output_dropped_edges_path,
          append_trailing_newlines=True,
      )
  )

  def write_metadata(_):
    try:
      json_spec_out = json.loads(request.output_pychunkedgraph_kvstore)
    except json.JSONDecodeError:
      json_spec_out = request.output_pychunkedgraph_kvstore
    kvstore = ts.KvStore.open(json_spec_out).result()

    coord_bits = 0
    for i in range(3):
      while (1 << coord_bits) < pychunkedgraph_grid_shape[i]:
        coord_bits += 1

    metadata = {
        'spatial_id_bits': coord_bits,
        'fan_out': 2,
        'layer_id_bits': 8,
        'chunk_shape': [int(x) for x in pychunkedgraph_block_shape],
        'volume_shape': [int(x) for x in volume_shape],
        'chunk_grid_shape': [int(x) for x in pychunkedgraph_grid_shape],
    }
    kvstore.write(
        'pychunkedgraph_metadata.json', json.dumps(metadata).encode('utf-8')
    ).result()

  _ = (
      pcoll_chosen_edges.pipeline
      | 'TriggerMetadata' >> beam.Create([1])
      | 'WriteMetadata' >> beam.Map(write_metadata)
  )


def main(argv):
  if not _REQUEST_TEXT.value:
    raise app.UsageError('Flag --request_text must be specified.')
  request = export_data_pb2.ExportPychunkedgraphRequest.from_json(
      _REQUEST_TEXT.value
  )

  # argv[1:] contains the arguments that were not parsed as absl flags.
  options = beam.options.pipeline_options.PipelineOptions(argv[1:])
  standard_options = options.view_as(
      beam.options.pipeline_options.StandardOptions
  )
  if _RUNNER.value:
    standard_options.runner = _RUNNER.value

  with beam.Pipeline(options=options) as p:
    if request.HasField('chosen_edges_table'):
      pcoll_edges = p | 'ReadEdgesFromBQ' >> beam_bq.ReadFromBigQuery(
          table=f'{request.chosen_edges_table.project}:{request.chosen_edges_table.dataset}.{request.chosen_edges_table.table}',
      )
      num_edges_counter = beam.metrics.Metrics.counter('main', 'num-edges')

      def parse_bq_row(row):
        label_a = int(row['label_a'])
        label_b = int(row['label_b'])
        assert label_a > 0, f'label_a must be positive, got {label_a}'
        assert label_b > 0, f'label_b must be positive, got {label_b}'
        num_edges_counter.inc()
        # Apply exp to ensure non-negative, matching C++ behavior.
        try:
          score = math.exp(float(row['score']))
        except OverflowError:
          score = float('inf')
        return (label_a, label_b, score)

      pcoll_edges = (
          pcoll_edges
          | 'ParseBQRow' >> beam.Map(parse_bq_row)
          | 'ReshuffleEdges' >> beam.Reshuffle()
      )
    else:
      pcoll_edges = p | 'EmptyEdges' >> beam.Create([])

    process_request(request, pcoll_edges, _OUTPUT_TENSORSTORE_SPEC.value)


if __name__ == '__main__':
  app.run(main)
