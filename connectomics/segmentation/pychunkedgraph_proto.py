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

"""Pure-Python proto-compatible message definitions for PyChunkedGraph export.

This module provides lightweight, dependency-free message classes that replicate
the exact protobuf wire format of the ExportPychunkedgraphRequest message and
its nested types.

Using pure-Python dataclasses instead of generated protobuf code avoids the need
for a protoc compilation step and proto library dependency. The wire format is
byte-compatible with standard protobuf encoding, so downstream PyChunkedGraph
consumers can parse the serialized output with either these classes or generated
proto code.
"""

import dataclasses
import json


def _encode_varint(value: int) -> bytes:  # pylint: disable=invalid-name
  """Encode an unsigned integer as a protobuf varint."""
  result = bytearray()
  while value > 0x7F:
    result.append((value & 0x7F) | 0x80)
    value >>= 7
  result.append(value & 0x7F)
  return bytes(result)


def _encode_bytes_field(field_number: int, data: bytes) -> bytes:  # pylint: disable=invalid-name
  """Encode a bytes field with protobuf wire type 2 (length-delimited)."""
  tag = _encode_varint((field_number << 3) | 2)
  length = _encode_varint(len(data))
  return tag + length + data


def _encode_packed_uint64_field(  # pylint: disable=invalid-name
    field_number: int, values: list[int]
) -> bytes:
  """Encode a packed repeated uint64 field."""
  if not values:
    return b''
  packed = b''
  for v in values:
    packed += _encode_varint(v)
  return _encode_bytes_field(field_number, packed)


def _encode_submessage_field(field_number: int, data: bytes) -> bytes:  # pylint: disable=invalid-name
  """Encode a submessage field."""
  return _encode_bytes_field(field_number, data)


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:  # pylint: disable=invalid-name
  """Decode a varint from bytes starting at pos. Returns (value, new_pos)."""
  result = 0
  shift = 0
  while True:
    b = data[pos]
    result |= (b & 0x7F) << shift
    pos += 1
    if not (b & 0x80):
      break
    shift += 7
  return result, pos


@dataclasses.dataclass
class PychunkedgraphEdgesMsg:
  """Matches ExportPychunkedgraphRequest.PychunkedgraphEdgesMsg."""

  # Contains n uint64le values for node endpoint 1.
  node_ids1: bytes = b''
  # Contains n uint64le values for node endpoint 2.
  node_ids2: bytes = b''
  # Contains n float32le values for affinities.
  affinities: bytes = b''
  # Contains n uint64le values for areas (or empty).
  areas: bytes = b''

  def SerializeToString(self) -> bytes:
    """Serialize to protobuf wire format."""
    result = b''
    if self.node_ids1:
      result += _encode_bytes_field(1, self.node_ids1)
    if self.node_ids2:
      result += _encode_bytes_field(2, self.node_ids2)
    if self.affinities:
      result += _encode_bytes_field(3, self.affinities)
    if self.areas:
      result += _encode_bytes_field(4, self.areas)
    return result

  def CopyFrom(self, other: 'PychunkedgraphEdgesMsg'):
    self.node_ids1 = other.node_ids1
    self.node_ids2 = other.node_ids2
    self.affinities = other.affinities
    self.areas = other.areas

  @classmethod
  def FromString(cls, data: bytes) -> 'PychunkedgraphEdgesMsg':
    """Parse from protobuf wire format."""
    msg = cls()
    pos = 0
    while pos < len(data):
      tag, pos = _decode_varint(data, pos)
      field_number = tag >> 3
      wire_type = tag & 0x07
      if wire_type == 2:  # Length-delimited
        length, pos = _decode_varint(data, pos)
        field_data = data[pos : pos + length]
        pos += length
        if field_number == 1:
          msg.node_ids1 = field_data
        elif field_number == 2:
          msg.node_ids2 = field_data
        elif field_number == 3:
          msg.affinities = field_data
        elif field_number == 4:
          msg.areas = field_data
      else:
        raise ValueError(f'Unexpected wire type {wire_type}')
    return msg

  def ParseFromString(self, data: bytes):
    """Parse from serialized protobuf bytes."""
    parsed = PychunkedgraphEdgesMsg.FromString(data)
    self.CopyFrom(parsed)


@dataclasses.dataclass
class PychunkedgraphChunkEdgesMsg:
  """Matches ExportPychunkedgraphRequest.PychunkedgraphChunkEdgesMsg."""

  in_chunk: PychunkedgraphEdgesMsg = dataclasses.field(
      default_factory=PychunkedgraphEdgesMsg
  )
  cross_chunk: PychunkedgraphEdgesMsg = dataclasses.field(
      default_factory=PychunkedgraphEdgesMsg
  )
  between_chunk: PychunkedgraphEdgesMsg = dataclasses.field(
      default_factory=PychunkedgraphEdgesMsg
  )

  def SerializeToString(self) -> bytes:
    result = b''
    in_chunk_data = self.in_chunk.SerializeToString()
    if in_chunk_data:
      result += _encode_submessage_field(1, in_chunk_data)
    cross_chunk_data = self.cross_chunk.SerializeToString()
    if cross_chunk_data:
      result += _encode_submessage_field(2, cross_chunk_data)
    between_chunk_data = self.between_chunk.SerializeToString()
    if between_chunk_data:
      result += _encode_submessage_field(3, between_chunk_data)
    return result

  @classmethod
  def FromString(cls, data: bytes) -> 'PychunkedgraphChunkEdgesMsg':
    """Parse from serialized protobuf bytes."""
    msg = cls()
    pos = 0
    while pos < len(data):
      tag, pos = _decode_varint(data, pos)
      field_number = tag >> 3
      wire_type = tag & 0x07
      if wire_type == 2:
        length, pos = _decode_varint(data, pos)
        field_data = data[pos : pos + length]
        pos += length
        if field_number == 1:
          msg.in_chunk = PychunkedgraphEdgesMsg.FromString(field_data)
        elif field_number == 2:
          msg.cross_chunk = PychunkedgraphEdgesMsg.FromString(field_data)
        elif field_number == 3:
          msg.between_chunk = PychunkedgraphEdgesMsg.FromString(field_data)
      else:
        raise ValueError(f'Unexpected wire type {wire_type}')
    return msg

  def ParseFromString(self, data: bytes):
    """Parse from serialized protobuf bytes."""
    parsed = PychunkedgraphChunkEdgesMsg.FromString(data)
    self.in_chunk = parsed.in_chunk
    self.cross_chunk = parsed.cross_chunk
    self.between_chunk = parsed.between_chunk


@dataclasses.dataclass
class PychunkedgraphChunkComponentsMsg:
  """Matches ExportPychunkedgraphRequest.PychunkedgraphChunkComponentsMsg."""

  components: list[int] = dataclasses.field(default_factory=list)

  def SerializeToString(self) -> bytes:
    return _encode_packed_uint64_field(1, self.components)

  @classmethod
  def FromString(cls, data: bytes) -> 'PychunkedgraphChunkComponentsMsg':
    """Parse from serialized protobuf bytes."""
    msg = cls()
    pos = 0
    while pos < len(data):
      tag, pos = _decode_varint(data, pos)
      field_number = tag >> 3
      wire_type = tag & 0x07
      if wire_type == 2 and field_number == 1:  # Packed repeated
        length, pos = _decode_varint(data, pos)
        end = pos + length
        while pos < end:
          value, pos = _decode_varint(data, pos)
          msg.components.append(value)
      elif wire_type == 0 and field_number == 1:  # Non-packed varint
        value, pos = _decode_varint(data, pos)
        msg.components.append(value)
      else:
        raise ValueError(
            f'Unexpected wire type {wire_type} for field {field_number}'
        )
    return msg

  def ParseFromString(self, data: bytes):
    """Parse from serialized protobuf bytes."""
    parsed = PychunkedgraphChunkComponentsMsg.FromString(data)
    self.components = parsed.components


@dataclasses.dataclass
class Vector3j:
  """Matches proto.Vector3j."""

  x: int = 0
  y: int = 0
  z: int = 0


@dataclasses.dataclass
class BigquerySource:
  """Matches ExportPychunkedgraphRequest.BigquerySource."""

  project: str = ''
  dataset: str = ''
  table: str = ''


@dataclasses.dataclass
class ExportPychunkedgraphRequest:
  """Pure-Python equivalent of the ExportPychunkedgraphRequest proto.

  This provides a compatible API for configuration purposes. The nested
  message types for wire-format serialization are defined above.
  """

  input_tensorstore_spec: str = ''
  pychunkedgraph_block_shape: Vector3j = dataclasses.field(
      default_factory=Vector3j
  )
  output_pychunkedgraph_kvstore: str = ''
  chosen_edges_table: BigquerySource | None = None
  output_dropped_edges_path: str = ''

  # Nested message types for wire format compatibility.
  PychunkedgraphEdgesMsg = PychunkedgraphEdgesMsg  # pylint: disable=invalid-name
  PychunkedgraphChunkEdgesMsg = PychunkedgraphChunkEdgesMsg  # pylint: disable=invalid-name
  PychunkedgraphChunkComponentsMsg = PychunkedgraphChunkComponentsMsg  # pylint: disable=invalid-name

  def HasField(self, field_name: str) -> bool:
    if field_name == 'chosen_edges_table':
      return self.chosen_edges_table is not None
    return bool(getattr(self, field_name, None))

  @classmethod
  def from_json(cls, json_str: str) -> 'ExportPychunkedgraphRequest':  # pylint: disable=invalid-name
    """Parse from JSON string."""
    d = json.loads(json_str)
    request = cls()
    request.input_tensorstore_spec = d.get('input_tensorstore_spec', '')
    if 'pychunkedgraph_block_shape' in d:
      bs = d['pychunkedgraph_block_shape']
      request.pychunkedgraph_block_shape = Vector3j(
          x=bs.get('x', 0), y=bs.get('y', 0), z=bs.get('z', 0)
      )
    request.output_pychunkedgraph_kvstore = d.get(
        'output_pychunkedgraph_kvstore', ''
    )
    request.output_dropped_edges_path = d.get('output_dropped_edges_path', '')
    if 'chosen_edges_table' in d:
      ct = d['chosen_edges_table']
      request.chosen_edges_table = BigquerySource(
          project=ct.get('project', ''),
          dataset=ct.get('dataset', ''),
          table=ct.get('table', ''),
      )
    return request
