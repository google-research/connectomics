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
"""Tests for file."""

import dataclasses
import os
from typing import Optional

from absl import flags
from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.common import file
from connectomics.volume import tensorstore as tsv
import dataclasses_json

FLAGS = flags.FLAGS

BBox = bounding_box.BoundingBox


@dataclasses.dataclass(frozen=True)
class TestDataClass(dataclasses_json.DataClassJsonMixin):
  a: int
  b: str
  c: float


@dataclasses.dataclass(frozen=True)
class AnotherTestDataClass(dataclasses_json.DataClassJsonMixin):
  a: int
  b: str
  c: float
  inner: Optional[TestDataClass] = None


class FileTest(absltest.TestCase):

  def test_load_dataclass_json(self):
    a = TestDataClass(a=1, b='foo', c=1.0)
    fname = os.path.join(FLAGS.test_tmpdir, 'dc_file')
    file.save_dataclass_json(a, fname)
    b = file.load_dataclass_json(TestDataClass, fname)
    self.assertEqual(a, b)

    a = AnotherTestDataClass(a=1, b='foo', c=1.0, inner=a)
    file.save_dataclass_json(a, fname)

    inner = file.load_dataclass_json(TestDataClass, fname, '/inner')
    self.assertEqual(inner, a.inner)

  def test_dataclass_from_serialized(self):
    a = TestDataClass(a=1, b='foo', c=1.0)
    fname = os.path.join(FLAGS.test_tmpdir, 'dc_file')
    file.save_dataclass_json(a, fname)
    b = file.dataclass_from_serialized(TestDataClass, fname)
    self.assertEqual(a, b)
    c = file.dataclass_from_serialized(TestDataClass, a.to_json())
    self.assertEqual(a, c)

  def test_dataclass_from_instance(self):
    ts_conf = tsv.TensorstoreConfig.from_dict({
        'spec': {
            'test': 'foo',
        },
        'metadata': {
            'voxel_size': [1, 2, 3],
            'bounding_boxes': [{
                'start': [1, 2, 3],
                'size': [4, 5, 6],
            }],
        },
    })
    new_conf = file.load_dataclass(tsv.TensorstoreConfig, ts_conf)
    self.assertIs(new_conf, ts_conf)

  def test_dataclass_from_dict(self):
    ts_conf = file.load_dataclass(
        tsv.TensorstoreConfig,
        {
            'spec': {
                'test': 'foo',
            },
            'metadata': {
                'voxel_size': [1, 2, 3],
                'bounding_boxes': [{
                    'start': [1, 2, 3],
                    'size': [4, 5, 6],
                }],
            },
        },
    )
    self.assertIsNotNone(ts_conf)
    self.assertEqual(ts_conf.spec, {'test': 'foo'})
    self.assertEqual(ts_conf.metadata.voxel_size, (1, 2, 3))
    self.assertLen(ts_conf.metadata.bounding_boxes, 1)
    self.assertEqual(
        ts_conf.metadata.bounding_boxes[0], BBox([1, 2, 3], [4, 5, 6])
    )

  def test_dataclass_from_json(self):
    ts_conf = file.load_dataclass(
        tsv.TensorstoreConfig,
        """{
      "spec": {
        "test": "foo"
      },
      "metadata": {
        "voxel_size": [1,2,3],
        "bounding_boxes": [{
          "start": [1,2,3],
          "size": [4,5,6]
        }]
      } 
    }""",
    )
    self.assertIsNotNone(ts_conf)
    self.assertEqual(ts_conf.spec, {'test': 'foo'})
    self.assertEqual(ts_conf.metadata.voxel_size, (1, 2, 3))
    self.assertLen(ts_conf.metadata.bounding_boxes, 1)
    self.assertEqual(
        ts_conf.metadata.bounding_boxes[0], BBox([1, 2, 3], [4, 5, 6])
    )

  def test_dataclass_from_file(self):
    fname = os.path.join(FLAGS.test_tmpdir, 'dc_file')
    with file.Open(fname, 'w') as f:
      f.write("""{
        "spec": {
          "test": "foo"
        },
        "metadata": {
          "voxel_size": [1,2,3],
          "bounding_boxes": [{
            "start": [1,2,3],
            "size": [4,5,6]
          }]
        } 
      }""")
    ts_conf = file.load_dataclass(tsv.TensorstoreConfig, fname)
    self.assertIsNotNone(ts_conf)
    self.assertEqual(ts_conf.spec, {'test': 'foo'})
    self.assertEqual(ts_conf.metadata.voxel_size, (1, 2, 3))
    self.assertLen(ts_conf.metadata.bounding_boxes, 1)
    self.assertEqual(
        ts_conf.metadata.bounding_boxes[0], BBox([1, 2, 3], [4, 5, 6])
    )

  def test_dataclass_loader(self):
    ts_conf = tsv.TensorstoreConfig.from_dict({
        'spec': {
            'test': 'foo',
        },
        'metadata': {
            'voxel_size': [1, 2, 3],
            'bounding_boxes': [{
                'start': [1, 2, 3],
                'size': [4, 5, 6],
            }],
        },
    })

    loader = file.dataclass_loader(tsv.TensorstoreConfig)

    new_conf = loader(ts_conf)
    self.assertIs(new_conf, ts_conf)

    new_conf = loader(ts_conf.to_json())
    self.assertEqual(new_conf, ts_conf)

    fname = os.path.join(FLAGS.test_tmpdir, 'dc_loader')
    with file.Open(fname, 'w') as f:
      f.write(ts_conf.to_json())
    new_conf = loader(fname)
    self.assertEqual(new_conf, ts_conf)


if __name__ == '__main__':
  absltest.main()
