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
"""Tests for descriptor."""

import dataclasses
import os

from absl import flags
from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.volume import descriptor
from connectomics.volume import tensorstore as tsv

FLAGS = flags.FLAGS

BBOX = bounding_box.BoundingBox


class DescriptorTest(absltest.TestCase):

  def test_volume_descriptor_from_string(self):
    desc: descriptor.VolumeDescriptor = descriptor.VolumeDescriptor.from_json(
        """
    {
      "decorator_specs": [],
      "tensorstore_config": {
        "spec": {
          "driver": "n5",
          "kvstore": {
            "driver": "file",
            "path": "/path/to/a/volume"
          }
        },
        "metadata": {
          "voxel_size": [
            8,
            8,
            33
          ],
          "bounding_boxes": [
            {
              "start": [
                0,
                0,
                0
              ],
              "size": [
                1000,
                1000,
                1000
              ],
              "is_border_start": [
                false,
                false,
                false
              ],
              "is_border_end": [
                false,
                false,
                false
              ],
              "type": "BoundingBox"
            }
          ]
        }
      },
      "volinfo": null
    }
    """)

    self.assertIsInstance(desc, descriptor.VolumeDescriptor)
    self.assertListEqual(desc.decorator_specs, [])

    self.assertIsInstance(desc.tensorstore_config, tsv.TensorstoreConfig)
    self.assertIsInstance(desc.tensorstore_config.spec, dict)
    self.assertIsInstance(desc.tensorstore_config.metadata,
                          tsv.TensorstoreMetadata)
    self.assertListEqual(desc.tensorstore_config.metadata.bounding_boxes,
                         [BBOX([0, 0, 0], [1000, 1000, 1000])])

  def test_volume_descriptor_with_ts_config_as_path(self):
    spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': '/path/to/a/volume',
        }
    }
    metadata = tsv.TensorstoreMetadata(
        (8, 8, 33),
        [bounding_box.BoundingBox(start=[0, 0, 0], size=[1000, 1000, 1000])])

    expected_desc = tsv.TensorstoreConfig(spec, metadata)

    tmp_config_file = os.path.join(FLAGS.test_tmpdir, 'config.json')
    with open(tmp_config_file, 'w') as f:
      f.write(expected_desc.to_json())

    expected_desc = descriptor.VolumeDescriptor(
        decorator_specs=[], tensorstore_config=expected_desc)
    original_desc = dataclasses.replace(  # pytype: disable=wrong-arg-types  # dataclasses-replace-types
        expected_desc, tensorstore_config=tmp_config_file)

    desc = descriptor.VolumeDescriptor.from_json(original_desc.to_json())

    self.assertEqual(desc, expected_desc)

  def test_volume_descriptor_with_metadata_as_path(self):
    spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': '/path/to/a/volume',
        }
    }
    expected_metadata = tsv.TensorstoreMetadata(
        (8, 8, 33),
        [bounding_box.BoundingBox(start=[0, 0, 0], size=[1000, 1000, 1000])])

    tmp_metadata_file = os.path.join(FLAGS.test_tmpdir, 'metadata.json')
    with open(tmp_metadata_file, 'w') as f:
      f.write(expected_metadata.to_json())

    original_desc = descriptor.VolumeDescriptor(
        decorator_specs=[],
        tensorstore_config=tsv.TensorstoreConfig(  # pytype: disable=wrong-arg-types
            spec=spec, metadata=tmp_metadata_file))

    desc = descriptor.VolumeDescriptor.from_json(original_desc.to_json())
    expected_desc = dataclasses.replace(
        original_desc,
        tensorstore_config=tsv.TensorstoreConfig(
            spec=spec, metadata=expected_metadata))

    self.assertEqual(desc, expected_desc)

  def test_volume_descriptor_with_chained_paths(self):
    spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': '/path/to/a/volume',
        }
    }
    expected_metadata = tsv.TensorstoreMetadata(
        (8, 8, 33),
        [bounding_box.BoundingBox(start=[0, 0, 0], size=[1000, 1000, 1000])])

    tmp_metadata_file = os.path.join(FLAGS.test_tmpdir, 'metadata.json')
    with open(tmp_metadata_file, 'w') as f:
      f.write(expected_metadata.to_json())

    tmp_config_file = os.path.join(FLAGS.test_tmpdir, 'config.json')
    intermediate_config = tsv.TensorstoreConfig(  # pytype: disable=wrong-arg-types
        spec=spec, metadata=tmp_metadata_file)
    with open(tmp_config_file, 'w') as f:
      f.write(intermediate_config.to_json())

    original_desc = descriptor.VolumeDescriptor(  # pytype: disable=wrong-arg-types
        decorator_specs=[], tensorstore_config=tmp_config_file)

    desc = descriptor.VolumeDescriptor.from_json(original_desc.to_json())
    expected_desc = descriptor.VolumeDescriptor(
        decorator_specs=[],
        tensorstore_config=tsv.TensorstoreConfig(
            spec=spec, metadata=expected_metadata))

    self.assertEqual(desc, expected_desc)


if __name__ == '__main__':
  absltest.main()
