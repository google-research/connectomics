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
"""Tests for metadata."""

import pathlib

from absl import flags
from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.volume import metadata
import numpy as np


FLAGS = flags.FLAGS

BBOX = bounding_box.BoundingBox


class VolumeMetadataTest(absltest.TestCase):

  def test_volume_metadata(self):
    meta = metadata.VolumeMetadata(
        volume_size=(100, 100, 100),
        pixel_size=(8, 8, 30),
        bounding_boxes=[BBOX([10, 10, 10], [100, 100, 100])],
    )

    # No scale
    scaled = meta.scale([1, 1, 1])
    self.assertCountEqual(scaled.volume_size, [100, 100, 100])
    self.assertCountEqual(scaled.pixel_size, [8, 8, 30])
    self.assertEqual(
        scaled.bounding_boxes, [BBOX([10, 10, 10], [100, 100, 100])]
    )

    # Scale up
    scaled = meta.scale([2, 2, 2])
    self.assertCountEqual(scaled.volume_size, [200, 200, 200])
    self.assertCountEqual(scaled.pixel_size, [4, 4, 15])
    self.assertEqual(
        scaled.bounding_boxes, [BBOX([20, 20, 20], [200, 200, 200])]
    )
    scaled = meta.scale(2)
    self.assertCountEqual(scaled.volume_size, [200, 200, 200])
    self.assertCountEqual(scaled.pixel_size, [4, 4, 15])
    self.assertEqual(
        scaled.bounding_boxes, [BBOX([20, 20, 20], [200, 200, 200])]
    )

    # Scale down
    scaled = meta.scale([0.5, 0.5, 0.5])
    self.assertCountEqual(scaled.volume_size, [50, 50, 50])
    self.assertCountEqual(scaled.pixel_size, [16, 16, 60])
    self.assertEqual(scaled.bounding_boxes, [BBOX([5, 5, 5], [50, 50, 50])])

    # Non uniform scale
    scaled = meta.scale([2, 3, 0.5])
    self.assertCountEqual(scaled.volume_size, [200, 300, 50])
    self.assertCountEqual(scaled.pixel_size, [4, 2.6666666666666665, 60])
    self.assertEqual(scaled.bounding_boxes, [BBOX([20, 30, 5], [200, 300, 50])])

    # Scale xy
    scaled = meta.scale_xy(2)
    self.assertCountEqual(scaled.volume_size, [200, 200, 100])
    self.assertCountEqual(scaled.pixel_size, [4, 4, 30])
    self.assertEqual(
        scaled.bounding_boxes, [BBOX([20, 20, 10], [200, 200, 100])]
    )

  def test_volume_save_metadata(self):
    meta = metadata.VolumeMetadata(
        volume_size=(100, 100, 100),
        pixel_size=(8, 8, 30),
        bounding_boxes=[BBOX([10, 10, 10], [100, 100, 100])],
        dtype=np.uint64,
    )
    temp_path = pathlib.Path(self.create_tempdir().full_path)
    vol = metadata.Volume(path=temp_path / 'foo.volinfo', meta=meta)
    vol.save_metadata()

    target_path = temp_path / 'foo.metadata.json'
    self.assertTrue(target_path.exists())
    file = open(target_path, 'r')
    self.assertEqual(metadata.VolumeMetadata.from_json(file.read()), vol.meta)
    file.close()


if __name__ == '__main__':
  absltest.main()
