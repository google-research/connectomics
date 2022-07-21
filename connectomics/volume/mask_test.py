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
"""Tests for mask.py."""

from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.volume import mask as m
import numpy as np


class MaskTest(absltest.TestCase):

  def test_build_mask(self):
    subvol_size = (150, 50, 75)
    image = np.random.random(subvol_size)

    mask_config = m.MaskConfig.from_dict({
        'image': {
            'channels': [{
                'channel': 0,
                'min_value': 0.5,
                'max_value': 1.0,
            }],
        }
    })
    chan_config = mask_config.image.channels[0]

    box = bounding_box.BoundingBox(start=(0, 0, 0), size=subvol_size[::-1])

    mask = m.build_mask([mask_config], box, image=image)

    np.testing.assert_array_equal(mask, image >= 0.5)

    # Test with an int image, and masking specific values only.
    image = np.random.randint(0, 10, subvol_size, dtype=np.uint8)
    chan_config.values = [1, 5, 8]

    mask = m.build_mask([mask_config], box, image=image)

    np.testing.assert_array_equal(mask,
                                  (image == 1) | (image == 5) | (image == 8))


if __name__ == '__main__':
  absltest.main()
