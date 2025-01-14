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

    mask = m.build_mask(
        [mask_config],
        box,
        decorated_volume_loader=lambda x: x,
        image=image,
    )

    np.testing.assert_array_equal(mask, image >= 0.5)

    # Test with an int image, and masking specific values only.
    image = np.random.randint(0, 10, subvol_size, dtype=np.uint8)
    chan_config.values = [1, 5, 8]

    self.called = False
    mask = m.build_mask(
        [mask_config],
        box,
        decorated_volume_loader=lambda x: x,
        image=image,
    )

    np.testing.assert_array_equal(
        mask, (image == 1) | (image == 5) | (image == 8)
    )


def test_build_mask_with_padding(self):
  subvol_size = (150, 50, 75)
  smaller_size = (149, 49, 75)

  smaller_mask = np.random.choice([True, False], size=smaller_size)
  image = np.random.choice(
      [0, 1], size=subvol_size
  )

  mask_config = m.MaskConfig.from_dict({
      'image': {
          'channels': [{
              'channel': 0,
              'min_value': 0,
              'max_value': 1,
          }],
      }
  })

  box = bounding_box.BoundingBox(start=(0, 0, 0), size=subvol_size[::-1])

  def mock_volume_loader(volume):
    del volume  # Unused
    return smaller_mask

  mask = m.build_mask(
      [mask_config],
      box,
      decorated_volume_loader=mock_volume_loader,
      image=image,
  )

  # Ensure the resulting mask matches the target dimensions
  self.assertEqual(mask.shape, subvol_size)

  # Verify that the padded regions are zeros in the final mask
  padded_image = np.zeros(subvol_size, dtype=bool)
  padded_image[: smaller_size[0], : smaller_size[1], : smaller_size[2]] = (
      smaller_mask
  )
  np.testing.assert_array_equal(mask, padded_image)


if __name__ == '__main__':
  absltest.main()
