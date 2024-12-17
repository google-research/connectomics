# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
from absl.testing import absltest
from connectomics.jax import grain_util
import numpy as np


class GrainUtilTest(absltest.TestCase):

  def test_clip_values(self):
    array = np.array([[0, 0, 0], [255, 255, 255]])
    expected = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    np.testing.assert_allclose(
        grain_util.ClipValues().map({'x': array})['x'], expected
    )

  def test_expand_dims(self):
    array = np.array([[0, 0, 0], [255, 255, 255]])
    expected = array[np.newaxis, ...]
    np.testing.assert_allclose(
        grain_util.ExpandDims(axis=0).map({'x': array})['x'], expected
    )

  def test_pad_values(self):
    array = np.array([[0, 0, 0], [255, 255, 255]])
    expected = np.array([[0, 0, 0, 0, 0], [0, 255, 255, 255, 0]])
    np.testing.assert_allclose(
        grain_util.PadValues(
            pad_width=((0, 0), (1, 1))).map({'x': array})['x'], expected
    )

  def test_rescale_values(self):
    array = np.array([[0, 0, 0], [255, 255, 255]])
    expected = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    np.testing.assert_allclose(
        grain_util.RescaleValues().map({'x': array})['x'], expected
    )

  def test_reshape_values(self):
    array = np.zeros((2, 4, 8))
    shape = (-1, 8)
    np.testing.assert_allclose(
        grain_util.ReshapeValues(newshape=shape).map({'x': array})['x'],
        array.reshape(shape)
    )

  def test_shift_and_divide_values(self):
    array = np.array([[0, 0, 0], [255, 255, 255]])
    expected = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    np.testing.assert_allclose(
        grain_util.ShiftAndDivideValues(
            divisor=255.0).map({'x': array})['x'], expected
    )

  def test_transpose_values(self):
    array = np.zeros((2, 4, 8))
    axis = (2, 1, 0)
    np.testing.assert_allclose(
        grain_util.TransposeValues(axis=axis).map({'x': array})['x'],
        array.transpose(axis)
    )

  def test_all_ops(self):
    all_ops = sum(map(grain_util.get_all_ops,
                      ['connectomics.jax.grain_util']), [])
    assert len(all_ops) >= 3

  def test_parse(self):
    array = np.array([[0, 0, 0], [255, 255, 255]])
    expected = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=np.float32)

    all_ops = sum(map(grain_util.get_all_ops,
                      ['connectomics.jax.grain_util']), [])
    transformations = grain_util.parse(
        'clip_values()|shift_and_divide_values(divisor=2.)', all_ops)
    assert len(transformations) == 2

    res = {'x': np.copy(array)}
    for op in transformations:
      res = op.map(res)
    np.testing.assert_allclose(res['x'], expected)

    transformations_with_extra_pipe = grain_util.parse(
        'clip_values()|shift_and_divide_values(divisor=2.)|', all_ops)
    assert len(transformations_with_extra_pipe) == 2


if __name__ == '__main__':
  absltest.main()
