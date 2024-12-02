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
"""Tests for decorators."""

import copy
import json

from absl.testing import absltest
from connectomics.volume import decorators
import numpy as np
import scipy.ndimage
import tensorstore as ts


class DecoratorsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._data = ts.open({
        'driver': 'n5',
        'kvstore': {
            'driver': 'memory',
        },
        'metadata': {
            'dataType': 'float64',
            'dimensions': (10, 10, 10),
            'blockSize': (1, 1, 1),
            'axes': ('x', 'y', 'z'),
        },
        'create': True,
        'delete_existing': True,
    }).result()
    rng = np.random.default_rng(seed=42)
    self._data[...] = np.array(
        rng.uniform(size=self._data.schema.shape), dtype=np.float64)

  def test_cast(self):
    dec = decorators.Cast(dtype='uint16', rescale=False)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        np.array(self._data).astype(np.uint16))

    dec = decorators.Cast(dtype='uint16', rescale=True)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        decorators._cast_img(np.array(self._data), dtype='uint16'))

    np.testing.assert_equal(
        decorators._cast_img(
            np.array([0, 255], dtype='uint8'), dtype='bool'),
        np.array([False, True], dtype=bool))
    np.testing.assert_equal(
        decorators._cast_img(
            np.array([0, 255], dtype='uint8'), dtype='float32'),
        np.array([0., 1.], dtype=np.float32))
    np.testing.assert_equal(
        decorators._cast_img(
            np.array([0, 255], dtype='uint8'), dtype='int16'),
        np.array([0, 32767], dtype=np.int16))
    np.testing.assert_equal(
        decorators._cast_img(
            np.array([0., 1.], dtype='float32'), dtype='uint8'),
        np.array([0, 255], dtype=np.uint8))
    np.testing.assert_equal(
        decorators._cast_img(
            np.array([0., 1.], dtype='float32'), dtype='uint16'),
        np.array([0, 65535], dtype=np.uint16))

  def test_active_contours_mask_filter(self):
    dec = decorators.ActiveContoursMaskFilter(
        min_chunksize=(10, 10, 10))
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        decorators._active_contours_mask(np.array(self._data)))

  def test_exposure_filter(self):
    filter_args = {'mode': 'rescale_intensity', 'cast_float64': False}
    dec = decorators.ExposureFilter(
        min_chunksize=(10, 10, 10), **filter_args)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        decorators._exposure_filter(np.array(self._data), **filter_args))

  def test_clahe_filter(self):
    filter_args = {'kernel_size': (10, 10, 2)}
    dec = decorators.CLAHEFilter(
        min_chunksize=(10, 10, 10), **filter_args)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        decorators._exposure_filter(
            np.array(self._data), mode='equalize_adapthist', cast_float64=True,
            **filter_args))

  def test_clahe_filter_with_overlap(self):
    # NOTE: CLAHE does intensity rescaling on the input image using its min/max.
    # Results will thus only be equal if the min/max of chunks corresponds to
    # the min/max of the image; we construct an image for which this holds.
    dim_x, dim_y = 10, 10
    data = ts.open({
        'driver': 'n5',
        'kvstore': {
            'driver': 'memory',
        },
        'metadata': {
            'dataType': 'float64',
            'dimensions': (dim_x, dim_y),
            'blockSize': (dim_x, 1),
            'axes': ('x', 'y'),
        },
        'create': True,
        'delete_existing': True,
    }).result()
    single_slice = np.arange(dim_x, dtype=np.float64).reshape(-1, 1) / dim_x
    data[...] = np.repeat(single_slice, dim_y, axis=1)
    filter_args = {'kernel_size': (3, 3)}
    dec = decorators.CLAHEFilter(
        min_chunksize=(dim_x, 1),
        overlap_width=((3, 3), (3, 3)),
        **filter_args)
    vc = dec.decorate(data)
    np.testing.assert_equal(
        vc[...].read().result(),
        decorators._exposure_filter(
            np.array(data), mode='equalize_adapthist', cast_float64=True,
            **filter_args))

  def test_clip_filter(self):
    filter_args = {'a_min': 0.5, 'a_max': None}
    dec = decorators.ClipFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    res = vc[...].read().result()
    np.testing.assert_equal(
        res, np.clip(np.array(self._data), **filter_args))
    self.assertTrue(np.any(np.not_equal(res, self._data)))

  def test_gaussian_filter(self):
    filter_args = {'sigma': [1.] * self._data.ndim}
    dec = decorators.GaussianFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        scipy.ndimage.gaussian_filter(self._data[...], **filter_args))

  def test_label_filter(self):
    dec = decorators.LabelFilter(
        min_chunksize=self._data.shape)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        decorators._label_filter(np.array(self._data)))

  def test_log1p_filter(self):
    dec = decorators.Log1pFilter(
        min_chunksize=self._data.shape)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(vc[...].read().result(), np.log1p(self._data[...]))

  def test_median_filter(self):
    filter_args = {'size': [3] * self._data.ndim}
    dec = decorators.MedianFilter(min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        scipy.ndimage.median_filter(self._data[...], **filter_args))

  def test_median_filter_with_overlap(self):
    filter_args = {'size': [3] * self._data.ndim, 'mode': 'reflect'}
    dec = decorators.MedianFilter(
        min_chunksize=(3, 3, 3),
        overlap_width=((1, 1), (1, 1), (1, 1)),
        **filter_args)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        scipy.ndimage.median_filter(self._data[...], **filter_args))

  def test_min_sum_filter(self):
    data_sum = self._data[...].read().result().sum()

    dec = decorators.MinSumFilter(
        min_sum=data_sum,
        min_chunksize=self._data.shape)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(vc[...].read().result(), self._data)

    dec = decorators.MinSumFilter(min_sum=data_sum + 1.)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(vc[...].read().result(), np.zeros_like(self._data))

  def test_nan_replace_filter(self):
    data_without_nans = np.ones((2, 2), dtype='float32')

    data_with_nans = data_without_nans.copy()
    data_with_nans[0, 0] = np.nan

    data_with_nans_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'memory'
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (2, 2),
            'axes': ('x', 'y'),
        },
        'create': True,
        'delete_existing': True,
    }
    data_with_nans_ts = ts.open(data_with_nans_spec).result()
    data_with_nans_ts[...] = data_with_nans

    dec = decorators.NaNReplaceFilter(nan_fill=1.)
    vc = dec.decorate(data_with_nans_ts)
    np.testing.assert_equal(vc[...].read().result(), data_without_nans)

  def test_peak_filter(self):
    filter_args = {'min_distance': 2, 'batch_dims': (2,)}
    dec = decorators.PeakFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    res = vc[...].read().result()
    assert res.sum() > 0
    np.testing.assert_equal(
        res, decorators._peak_filter_nd(np.array(self._data), **filter_args))

  def test_percentile_filter(self):
    filter_args = {'percentile': 0.9, 'size': [3] * self._data.ndim}
    dec = decorators.PercentileFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        scipy.ndimage.percentile_filter(self._data[...], **filter_args))

  def test_scale_filter(self):
    filter_args = {'factor': 0.5}
    dec = decorators.ScaleFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        filter_args['factor'] * self._data[...].read().result())

  def test_threshold_filter(self):
    filter_args = {'threshold': 0.25}
    dec = decorators.ThresholdFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    res = vc[...].read().result()
    np.testing.assert_equal(
        res, decorators._threshold(np.array(self._data), **filter_args))

  def test_standardize_filter(self):
    filter_args = {'mean': 5, 'std': 3}
    dec = decorators.StandardizeFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    res = vc[...].read().result()
    res_true = (np.array(self._data) - 5) / 3
    np.testing.assert_equal(res_true, res)

  def test_lowpass_filter(self):
    filter_args = {'cutoff_freq': 0.1, 'axis': -1}
    dec = decorators.LowpassFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    res = vc[...].read().result()
    res_true = decorators._lowpass_filter(np.array(self._data), **filter_args)
    np.testing.assert_equal(res_true, res)

  def test_zscore_filter(self):
    filter_args = {'axis': None}
    dec = decorators.ZScoreFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    np.testing.assert_equal(
        vc[...].read().result(),
        scipy.stats.zscore(self._data[...], **filter_args))

  def test_unsharp_mask_filter(self):
    filter_args = {'num_iterations': 2, 'amount': 0.3, 'radius': 2.}
    dec = decorators.UnsharpMaskFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    res = vc[...].read().result()
    np.testing.assert_equal(
        res, decorators._unsharp_mask(np.array(self._data), **filter_args))

  def test_max_projection(self):
    for projection_dim in (0, 1):
      dec = decorators.MaxProjection(projection_dim=projection_dim)
      vc = dec.decorate(self._data)
      np.testing.assert_almost_equal(
          vc[...].read().result(),
          np.max(self._data[...], axis=projection_dim))

  def test_mean_projection(self):
    for projection_dim in (0, 1):
      dec = decorators.MeanProjection(projection_dim=projection_dim)
      vc = dec.decorate(self._data)
      np.testing.assert_almost_equal(
          vc[...].read().result(),
          np.mean(self._data[...], axis=projection_dim))

  def test_sum_projection(self):
    for projection_dim in (0, 1):
      dec = decorators.SumProjection(projection_dim=projection_dim)
      vc = dec.decorate(self._data)
      np.testing.assert_almost_equal(
          vc[...].read().result(),
          np.sum(self._data[...], axis=projection_dim))

  def test_interpolation(self):
    data = np.array([
        [[0., 0.],
         [0., 0.]],
        [[10., 20.],
         [30., 40.]],], dtype=np.float32)
    data_ts = ts.open({
        'driver': 'n5',
        'kvstore': {
            'driver': 'memory'
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (2, 2, 2),
            'axes': ('x', 'y', 'z'),
        },
        'create': True,
        'delete_existing': True,
    }).result()
    data_ts[...] = data

    expected_res = np.array([
        [[0., 0.],
         [0., 0.]],
        [[10.//2, 20.//2],
         [30.//2, 40.//2]],
        [[10., 20.],
         [30., 40.]],])
    backends = ('scipy_map_coordinates', 'jax_map_coordinates', 'jax_resize')
    for backend in backends:
      kwargs = {'method': 'linear'} if backend == 'jax_resize' else {'order': 1}
      dec = decorators.Interpolation(
          size=(3, 2, 2), backend=backend, **kwargs
      )
      vc = dec.decorate(data_ts)
      np.testing.assert_equal(vc[...].read().result(), expected_res)

  def test_padding_interpolation(self):
    data = np.array([
        [1, 2],
        [3, 4]
    ]).astype(np.float32)
    data_ts = ts.open({
        'driver': 'n5',
        'kvstore': {'driver': 'memory'},
        'metadata': {
            'dataType': 'float32',
            'dimensions': (2, 2),
            'axes': ('x', 'y'),
        },
        'create': True,
        'delete_existing': True,
    }).result()
    data_ts[...] = data

    expected_res = np.zeros((4, 4)).astype(np.float32)
    expected_res[1:3, 1:3] = data
    dec = decorators.Interpolation(size=(4, 4), backend='pad',
                                   constant_values=0)
    vc = dec.decorate(data_ts)
    np.testing.assert_equal(vc[...].read().result(), expected_res)

  def test_multiply(self):
    mask = np.zeros_like(self._data, dtype='float32')

    multiply_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (10, 10, 10),
            'axes': ('x', 'y', 'z'),
        },
        'create': True,
        'delete_existing': True,
    }
    multiply_ts = ts.open(multiply_spec).result()
    multiply_ts[...].write(mask).result()

    multiply_spec['create'] = False
    multiply_spec['delete_existing'] = False
    multiply_spec['open'] = True

    dec = decorators.MultiplyPointwise(multiply_spec=multiply_spec)
    vc = dec.decorate(self._data)

    np.testing.assert_equal(
        vc[...].read().result(),
        mask)

  def test_interval_indexing_str_to_slice(self):
    parse_fn = decorators._interval_indexing_str_to_slice
    assert parse_fn(':') == slice(None, None)
    assert parse_fn('::') == slice(None, None, None)
    assert parse_fn('1') == slice(1)
    assert parse_fn('1:') == slice(1, None)
    assert parse_fn(':1') == slice(None, 1)
    assert parse_fn('1:2') == slice(1, 2)
    assert parse_fn('1:2:') == slice(1, 2, None)
    assert parse_fn(':1:2') == slice(None, 1, 2)
    assert parse_fn('1::5') == slice(1, None, 5)
    assert parse_fn('5::') == slice(5, None, None)
    assert parse_fn('::5') == slice(None, None, 5)
    assert parse_fn('1:2:3') == slice(1, 2, 3)
    assert parse_fn('0:1:5j') == slice(0, 1, 5j)

  def test_center_crop(self):
    test_3x3 = np.ones((3, 3))
    test_3x3[1, 1] = 2
    np.testing.assert_equal(decorators._center_crop(test_3x3, size=(1, 1)),
                            np.array([[2]]))
    np.testing.assert_equal(decorators._center_crop(test_3x3, size=(2, 2)),
                            np.array([[1, 1],
                                      [1, 2]]))

    test_4x4 = np.ones((4, 4))
    test_4x4[1:3, 1:3] = 2
    np.testing.assert_equal(decorators._center_crop(test_4x4, size=(1, 1)),
                            np.array([[2]]))
    np.testing.assert_equal(decorators._center_crop(test_4x4, size=(2, 2)),
                            np.array([[2, 2],
                                      [2, 2]]))
    np.testing.assert_equal(decorators._center_crop(test_4x4, size=(3, 3)),
                            np.array([[1, 1, 1],
                                      [1, 2, 2],
                                      [1, 2, 2]]))

  def test_objects_context(self):
    labelled_objects = np.zeros_like(self._data, dtype=np.uint64)
    labelled_objects[1, 1, 0] = 1
    labelled_objects[1, 2, 0] = 2
    labelled_objects[2, 1, 0] = 2
    labelled_objects[2, 2, 0] = 2
    labelled_objects[2, 2, 0] = 3
    num_objects = len(np.unique(labelled_objects))

    objects_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'uint64',
            'dimensions': (10, 10, 10),
            'axes': ('x', 'y', 'z'),
        },
        'create': True,
        'delete_existing': True,
    }
    objects_ts = ts.open(objects_spec).result()
    objects_ts[...].write(labelled_objects).result()

    objects_spec['create'] = False
    objects_spec['delete_existing'] = False
    objects_spec['open'] = True

    spec_overrides = {
        'schema': {
            'chunk_layout': {
                'grid_origin': [0, 0, 0, 0],
                'inner_order': [3, 2, 1, 0],
                'read_chunk': {'shape': [3, 3, 10, 1]},
                'write_chunk': {'shape': [3, 3, 10, 1]},
            },
            'domain': {
                'inclusive_min': [0, 0, 0, 0],
                'exclusive_max': [[3], [3], [10], [num_objects + 1]],
                'labels': ['x', 'y', 'z', 'o'],
            },
            'rank': 4,
        }
    }

    dec = decorators.ObjectsContext(
        width=[(1, 1), (1, 1), (0, 9)],
        objects_spec=objects_spec,
        spec_overrides=spec_overrides)
    vc = dec.decorate(self._data)

    np.testing.assert_equal(vc[..., 1].read().result(), self._data[0:3, 0:3, :])
    np.testing.assert_equal(vc[..., 2].read().result(), self._data[0:3, 0:3, :])
    np.testing.assert_equal(vc[..., 3].read().result(), self._data[1:4, 1:4, :])

    np.testing.assert_almost_equal(vc[..., 0].read().result(),
                                   np.zeros_like(self._data[0:3, 0:3, :]))

  def test_downsample(self):
    data = ts.open({
        'driver': 'n5',
        'kvstore': {
            'driver': 'memory',
        },
        'metadata': {
            'dataType': 'float64',
            'dimensions': (2, 3),
        },
        'create': True,
        'delete_existing': True,
    }).result()
    data[...] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

    dec = decorators.Downsample(downsample_factors=[1, 2], method='mean')
    vc = dec.decorate(data)
    np.testing.assert_equal(
        np.array([[1.5, 3], [4.5, 6]], dtype=np.float32),
        vc[...].read().result())

  def test_index_dim_by_ints(self):
    f = self.create_tempfile(content='[0, 1]')
    dec = decorators.IndexDimByInts(dim=0, json_path=f.full_path)
    vc = dec.decorate(self._data)
    res = vc.read().result()
    self.assertEqual(res.shape, (2, 10, 10), 'Shape mismatch.')

  def test_df_over_f(self):
    dtype = 'float32'

    f_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': dtype,
            'dimensions': (10,),
        },
        'create': True,
        'delete_existing': True,
    }
    f = ts.open(f_spec).result()
    f_data = np.array(np.random.uniform(size=f.schema.shape), dtype=dtype)
    f[...] = f_data

    f0_spec = f_spec
    f0_spec['kvstore']['path'] = self.create_tempdir().full_path
    f0 = ts.open(f0_spec).result()
    f0_data = np.array(np.random.uniform(size=f0.schema.shape), dtype=dtype)
    f0[...] = f0_data
    f0_spec['create'] = False
    f0_spec['delete_existing'] = False
    f0_spec['open'] = True

    baseline = 0.42
    df_over_f = (f_data - f0_data) / (f0_data - baseline)

    dec = decorators.DfOverF(f0_spec=f0_spec, baseline=baseline)
    vc = dec.decorate(f)
    np.testing.assert_equal(vc[...].read().result(), df_over_f)

  def test_evoked_response_filter_periodic(self):
    data = np.linspace(0, 9, 10).reshape(-1, 1)
    filter_args = {
        'condition_offsets': [0, 10],
        'condition_period_offsets': [[3, 3, 4],],
        'pad_side': 'left'
    }
    filtered = decorators._compute_average_evoked_response(
        data.copy(), **filter_args)
    # [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], pad left with nan, take average
    average_response = np.array([6, 10 / 3, 13 / 3, 16 / 3]).reshape(-1, 1)
    np.testing.assert_equal(filtered[-4:], data[-4:] - average_response)

  def test_evoked_response_filter_aperiodic(self):
    data = np.linspace(0, 9, 10).reshape(-1, 1)
    filter_args = {
        'condition_offsets': [0, 10],
        'condition_period_offsets': [[],],
    }
    average_response = data.mean()
    filtered = decorators._compute_average_evoked_response(
        data.copy(), **filter_args)
    np.testing.assert_equal(filtered, data - average_response)

  def test_evoked_response_filter_periodic_with_exclusive_max(self):
    data = np.linspace(0, 13, 14).reshape(-1, 1)
    filter_args = {
        'condition_offsets': [0, 14],
        'condition_period_offsets': [[3, 3, 4, 4],],
        'condition_baselines_exclusive_max': [-1,],
        'pad_side': 'left',
        'return_difference': False,
    }
    filtered = decorators._compute_average_evoked_response(
        data.copy(), **filter_args)
    # [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], pad left with nan, take average
    average_response = np.array([6, 10 / 3, 13 / 3, 16 / 3]).reshape(-1, 1)
    np.testing.assert_equal(filtered[-4:], average_response)

  def test_evoked_response_decorator(self):
    filter_args = {
        'condition_offsets': [0, 10],
        'condition_period_offsets': [[3, 3, 4],],
        'axis': 1,
        'pad_side': 'left'
    }
    dec = decorators.AverageEvokedResponseFilter(
        min_chunksize=self._data.shape, **filter_args)
    vc = dec.decorate(self._data)
    res = vc[...].read().result()
    np.testing.assert_equal(
        res,
        decorators._compute_average_evoked_response(
            np.array(self._data), **filter_args))


def get_written_tensorstores(
    multiscale_writer: decorators.MultiscaleWrite
) -> [ts.TensorStore]:
  tensorstores = []
  for dec in multiscale_writer._chain:
    if isinstance(dec, decorators.Writer):
      tensorstores.append(dec._output_ts)
  return tensorstores


class MultiscaleWriteTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._input_ts = ts.open({
        'driver': 'n5',
        'kvstore': {'driver': 'memory'},
        'metadata': {'dataType': 'float64', 'dimensions': (4, 4)},
        'schema': {'dimension_units': ['5um', '6nm'],
                   'domain': {'labels': ['x', 'y']}},
        'create': True,
    }).result()
    self._example_data = np.arange(16, dtype=np.float64).reshape((4, 4))
    self._input_ts[...] = self._example_data
    self._input_spec = dict(driver='n5', schema=self._input_ts.schema.to_json())
    self._ds_factors = [[2, 2], [2, 2]]  # 4 x 4 -> 2 x 2 -> 1 x 1
    self._ms_spec = dict(
        downsamplingFactors=[[1, 1], [2, 2], [4, 4]],
        axes=('x', 'y'),
        units=['um', 'nm'],
        resolution=[5, 6]
    )

  def test_multiscale_write(self):
    overwrite_base_spec = {
        'driver': 'n5', 'create': True, 'kvstore': {'driver': 'memory'},
    }
    dec = decorators.MultiscaleWrite(
        overwrite_base_spec, self._ds_factors, downsample_method='max')
    dec.initialize(self._input_spec, self._input_ts, dryrun=False)
    dec_ts = dec.decorate(self._input_ts)
    self.assertEqual(dec_ts.shape, (1, 1))
    self.assertEqual(dec_ts[0, 0].read().result(), np.max(self._example_data))
    written_ts = get_written_tensorstores(dec)
    self.assertEqual(written_ts[0].shape, self._input_ts.shape)
    self.assertEqual(written_ts[1].shape, (2, 2))
    self.assertEqual(written_ts[2].shape, (1, 1))
    true_units = [ts.Unit('20um'), ts.Unit('24nm')]
    for true_unit, ds_unit in zip(true_units, written_ts[2].dimension_units):
      self.assertEqual(true_unit, ds_unit)
    for mspec_key in ['axes', 'units', 'resolution', 'downsamplingFactors']:
      self.assertEqual(dec.multiscale_spec[mspec_key], self._ms_spec[mspec_key])

  def test_multiscale_write_to_zarr(self):
    overwrite_base_spec = {
        'driver': 'zarr',
        'create': True,
        'kvstore': {'driver': 'memory'},
        'schema': {
            'dimension_units': decorators.SpecAction.CLEAR,
            'codec': {
                'driver': 'zarr',
                'compression': decorators.SpecAction.CLEAR,
                'compressor': {
                    'id': 'blosc',
                    'cname': 'zstd',
                    'clevel': 9
                }
            }
        }
    }
    dec = decorators.MultiscaleWrite(
        overwrite_base_spec, self._ds_factors, downsample_method='max')
    dec.initialize(self._input_spec, self._input_ts, dryrun=False)
    dec_ts = dec.decorate(self._input_ts)
    self.assertEqual(dec_ts.shape, (1, 1))
    self.assertEqual(dec_ts[0, 0].read().result(), np.max(self._example_data))
    written_ts = get_written_tensorstores(dec)
    self.assertEqual(written_ts[1].shape, (2, 2))
    self.assertEqual(written_ts[2].shape, (1, 1))
    wspec = written_ts[0].schema.to_json()
    self.assertEqual(wspec['codec']['driver'], 'zarr')
    self.assertEqual(wspec['codec']['compressor']['clevel'], 9)
    for mspec_key in ['axes', 'units', 'resolution', 'downsamplingFactors']:
      self.assertEqual(dec.multiscale_spec[mspec_key], self._ms_spec[mspec_key])

  def test_multiscale_without_units(self):
    # when neither input nor override spec contains unit information
    input_ts = ts.open({
        'driver': 'n5',
        'kvstore': {'driver': 'memory'},
        'metadata': {'dataType': 'float64', 'dimensions': (4, 4)},
        'create': True
    }).result()
    input_ts[...] = self._example_data
    overwrite_base_spec = {
        'driver': 'n5', 'create': True, 'kvstore': {'driver': 'memory'},
    }
    input_spec = dict(driver='n5', schema=input_ts.schema.to_json())
    dec = decorators.MultiscaleWrite(
        overwrite_base_spec, self._ds_factors, downsample_method='max')
    dec.initialize(input_spec, input_ts, dryrun=False)
    self.assertNotIn('units', dec.multiscale_spec)
    self.assertNotIn('resolution', dec.multiscale_spec)


class MergeSpecsTest(absltest.TestCase):
  # Base is built from a decorated virtual_chunked and has TensorStore default
  # compression.
  _base = {
      'driver': 'n5',
      'schema': {
          'chunk_layout': {
              'grid_origin': [0, 0, 0, 0, 0],
              'inner_order': [4, 3, 2, 1, 0],
              'read_chunk': {
                  'shape': [256, 1, 1, 1024, 1]
              },
              'write_chunk': {
                  'shape': [256, 1, 1, 1024, 1]
              }
          },
          'codec': {
              'compression': {
                  'blockSize': 9,
                  'type': 'bzip2'
              },
              'driver': 'n5'
          },
          'domain': {
              'exclusive_max': [[2048], [1328], [72], [7879], [4]],
              'inclusive_min': [0, 0, 0, 0, 0]
          },
          'dtype': 'uint16',
          'rank': 5,
      }
  }

  # Manual override with Neuroglancer compatible compression.
  _overrides = {
      'schema': {
          'codec': {
              'compression': {
                  'blocksize': 0,
                  'clevel': 9,
                  'cname': 'zstd',
                  'shuffle': 2,
                  'type': 'blosc'
              },
          },
      }
  }

  # Input spec also has Neuroglancer compatible compression.
  _input_spec = {
      'schema': {
          'codec': {
              'compression': {
                  'blocksize': 0,
                  'clevel': 9,
                  'cname': 'zstd',
                  'shuffle': 2,
                  'type': 'blosc'
              },
          },
      }
  }

  def test_basic_merge(self):
    base = copy.deepcopy(self._base)
    overrides = copy.deepcopy(self._overrides)
    input_spec = copy.deepcopy(self._input_spec)
    output_spec = decorators._merge_specs(base, overrides, input_spec)

    # There is a problem, because the override is merged without clobbering
    # nested values, so blocksize fails to override blockSize.
    self.assertEqual(
        output_spec['schema']['codec']['compression'], {
            'blockSize': 9,
            'blocksize': 0,
            'clevel': 9,
            'cname': 'zstd',
            'shuffle': 2,
            'type': 'blosc'
        })

    # One solution is to explicitly clear the blockSize.
    overrides['schema']['codec']['compression'][
        'blockSize'] = decorators.SpecAction.CLEAR
    output_spec = decorators._merge_specs(base, overrides, input_spec)

    self.assertEqual(output_spec['schema']['codec']['compression'], {
        'blocksize': 0,
        'clevel': 9,
        'cname': 'zstd',
        'shuffle': 2,
        'type': 'blosc'
    })

  def test_clobber_with_input_spec(self):
    base = copy.deepcopy(self._base)
    overrides = copy.deepcopy(self._overrides)
    input_spec = copy.deepcopy(self._input_spec)

    # Sometimes the undecorated input has a setting we want to carry over. This
    # is often useful for compression settings.
    overrides['schema']['codec'] = decorators.SpecAction.CLOBBER_WITH_INPUT_SPEC
    output_spec = decorators._merge_specs(base, overrides, input_spec)
    self.assertEqual(output_spec['schema']['codec']['compression'], {
        'blocksize': 0,
        'clevel': 9,
        'cname': 'zstd',
        'shuffle': 2,
        'type': 'blosc'
    })


class TestDecorator(decorators.Decorator):

  def __init__(self, foo: int, bar: str):
    self.foo = foo
    self.bar = bar


class DecoratorSpecTest(absltest.TestCase):

  def test_decorator_args_unknown(self):
    expected_args = {
        'downsample_factors': [1, 2],
        'method': 'max',
    }
    args = decorators.DecoratorArgs.from_json(json.dumps(expected_args))
    self.assertEqual(args.values['downsample_factors'], [1, 2])
    self.assertEqual(args.values['method'], 'max')
    self.assertEqual(args.to_dict(), expected_args)

  def test_decorator_spec(self):
    expected_spec = {
        'name': 'Downsample',
    }
    spec = decorators.DecoratorSpec.from_json(json.dumps(expected_spec))
    self.assertEqual(spec.name, 'Downsample')
    self.assertIsNone(spec.args)
    self.assertIsNone(spec.package)

    expected_spec = {
        'name': 'Downsample',
        'args': {
            'downsample_factors': [1, 2],
            'method': 'max',
        },
        'package': 'foo.bar.baz',
    }
    spec = decorators.DecoratorSpec.from_json(json.dumps(expected_spec))
    args = decorators.DecoratorArgs.from_dict(expected_spec['args'])
    self.assertEqual(args.to_dict(), expected_spec['args'])
    self.assertEqual(spec.to_dict(), expected_spec)

  def test_build_decorator(self):
    spec = decorators.DecoratorSpec.from_json(
        json.dumps({
            'name': 'Downsample',
            'args': {
                'downsample_factors': [2, 4],
                'method': 'max',
            },
        })
    )
    decorator = decorators.build_decorator(spec)
    self.assertIsInstance(decorator, decorators.Downsample)
    self.assertEqual(decorator._downsample_factors, [2, 4])
    self.assertEqual(decorator._method, 'max')

  def test_build_decorator_with_bad_args(self):
    spec = decorators.DecoratorSpec.from_json(
        json.dumps({
            'name': 'Downsample',
            'args': {
                'downsample_factors': [2, 4],
                'method': 'max',
                'BAD_ARG': 'very_bad',
            },
        })
    )
    with self.assertRaises(TypeError):
      decorators.build_decorator(spec)

    spec = decorators.DecoratorSpec.from_json(
        json.dumps({
            'name': 'Downsample',
            'args': {
                'downsample_factors': [2, 4],
                # missing method
            },
        })
    )
    with self.assertRaises(TypeError):
      decorators.build_decorator(spec)

  def test_build_decorator_with_package(self):
    spec = decorators.DecoratorSpec.from_json(
        json.dumps({
            'name': 'TestDecorator',
            'args': {
                'foo': 1,
                'bar': 'baz',
            },
            'package': 'connectomics.volume.decorators_test',
        })
    )
    decorator = decorators.build_decorator(spec)
    # Can't use assertIsInstance because the package is imported.
    self.assertEqual(decorator.__class__.__name__, TestDecorator.__name__)
    self.assertEqual(decorator.foo, 1)
    self.assertEqual(decorator.bar, 'baz')


if __name__ == '__main__':
  absltest.main()
