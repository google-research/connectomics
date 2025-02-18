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
"""TensorStore decorators.

Example usage:
  vd = SomeDecorator(...)
  decorated_ts = vd.decorate(input_ts)
"""

import copy
import dataclasses
import enum
import json as json_lib
import pprint
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Union

from absl import logging
from connectomics.common import counters
from connectomics.common import file
from connectomics.common import import_util
from connectomics.common import metadata_utils
import dataclasses_json
import gin
import jax
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.stats
import skimage.feature
import tensorstore as ts
from typing_extensions import Protocol

JsonSpec = Mapping[str, Any]
MutableJsonSpec = MutableMapping[str, Any]
ObjectBounds = tuple[Union[int, str, slice], ...]


COUNTER_STORE = counters.ThreadsafeCounterStore()


@gin.constants_from_enum
class SpecAction(enum.Enum):
  """Behaviors for building output specs for special cases.

  The default action is to merge user provided overrides into the base derived
  from the decorated input, without clobbering nested values.
  """
  CLEAR = 0
  CLOBBER_WITH_INPUT_SPEC = 1


def _merge_specs(base: MutableJsonSpec, overrides: JsonSpec,
                 input_spec: JsonSpec) -> MutableJsonSpec:
  """Recursively update nested dict Spec.

  Args:
    base: Settings from base are kept by default, unless overridden.  Mutated in
      place with any overrides.
    overrides: Limited direct overrides; these are merged in recursively, so
      only direct name collisions get clobbered while other nested settings are
      retained.
    input_spec: This can be used as alternative override, e.g. for codec
      compression settings.  Selected via SpecAction.

  Returns:
    base, which is also mutated in place.
  """
  for k, v in overrides.items():
    if isinstance(v, Mapping):
      base[k] = _merge_specs(base.get(k, {}), v, input_spec.get(k, {}))
    elif v == SpecAction.CLEAR:
      _ = base.pop(k, None)
    elif v == SpecAction.CLOBBER_WITH_INPUT_SPEC:
      if k in input_spec:
        base[k] = input_spec[k]
    else:
      base[k] = v
  return base


def adjust_schema_for_chunksize(schema: ts.Schema,
                                other_chunksize: Sequence[int]) -> ts.Schema:
  chunksize = np.lcm(
      [s if s else 1 for s in schema.chunk_layout.read_chunk.shape],
      [s if s else 1 for s in other_chunksize])
  chunksize = np.minimum(chunksize, schema.shape)
  json = schema.to_json()
  json['chunk_layout']['read_chunk']['shape'] = chunksize
  json['chunk_layout']['write_chunk']['shape'] = chunksize
  return ts.Schema(json)


def adjust_schema_for_virtual_chunked(schema: ts.Schema) -> ts.Schema:
  json = schema.to_json()
  for k in ['fill_value']:
    if k in json:
      del json[k]
  return ts.Schema(json)


def _make_independent(
    context_spec: Optional[MutableJsonSpec] = None,
    default_data_copy_concurrency_limit: int = 8,
) -> JsonSpec:
  """Make a new context spec for a fully independent context.

  The spec is made from scratch or can be based on an existing one. All specs
  returned contain a limit for `data_copy_concurrency`.

  Args:
    context_spec: Context spec that is used as basis for the new spec. If it
      does not contain a limit for `data_copy_concurrency`, the default is used.
    default_data_copy_concurrency_limit: Default `data_copy_concurrency` limit.

  Returns:
    Dict object of JSON for a new, fully independent context.
  """
  if context_spec is None:
    context_spec = {}
  context_spec = copy.deepcopy(context_spec)
  _ = context_spec.setdefault('data_copy_concurrency', {}).setdefault(
      'limit', default_data_copy_concurrency_limit)
  return context_spec


class Decorator:
  """ABC for Decorators."""

  # Set this if the Decorator requires extra context beyond the requested pull
  # domain, as this will lead to read amplification.
  requires_context: bool = False

  def __init__(self, context_spec: Optional[MutableJsonSpec] = None):
    # Use this context to avoid danger of deadlock if a virtual_chunked reads
    # from an underlying TensorStore and shares its context.
    self._context = ts.Context(_make_independent(context_spec=context_spec))

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    raise NotImplementedError

  def debug_string(self) -> str:
    """Returns string for debugging purposes."""
    return ''

  def print_debug_string(self):
    """Prints debug string if not empty."""
    debug_str = self.debug_string()
    if debug_str:
      print(f'{self.__class__.__name__} debug_string:')
      print(debug_str)


def _cast_img(
    img: np.ndarray,
    dtype: str,
    rescale: bool = True,
    force_copy: bool = False,
) -> np.ndarray:
  """Casts image to new datatype with optional rescaling.

  Args:
    img: Input image to convert.
    dtype: New dtype.
    rescale: Whether to apply rescaling to values in addition to dtype
      conversion. Rescaling is handled by skimage utility functions. The
      following dtypes are supported and rescaling is performed as follows:
        `bool`: Negative values are `False`. Upper half of input image dtype
          positive range is `True`, lower half `False`.
        `float32`, `float64`: When converting signed dtypes, rescales to
          [-1.0, 1.0], for unsigned to [0.0, 1.0]. Float inputs are unchanged.
        `int16`: Rescales between [-32768, 32767]. When converting unsigned
          dtypes, all values are positive.
        `uint8`: Negative values are clipped. Positive values are rescaled
          between 0 and 255.
        `uint16`: Negative values are clipped. Positive values are rescaled
          between 0 and 65535.
    force_copy: Whether or not to force copying data.

  Returns:
    Recast image.
  """
  dtype = str(dtype)

  if rescale:
    conversion_utils = {
        'bool': skimage.util.img_as_bool,
        'float32': skimage.util.img_as_float32,
        'float64': skimage.util.img_as_float64,
        'int16': skimage.util.img_as_int,
        'uint8': skimage.util.img_as_ubyte,
        'uint16': skimage.util.img_as_uint,
    }
    if dtype not in conversion_utils:
      raise TypeError(
          f'Only conversions to dtypes {conversion_utils.keys()} are ' +
          f'supported but {dtype} was requested.')
    return conversion_utils[dtype](img, force_copy=force_copy)
  else:
    return img.astype(dtype, copy=force_copy)


@gin.register
class Cast(Decorator):
  """Casts inputs to new datatype with optional rescaling."""

  def __init__(self,
               dtype: str,
               rescale: bool = True,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None):
    """Cast.

    Args:
      dtype: New datatype.
      rescale: Whether or not rescaling should be applied.
      min_chunksize: Defines the minimum chunk size.
      context_spec: Spec for virtual chunked context overriding its defaults.
    """
    super().__init__(context_spec)
    self._dtype = dtype
    self._rescale = rescale
    self._min_chunksize = min_chunksize

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps input TensorStore for casting."""

    if not self._rescale:
      return ts.cast(input_ts, self._dtype)

    def read_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      array[...] = _cast_img(np.array(input_ts[domain]), self._dtype,
                             rescale=True)

    json = input_ts.schema.to_json()
    json['dtype'] = self._dtype
    schema = ts.Schema(json)
    if self._min_chunksize is not None:
      schema = adjust_schema_for_chunksize(schema, self._min_chunksize)
    schema = adjust_schema_for_virtual_chunked(schema)
    return ts.virtual_chunked(read_fn, schema=schema, context=self._context)


class Processor(Protocol):

  def __call__(self, data: np.ndarray, **processor_args) -> np.ndarray:
    ...


def _process_planes_nd(
    processor: Processor,
    data: np.ndarray,
    batch_dims: Optional[Sequence[int]] = None,
    output: Optional[Union[np.dtype, np.ndarray]] = None,
    **processor_args) -> np.ndarray:
  """Apply function processing planes on n-dimensional inputs.

  Batched application of arbitrary function processing planes, e.g., 2D images.

  Args:
    processor: Function processing 2-dimensional data in form of arrays,
      returning 2-dimensional outputs.
    data: Input data to process, at least 2-dimensional.
    batch_dims: Batch dimensions. Required unless data is 2-dimensional.
    output: Output array or dtype of output array. By default, an output array
      with the same shape and dtype as data is used.
    **processor_args: Passed to `processor`.

  Returns:
    Processed input data.
  """
  num_dim = data.ndim
  if num_dim < 2:
    raise ValueError(f'Data must at least be 2-dimensional, but is {num_dim}d.')
  num_batch_dim = 0 if batch_dims is None else len(batch_dims)
  if (2 + num_batch_dim) != num_dim:
    raise ValueError(
        f'Expected {num_dim - 2} batch dimension(s), ' +
        f'but got {num_batch_dim} batch dimension(s) instead.')
  if num_dim == 2:
    return processor(data, **processor_args)

  win = list(data.shape)
  for b in batch_dims:
    win[b] = 1
  data_view = np.lib.stride_tricks.sliding_window_view(data, window_shape=win)

  if isinstance(output, np.dtype):
    output = np.empty(shape=data.shape, dtype=output)
  elif output is None:
    output = np.empty_like(data)

  it = np.nditer(output, op_axes=[batch_dims], flags=['multi_index'])
  with it:
    idx = [slice(None) for _ in range(num_dim)]
    for _ in it:
      for i, b in enumerate(batch_dims):
        idx[b] = list(it.multi_index)[i]
      output[tuple(idx)] = processor(
          data_view[tuple(idx + [...])].squeeze(), **processor_args)[...]
  return output


@gin.register
class Filter(Decorator):
  """Runs filter function over image."""

  def __init__(self,
               filter_fun: Any,
               min_chunksize: Optional[Sequence[int]] = None,
               overlap_width: Optional[Sequence[tuple[int, int]]] = None,
               pad_border: Optional[MutableMapping[str, Any]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    """Filter.

    Filters preserve the input's size.

    If edge effects are of concern, this can be addressed in two ways: By
    setting `min_chunksize` as large as possible for relevant dimensions, or,
    by specifying `overlap_width`. In the latter case, reads pull additional
    context at chunks' margins. This context gets discarded after applying the
    filter function.

    Args:
      filter_fun: Filter function.
      min_chunksize: Defines the minimum chunk size.
      overlap_width: Width of overlap per dimension. For each dimension, a pair
        of integers indicating amount to expand lower/upper domain bounds by.
      pad_border: If `overlap_width` is specified, the volume can optionally be
        padded at borders where no overlap context is available. If `pad_border`
        is not `None`, it is used as a keyword dictionary passed to `np.pad`.
      context_spec: Spec for virtual chunked context overriding its defaults.
      **filter_args: Passed to `filter_fun`.
    """
    super().__init__(context_spec)
    self._filter_fun = filter_fun
    self._filter_args = filter_args
    self._min_chunksize = min_chunksize
    self._overlap_width = overlap_width
    self._pad_border = pad_border

  @property
  def requires_context(self) -> bool:
    return False if self._overlap_width is None else True

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps input TensorStore with a filtered virtual_chunked."""

    def filt_read(domain: ts.IndexDomain, array: np.ndarray,
                  unused_read_params: ts.VirtualChunkedReadParameters):
      if self._overlap_width is None:
        array[...] = self._filter_fun(
            np.array(input_ts[domain]), **self._filter_args)
      else:
        read_domain, pad_width = _expand_domain(
            domain, self._overlap_width, input_ts.shape)
        if self._pad_border is None:
          data = np.array(input_ts[read_domain])
        else:
          data = np.pad(np.array(input_ts[read_domain]), pad_width=pad_width,
                        **self._pad_border)
        result = self._filter_fun(data, **self._filter_args)
        if self._pad_border is None:
          idxs = [slice(w[0] - p[0], s - w[1] + p[1]) for w, p, s in zip(
              self._overlap_width, pad_width, data.shape)]
        else:
          idxs = [slice(w[0], s - w[1]) for w, s in zip(
              self._overlap_width, data.shape)]
        array[...] = result[tuple(idxs)]

    schema = input_ts.schema
    if self._min_chunksize is not None:
      schema = adjust_schema_for_chunksize(schema, self._min_chunksize)
    schema = adjust_schema_for_virtual_chunked(schema)
    return ts.virtual_chunked(
        filt_read, schema=schema, context=self._context)


def _active_contours_mask(
    data: np.ndarray,
    background: Optional[Sequence[int]] = None,
    num_iter_chan_vese: int = 2,
    num_iter_erosion: int = 10,
    num_iter_dilation: int = 10,
    fill_holes: bool = True) -> np.ndarray:
  """Runs active contours to create image mask."""
  mask = skimage.segmentation.morphological_chan_vese(
      _cast_img(data, 'float64').squeeze(),
      num_iter=num_iter_chan_vese).astype(bool)
  if background is not None:
    mask = ~mask if mask[background].all() else mask
  if num_iter_erosion > 0:
    mask = scipy.ndimage.binary_erosion(mask, iterations=num_iter_erosion)
  if num_iter_dilation > 0:
    mask = scipy.ndimage.binary_dilation(mask, iterations=num_iter_dilation)
  if fill_holes:
    mask = scipy.ndimage.binary_fill_holes(mask)
  return mask.astype(data.dtype).reshape(data.shape)


@gin.register
class ActiveContoursMaskFilter(Filter):
  """Runs active contours to create image mask."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_active_contours_mask,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _exposure_filter(
    data: np.ndarray, mode: str, cast_float64: bool, **filter_args
) -> np.ndarray:
  """Exposure filters."""
  supported_modes = {
      'adjust_gamma': skimage.exposure.adjust_gamma,
      'adjust_log': skimage.exposure.adjust_log,
      'adjust_sigmoid': skimage.exposure.adjust_sigmoid,
      'equalize_adapthist': skimage.exposure.equalize_adapthist,
      'equalize_hist': skimage.exposure.equalize_hist,
      'rescale_intensity': skimage.exposure.rescale_intensity,
  }
  if mode not in supported_modes:
    raise ValueError(
        f'Mode {mode} not supported. ' +
        f'Supported modes are: {supported_modes.keys()}.')

  if not cast_float64:
    return supported_modes[mode](data, **filter_args)
  else:
    result = supported_modes[mode](_cast_img(data, 'float64'), **filter_args)
    return _cast_img(result, data.dtype)


@gin.register
class ExposureFilter(Filter):
  """Runs filter changing exposure over image."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               overlap_width: Optional[Sequence[tuple[int, int]]] = None,
               pad_border: Optional[MutableMapping[str, Any]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_exposure_filter,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        overlap_width=overlap_width,
        pad_border=pad_border,
        **filter_args)


@gin.register
class CLAHEFilter(ExposureFilter):
  """Runs Contrast Limited Adaptive Histogram Equalization over image."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               overlap_width: Optional[Sequence[tuple[int, int]]] = None,
               pad_border: Optional[MutableMapping[str, Any]] = None,
               **filter_args):
    super().__init__(
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        overlap_width=overlap_width,
        pad_border=pad_border,
        mode='equalize_adapthist',
        cast_float64=True,
        **filter_args)


@gin.register
class GaussianFilter(Filter):
  """Runs gaussian filter over image."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=scipy.ndimage.gaussian_filter,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _label_filter(data: np.ndarray, **label_kwargs) -> np.ndarray:
  label, _ = scipy.ndimage.label(data, **label_kwargs)
  return label


@gin.register
class LabelFilter(Filter):
  """Labels features in an image."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_label_filter,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


@gin.register
class Log1pFilter(Filter):
  """Applies log(1+x)."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=np.log1p,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


@gin.register
class MedianFilter(Filter):
  """Runs median filter over image."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               overlap_width: Optional[Sequence[tuple[int, int]]] = None,
               pad_border: Optional[MutableMapping[str, Any]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=scipy.ndimage.median_filter,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        overlap_width=overlap_width,
        pad_border=pad_border,
        **filter_args)


def _min_sum_filter(data: np.ndarray, min_sum: Union[float, int]) -> np.ndarray:
  return data if data.sum() >= min_sum else np.zeros_like(data)


@gin.register
class MinSumFilter(Filter):
  """Runs min sum filter over image; zeros out image if its below `min_sum`."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_min_sum_filter,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _nan_replace(
    data: np.ndarray, nan_fill: Union[int, float] = 0) -> np.ndarray:
  """Replaces NaN values in data."""
  data[np.isnan(data)] = nan_fill
  return data


@gin.register
class NaNReplaceFilter(Filter):
  """Runs NaN replacement over image."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_nan_replace,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _peak_filter_2d(data: np.ndarray, **peak_local_max_kwargs) -> np.ndarray:
  """Peak filter on 2-dimensional inputs via skimage's `peak_local_max`.

  Args:
    data: Input data, must be 2-dimensional.
    **peak_local_max_kwargs: Passed to `skimage.feature.peak_local_max`.

  Returns:
    Masked array of image peaks.
  """
  assert data.ndim == 2, (
      f'Only 2-dimensional data supported, but is {data.ndim}-dimensional.')
  mask = np.zeros_like(data)
  idx = skimage.feature.peak_local_max(data, **peak_local_max_kwargs)
  mask[tuple(idx.T)] = True
  return mask


def _peak_filter_nd(
    data: np.ndarray, batch_dims: Optional[Sequence[int]] = None,
    **peak_local_max_kwargs) -> np.ndarray:
  """Peak filter on n-dimensional inputs."""
  return _process_planes_nd(
      _peak_filter_2d, data, batch_dims, **peak_local_max_kwargs)


@gin.register
class PeakFilter(Filter):
  """Runs peak filter over image."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_peak_filter_nd,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


@gin.register
class PercentileFilter(Filter):
  """Runs percentile filter over image."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=scipy.ndimage.percentile_filter,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _scale(data: np.ndarray, factor: float) -> np.ndarray:
  return data * factor


@gin.register
class ScaleFilter(Filter):
  """Scales data."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_scale,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _threshold(
    data: np.ndarray, threshold: Union[int, float], indices: bool = False
) -> np.ndarray:
  """Thresholds image."""
  idxs = np.where(data > threshold)
  if indices:
    return idxs
  else:
    mask = np.zeros_like(data)
    mask[idxs] = True
    return mask


@gin.register
class ThresholdFilter(Filter):
  """Runs thresholding over image."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_threshold,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


@gin.register
class ClipFilter(Filter):
  """Clips image to min and max values."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=np.clip,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _standardize(
    data: np.ndarray, mean: float, std: float
) -> np.ndarray:
  """Standardizes data given mean and standard deviation estimate."""
  return (data - mean) / std


@gin.register
class StandardizeFilter(Filter):
  """Standardizes volume based on provided mean and standard deviation."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_standardize,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


@gin.register
class ZScoreFilter(Filter):
  """Applies z-scoring based on calculated mean and standard deviation."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=scipy.stats.zscore,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _lowpass_filter(
    data: np.ndarray,
    cutoff_freq: float,
    axis: int,
    sampling_freq: float = 1,
    order: int = 5,
):
  """Applies digital butterworth lowpass filter to data."""
  sos = scipy.signal.butter(
      order,
      cutoff_freq,
      analog=False,
      btype='low',
      output='sos',
      fs=sampling_freq)
  return scipy.signal.sosfiltfilt(sos, data, axis=axis, padlen=0)


@gin.register
class LowpassFilter(Filter):
  """Applies butterworth lowpass filter."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_lowpass_filter,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _unsharp_mask(
    data: np.ndarray, num_iterations: int = 1, cast_float64: bool = True,
    **unsharp_mask_kwargs) -> np.ndarray:
  """Unsharp masking filter."""
  result = data if not cast_float64 else _cast_img(data, 'float64')
  for _ in range(num_iterations):
    result = skimage.filters.unsharp_mask(result, **unsharp_mask_kwargs)
  return result if not cast_float64 else _cast_img(result, data.dtype)


@gin.register
class UnsharpMaskFilter(Filter):
  """Runs unsharp masking over image."""

  def __init__(self,
               min_chunksize: Sequence[int] | None = None,
               context_spec: MutableJsonSpec | None = None,
               **filter_args):
    super().__init__(
        filter_fun=_unsharp_mask,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


@gin.register
class Interpolation(Decorator):
  """Interpolates input TensorStore."""

  def __init__(self,
               size: Sequence[int],
               backend: str = 'scipy_map_coordinates',
               context_spec: Optional[MutableJsonSpec] = None,
               **interpolation_args):
    """Interpolation.

    Args:
      size: New size of TensorStore.
      backend: Backend to use for interpolation. One of 'scipy_map_coordinates',
        'jax_map_coordinates', or 'jax_resize'. Defaults to the first.
      context_spec: Spec for virtual chunked context overriding its defaults.
      **interpolation_args: Passed to `scipy.ndimage.map_coordinates`,
        `jax.scipy.ndimage.map_coordinates` , or`jax.image.resize` depending
        on `backend`, respectively.
    """
    super().__init__(context_spec)
    self._size = size
    backends = (
        'scipy_map_coordinates', 'jax_map_coordinates', 'jax_resize', 'pad')
    if backend not in backends:
      raise ValueError(f'Unsupported backend: {backend} not in {backends}.')
    self._backend = backend
    self._interpolation_args = interpolation_args

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps input TensorStore with virtual_chunked for interpolation."""

    if len(self._size) != input_ts.ndim:
      raise ValueError(
          f'Length of `size` ({len(self._size)}) does not match ' +
          f'dimensionality of input TensorStore ({input_ts.ndim}).')
    if (any(new < old for new, old in zip(self._size, input_ts.shape))
        and self._backend == 'pad'):
      raise ValueError('Can only pad to increase size.')
    inclusive_min = input_ts.schema.domain.inclusive_min
    if inclusive_min != tuple([0 for _ in range(input_ts.ndim)]):
      raise ValueError(
          'Only input TensorStores with `inclusive_min` all zeros are ' +
          f'currently supported, but `inclusive_min` is: {inclusive_min}.')

    resize_dim = [d for d, s in enumerate(self._size) if s != input_ts.shape[d]]
    map_coordinates = (scipy.ndimage.map_coordinates if 'scipy' in self._backend
                       else jax.scipy.ndimage.map_coordinates)

    def read_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      read_domain = list(domain)
      for d in resize_dim:
        read_domain[d] = ts.Dim(inclusive_min=0,
                                exclusive_max=input_ts.shape[d],
                                label=input_ts.domain.labels[d])
      read_domain = ts.IndexDomain(read_domain)
      data = np.array(input_ts[read_domain])

      slices = []
      for d in range(input_ts.ndim):
        if d in resize_dim:
          # A complex step length is used since np.mgrid interprets it as the
          # number of points to create between start and inclusive stop values.
          slices.append(slice(0, data.shape[d]-1, complex(self._size[d])))
        else:
          slices.append(slice(0, data.shape[d]))

      if self._backend == 'jax_resize':
        sub_size = [s if d in resize_dim else data.shape[d]
                    for d, s in enumerate(self._size)]
        array[...] = jax.image.resize(data, sub_size,
                                      **self._interpolation_args)
      elif self._backend == 'pad':
        pad_width = []
        for d in range(input_ts.ndim):
          if d not in resize_dim:
            pad_width.append((0, 0))
          else:
            difference = self._size[d] - data.shape[d]
            left = difference // 2
            pad_width.append((left, difference - left))
        array[...] = np.pad(data, pad_width=pad_width,
                            **self._interpolation_args)
      else:
        array[...] = map_coordinates(data, np.mgrid[slices],
                                     **self._interpolation_args)

    json = input_ts.schema.to_json()
    json['domain']['exclusive_max'] = self._size

    min_chunksize = [s if s != input_ts.shape[d] else 1
                     for d, s in enumerate(self._size)]
    schema = adjust_schema_for_chunksize(ts.Schema(json), min_chunksize)
    schema = adjust_schema_for_virtual_chunked(schema)
    return ts.virtual_chunked(read_fn, schema=schema, context=self._context)


class ProjectionFn(Protocol):
  """Reduces N-dimensional array along `axis` dimension."""

  def __call__(self, a: np.ndarray, axis: int, **projection_args) -> np.ndarray:
    ...


@gin.register
class Projection(Decorator):
  """Reduces input TensorStore along given dimension via some function."""

  def __init__(self,
               projection_fn: ProjectionFn,
               projection_dim: int,
               remove_infs: bool = True,
               context_spec: Optional[MutableJsonSpec] = None,
               **projection_args):
    """Projection.

    Args:
      projection_fn: Function processing and reducing data.
      projection_dim: Dimension along which the input is reduced.
      remove_infs: If `True`, sets Inf values to zero before projecting.
      context_spec: Spec for virtual chunked context overriding its defaults.
      **projection_args: Passed to `projection`.
    """
    super().__init__(context_spec=context_spec)
    self._projection_fn = projection_fn
    self._projection_dim = projection_dim
    self._remove_infs = remove_infs
    self._projection_args = projection_args

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps the input TensorStore with a projected virtual_chunked."""

    def projected_read(domain: ts.IndexDomain, array: np.ndarray,
                       unused_read_params: ts.VirtualChunkedReadParameters):
      read_domain = list(domain)
      read_domain.insert(self._projection_dim,
                         input_ts.domain[self._projection_dim])
      read_domain = ts.IndexDomain(read_domain)

      # Remove Infs, then project.
      data = np.array(input_ts[read_domain])
      if self._remove_infs:
        data[np.isinf(data)] = 0
      array[...] = self._projection_fn(
          data, axis=self._projection_dim, **self._projection_args)

    schema = adjust_schema_for_virtual_chunked(input_ts.schema)
    projected_schema = schema[ts.d[self._projection_dim][0]]
    return ts.virtual_chunked(
        projected_read, schema=projected_schema, context=self._context)


@gin.register
class MaxProjection(Projection):
  """Reduces input TensorStore along given dimension via nanmax()."""

  def __init__(self,
               projection_dim: int,
               context_spec: Optional[MutableJsonSpec] = None):
    super().__init__(
        projection_fn=np.nanmax,
        projection_dim=projection_dim,
        context_spec=context_spec)


@gin.register
class MeanProjection(Projection):
  """Reduces input TensorStore along given dimension via nanmean()."""

  def __init__(self,
               projection_dim: int,
               context_spec: Optional[MutableJsonSpec] = None):
    super().__init__(
        projection_fn=np.nanmean,
        projection_dim=projection_dim,
        context_spec=context_spec)


@gin.register
class StdProjection(Projection):
  """Reduces input TensorStore along given dimension via nanstd()."""

  def __init__(self,
               projection_dim: int,
               context_spec: Optional[MutableJsonSpec] = None):
    super().__init__(
        projection_fn=np.nanstd,
        projection_dim=projection_dim,
        context_spec=context_spec)


@gin.register
class SumProjection(Projection):
  """Reduces input TensorStore along given dimension via nansum()."""

  def __init__(self,
               projection_dim: int,
               context_spec: Optional[MutableJsonSpec] = None):
    super().__init__(
        projection_fn=np.nansum,
        projection_dim=projection_dim,
        context_spec=context_spec)


@gin.register
class MultiplyPointwise(Decorator):
  """Multiply TensorStores pointwise."""

  def __init__(self,
               multiply_spec: JsonSpec,
               context_spec: Optional[MutableJsonSpec] = None):
    """Multiply pointwise.

    Args:
      multiply_spec: Spec for TensorStore to multiply input with.
      context_spec: Spec for virtual chunked context overriding its defaults.
    """
    super().__init__(context_spec)
    self._multiply_spec = multiply_spec

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps the input TensorStore with virtual chunked for multiplication."""

    multiply_ts = ts.open(self._multiply_spec).result()

    def read_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      array[...] = np.array(input_ts[domain]) * np.array(multiply_ts[domain])

    schema = adjust_schema_for_virtual_chunked(input_ts.schema)
    return ts.virtual_chunked(
        read_fn, schema=schema, context=self._context)


def _expand_domain(
    domain: ts.IndexDomain,
    width: Sequence[tuple[int, int]],
    limits: Optional[Sequence[int]] = None
) -> tuple[ts.IndexDomain, Sequence[tuple[int, int]]]:
  """Expands domain to include additional context.

  Args:
    domain: Input domain.
    width: Width of additional context to include. For each dimension, a pair
      of integers indicating amount to expand lower and upper domain bounds by.
    limits: Optional upper limits beyond which the expanded domain is truncated,
      e.g., the shape of a TensorStore.

  Returns:
    Expanded domain and trunction width. Trunction width indicates the amount
    of trunctation applied to lower and upper domain bounds per dimension;
    can for example be used as `pad_width` keyword to `np.pad`.
  """
  assert len(domain) == len(width), (
      'Length of domain and context must be equal, '
      f'but are {len(domain)} and {len(width)}.')
  if limits is not None:
    assert len(domain) == len(limits), (
        'Length of domain and limits must be equal, ' +
        f'but are {len(domain)} and {len(limits)}.')
    bounds = [(0, l) for l in limits]
  else:
    bounds = [(0, +np.inf) for _ in range(len(domain))]

  expanded_domain, truncation_width = [], []
  for d, w, b in zip(list(domain), width, bounds):
    assert len(w) == 2, (
        'Width must be specified as a pair for each dimension, indicating ' +
        'the amount to expand lower and upper domain bounds by.')
    assert w[0] >= 0 and w[1] >= 0, 'Width must be non-negative.'

    incl_min, excl_max = d.inclusive_min - w[0], d.exclusive_max + w[1]
    expanded_domain.append(ts.Dim(max(b[0], incl_min), min(b[1], excl_max),
                                  label=d.label))

    truncation_before = 0 if incl_min >= b[0] else b[0] - incl_min
    truncation_after = 0 if excl_max <= b[1] else excl_max - b[1]
    truncation_width.append((truncation_before, truncation_after))

  return ts.IndexDomain(expanded_domain), truncation_width


def _interval_indexing_str_to_slice(expr: str) -> slice:
  """Parses NumPy-style interval indexing expression to slice representation."""
  if not isinstance(expr, str):
    raise ValueError(f'`expr` must be of type str but is {type(expr)}.')
  parts = []
  if expr:
    for p in expr.split(':'):
      if p:
        parts.append(int(p) if 'j' not in p else complex(p))
      else:
        parts.append(None)
    if len(parts) == 1:
      parts.insert(0, None)
  return slice(*parts)


def _object_bounds_to_slices(
    object_bounds: Sequence[ObjectBounds]
) -> list[tuple[slice, ...]]:
  """Parses different representations of objects to slice representation.

  In slice represention, N-D objects are described by a tuple of N slices,
  describing their minimal parallelepipeds.

  Args:
    object_bounds: A sequence of object bounds to parse, valid incoming
      representations are tuples of integers, e.g., (1, 2) for a 2D object,
      strings that are NumPy-style interval indexing expressions, e.g.,
      ("1:3", "3:6"), and slices, in which case no conversion is needed.

  Returns:
    Objects in slice representation.
  """
  object_slices = []
  for obj in object_bounds:
    slices = []
    for d in obj:
      if isinstance(d, slice):
        # Instantiating new slice to avoid spurious pytype errors.
        slices.append(slice(d.start, d.stop, d.step))
      elif isinstance(d, int):
        slices.append(slice(d, d + 1))
      elif isinstance(d, str):
        slices.append(_interval_indexing_str_to_slice(d))
      else:
        raise ValueError('Invalid type encountered in object bounds: ' +
                         f'Valid types are slice, int, str but got {type(d)}.')
    object_slices.append(tuple(slices))
  return object_slices


def _object_slices_to_domains(
    object_slices: Sequence[Union[tuple[slice, ...], None]],
    labels: list[str],
    offset: int = 0,
) -> dict[int, ts.IndexDomain]:
  """Creates IndexDomains from object slices.

  Args:
    object_slices: A sequence of tuples with each tuple containing N slices.
        Slices describe the minimal parallelepiped containing an N-D object.
    labels: Labels for N dimensions.
    offset: Optional offset added to object index.

  Returns:
    Dictionary mapping object indices to IndexDomains.
  """
  objects = {}
  for i, obj in enumerate(object_slices):
    if obj:
      objects[i + offset] = ts.IndexDomain(
          [ts.Dim(inclusive_min=obj[d].start, exclusive_max=obj[d].stop,
                  label=label)
           for d, label in enumerate(labels)])
  return objects


def _center_crop(array: np.ndarray, size: Sequence[int]) -> np.ndarray:
  """Takes center crop of given size from array."""
  if array.shape == size:
    return array
  if len(size) != array.ndim:
    raise ValueError(
        '`size` needs to have same length as array dimensionality ' +
        f'but has length {len(size)} rather than {array.ndim}.')

  slices = []
  for d in range(array.ndim):
    if size[d] <= 0:
      raise ValueError(
          '`size` entries must be greater than zero ' +
          f'but is {size[d]} <= 0 for dimension {d}.')
    if size[d] > array.shape[d]:
      raise ValueError(
          '`size` entries must be less than or equal array shape ' +
          f'but is {size[d]} > {array.shape[d]} for dimension {d}.')

    offset = int(round((array.shape[d] - size[d]) / 2.))
    slices.append(slice(offset, offset + size[d]))

  return array[tuple(slices)]


@gin.register
class ObjectsContext(Decorator):
  """Extracts context around labelled objects."""

  def __init__(self,
               width: Sequence[tuple[int, int]],
               objects_spec: Union[JsonSpec, Sequence[ObjectBounds]],
               spec_overrides: JsonSpec,
               context_spec: Optional[MutableJsonSpec] = None,
               **padding_args):
    """Objects with context.

    Extracts context around labelled objects, e.g., useful for segmentation of
    functional imaging data (with seeds as objects). The resulting TS has a
    last extra dimension relative to the input one, indexing into objects.

    If an object does not exist, the read returns all zeros.

    Args:
      width: Width of context to include. For each dimension, a pair of integers
        indicating amount to expand lower and upper domain bounds by. Note that
        if this is smaller than the minimal parallelepiped containing an object,
        the object is center cropped.
      objects_spec: TensorStore containing labelled objects. Alternatively,
        objects can be passed directly as a sequence of object bounds. Object
        bounds are internally converted to slices. For valid representations
        see `_object_bounds_to_slices`.
      spec_overrides: Spec overrides applied to input TS schema to generate the
        virtual chunked schema. Since a last extra dimension is added, overrides
        must contain specification of the new chunk layout, domain, and rank.
      context_spec: Spec for virtual chunked context overriding its defaults.
        Note that the term context is overloaded; context_spec refers to the
        shared resource context, just as it does for all other decorators.
      **padding_args: Passed to `np.pad`, which is used to pad data in case the
        specified width extends beyond the input TS bounds for an object.
    """
    super().__init__(context_spec)
    self._width = width
    self._objects_spec = objects_spec
    self._spec_overrides = spec_overrides
    self._padding_args = padding_args
    self._cached_objects = None

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps the input TensorStore with a virtual_chunked for object context."""

    def read_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      if not self._cached_objects:
        if isinstance(self._objects_spec, Sequence):
          self._cached_objects = _object_slices_to_domains(
              _object_bounds_to_slices(self._objects_spec),
              input_ts.domain.labels, offset=0)
        else:
          object_ts = ts.open(self._objects_spec).result()
          self._cached_objects = _object_slices_to_domains(
              scipy.ndimage.find_objects(np.array(object_ts, dtype=np.uint64)),
              input_ts.domain.labels, offset=1)

      for i, o in enumerate(list(domain[-1])):
        if o in self._cached_objects:
          read_domain, pad_width = _expand_domain(
              self._cached_objects[o], self._width, limits=input_ts.shape)
          array[..., i] = _center_crop(
              np.pad(np.array(input_ts[read_domain]),
                     pad_width=pad_width, **self._padding_args),
              size=array[..., i].shape)
        else:
          array[..., i] = np.zeros_like(array[..., i])

    if 'schema' not in self._spec_overrides:
      raise ValueError('`spec_overrides` is missing key `schema`.')
    for key in ('chunk_layout', 'domain', 'rank'):
      if key not in self._spec_overrides['schema']:
        raise ValueError(f'`spec_overrides` schema is missing key `{key}`.')
    for key in ('grid_origin', 'inner_order', 'read_chunk', 'write_chunk'):
      if key not in self._spec_overrides['schema']['chunk_layout']:
        raise ValueError(f'`spec_overrides` chunk_layout misses key `{key}`.')
    for key in ('inclusive_min', 'exclusive_max'):
      if key not in self._spec_overrides['schema']['domain']:
        raise ValueError(f'`spec_overrides` domain misses key `{key}`.')
    if self._spec_overrides['schema']['rank'] != (input_ts.rank + 1):
      raise ValueError(
          f'`spec_overrides` rank must be {self._spec_overrides["rank"] + 1} ' +
          f'but is {self._spec_overrides["rank"]}.')

    input_spec = dict(schema=input_ts.schema.to_json())
    schema = ts.Schema(_merge_specs(input_spec['schema'],
                                    self._spec_overrides['schema'], {}))
    schema = adjust_schema_for_virtual_chunked(schema)
    return ts.virtual_chunked(read_fn, schema=schema, context=self._context)


@gin.register
class Downsample(Decorator):
  """Downsamples input TensorStore."""

  def __init__(self, downsample_factors: Sequence[int], method: str):
    """Downsample TensorStore.

    Args:
      downsample_factors: Factors by which to downsample each dimension of base.
      method: Downsampling method. One of `stride`, `median`, `mode`, `mean`,
        `min`, or `max`. See TensorStore downsample Driver for details.
    """
    super().__init__(context_spec=None)
    self._downsample_factors = downsample_factors
    self._method = method

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    return ts.downsample(input_ts, self._downsample_factors, self._method)


def _build_output_spec(
    input_schema_spec: JsonSpec,
    decorated_ts: ts.TensorStore,
    overrides: JsonSpec,
    dryrun: bool,
) -> JsonSpec:
  """Builds output spec, prints to console if dryrun."""
  output_spec = dict(schema=decorated_ts.schema.to_json())
  _merge_specs(output_spec, overrides, input_schema_spec)
  if dryrun:
    # Override driver and create to avoid side effects in dryrun.
    output_spec['kvstore'] = dict(driver='memory')
    output_spec['create'] = True
  return output_spec


class Writer(Decorator):
  """ABC for Decorators that write output."""

  def initialize(self, input_schema_spec: JsonSpec,
                 decorated_ts: ts.TensorStore, dryrun: bool):
    raise NotImplementedError

  def post_jsons(self) -> Iterable[tuple[JsonSpec, ts.KvStore.Spec]]:
    raise NotImplementedError


@gin.register
class Write(Writer):
  """Passes through data, writing it out as side effect."""

  def __init__(self,
               output_spec_overrides: JsonSpec,
               context_spec: Optional[MutableJsonSpec] = None,
               keep_existing_chunks: bool = False):
    super().__init__(context_spec)
    self._output_spec_overrides = output_spec_overrides
    self._output_ts = None
    self._keep_existing_chunks = keep_existing_chunks
    self._initialized = False

  def initialize(self, input_schema_spec: JsonSpec,
                 decorated_ts: ts.TensorStore, dryrun: bool):
    """Get output_ts from decorated_ts, input_schema_spec, and overrides."""
    if self._initialized:
      raise AssertionError('Write.initialize called twice.')
    output_spec = _build_output_spec(input_schema_spec, decorated_ts,
                                     self._output_spec_overrides, dryrun)
    self._output_ts = ts.open(output_spec).result()

    # This must be cleared, otherwise serializing the Write can exceed the
    # recursion limit, putatively due to the possible presence of SpecActions
    # that are gin wrapped.
    self._output_spec_overrides = None

    self._initialized = True

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps input TensorStore in virtual_chunked with writing side-effect."""
    if not self._initialized:
      raise AssertionError('Write.initialize must be called first.')

    def side_write_read(domain: ts.IndexDomain, array: np.ndarray,
                        unused_read_params: ts.VirtualChunkedReadParameters):
      if self._keep_existing_chunks:
        existing_data = self._output_ts[domain].read().result()
        if existing_data.sum() != 0.0:
          if all(self._output_ts.dimension_units):
            resolution = '-' + '-'.join(
                str(u.multiplier) for u in self._output_ts.dimension_units
            )
          else:
            resolution = ''
          COUNTER_STORE.get_counter(
              f'write-chunk{resolution}-skipped-existing').inc()
          logging.vlog(2, 'Skipping chunk %s; existing data (nonzero)', domain)
          array[...] = existing_data
          return

      logging.vlog(2, 'Processing chunk %s', domain)
      data = input_ts[domain]
      self._output_ts[domain] = data
      array[...] = data

    schema = adjust_schema_for_chunksize(
        input_ts.schema, self._output_ts.chunk_layout.read_chunk.shape)
    schema = adjust_schema_for_virtual_chunked(schema)
    return ts.virtual_chunked(
        side_write_read, schema=schema, context=self._context)

  def post_jsons(self) -> Iterable[tuple[JsonSpec, ts.KvStore.Spec]]:
    if not self._initialized:
      raise AssertionError('Write.initialize must be called first.')
    materialize_log = metadata_utils.get_run_log()
    materialize_log['gin_operative_config'] = gin.operative_config_str()
    log_kvstore = self._output_ts.kvstore.spec() / 'materialize_log.json'
    yield materialize_log, log_kvstore

  def debug_string(self) -> str:
    if not self._initialized:
      raise AssertionError('Write.initialize must be called first.')
    ds = pprint.pformat(self._output_ts.schema.to_json())
    return '_output_ts.schema:\n' + ds


@gin.register
class MultiscaleWrite(Writer):
  """Composes multiple Write / Downsamples."""

  def __init__(self,
               output_base_spec_overrides: MutableJsonSpec,
               sequential_downsample_factors: Sequence[Sequence[int]],
               downsample_method: str,
               context_spec: Optional[MutableJsonSpec] = None,
               keep_existing_chunks: bool = False):
    super().__init__(context_spec)
    self._output_base_spec_overrides = output_base_spec_overrides
    self._sequential_downsample_factors = sequential_downsample_factors
    self._downsample_method = downsample_method
    self._keep_existing_chunks = keep_existing_chunks

    self._output_base_spec = None
    self._chain = None
    self._downsampling_factors = None
    self._axes = None
    self._dimension_units = None

    self._initialized = False

  def initialize(self, input_schema_spec: JsonSpec,
                 decorated_ts: ts.TensorStore, dryrun: bool):
    if self._initialized:
      raise AssertionError('MultiscaleWrite.initialize called twice.')

    self._output_base_spec = ts.Spec(
        _build_output_spec(input_schema_spec, decorated_ts,
                           self._output_base_spec_overrides, dryrun))
    self._axes = self._output_base_spec.domain.labels
    if all(self._output_base_spec.dimension_units):
      self._dimension_units = self._output_base_spec.dimension_units
    elif all(ts.Spec(input_schema_spec).dimension_units):
      self._dimension_units = ts.Spec(input_schema_spec).dimension_units
    else:
      self._dimension_units = None

    self._downsampling_factors = []
    self._chain = []
    cur_ds = [1] * decorated_ts.rank
    seq_ds_factors = [cur_ds] + list(self._sequential_downsample_factors)
    for i, sdf in enumerate(seq_ds_factors):
      self._downsampling_factors.append(list(np.array(sdf) * cur_ds))
      cur_ds = self._downsampling_factors[-1]
      self._chain.append(Downsample(sdf, self._downsample_method))

      spec_overrides = self._output_base_spec_overrides.copy()
      spec_overrides['kvstore'] = (
          self._output_base_spec.kvstore / f's{i}').to_json()
      write = Write(
          spec_overrides, keep_existing_chunks=self._keep_existing_chunks)
      self._chain.append(write)

    for c in self._chain:
      if isinstance(c, Writer):
        c.initialize(input_schema_spec, decorated_ts, dryrun)
      decorated_ts = c.decorate(decorated_ts)

    self._initialized = True

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    if not self._initialized:
      raise AssertionError('MultiscaleWrite.initialize must be called first.')
    decorated_ts = input_ts
    for c in self._chain:
      decorated_ts = c.decorate(decorated_ts)
    return decorated_ts

  @property
  def multiscale_spec(self) -> JsonSpec:
    """Construct multiscale spec from base spec."""
    if not self._initialized:
      raise AssertionError('MultiscaleWrite.initialize must be called first.')
    ms_spec = dict(
        multiScale=True,
        downsamplingFactors=self._downsampling_factors,
        axes=self._axes
    )
    if self._dimension_units is not None:
      ms_spec['units'] = [du.base_unit for du in self._dimension_units]
      ms_spec['resolution'] = [du.multiplier for du in self._dimension_units]
    return ms_spec

  def post_jsons(self) -> Iterable[tuple[JsonSpec, ts.KvStore.Spec]]:
    if not self._initialized:
      raise AssertionError('MultiscaleWrite.initialize must be called first.')
    for c in self._chain:
      if isinstance(c, Writer):
        for pj in c.post_jsons():
          yield pj

    multiscale_kvstore = self._output_base_spec.kvstore / 'attributes.json'
    yield self.multiscale_spec, multiscale_kvstore

  def debug_string(self) -> str:
    if not self._initialized:
      raise AssertionError('MultiscaleWrite.initialize must be called first.')
    return (
        '_output_base_spec:\n' +
        pprint.pformat(self._output_base_spec.to_json()) + '\n' +
        'multiscale_spec:\n' +
        pprint.pformat(self.multiscale_spec)
    )


@dataclasses_json.dataclass_json(undefined=dataclasses_json.Undefined.INCLUDE)
@dataclasses.dataclass(frozen=True)
class DecoratorArgs:
  """Empty dataclass to allow automatic parsing of decorator args.

  This precludes the need to define a dataclass for each decorator. All
  undefined fields are included in the resulting python object.
  """

  values: dataclasses_json.CatchAll


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class DecoratorSpec:
  """Decorator specification.

  Attributes:
    name: Name of the decorator.
    args: Arguments for decorator's constructor.
    package: Package where the decorator is defined.
  """

  name: str
  args: DecoratorArgs | None = None
  package: str | None = None


def build_decorator(spec: DecoratorSpec) -> Decorator:
  """Builds a Decorator from a DecoratorSpec."""
  package = spec.package
  if package is None:
    package = 'connectomics.volume.decorators'
  decorator_cls = import_util.import_symbol(spec.name, package)
  args = spec.args.values if spec.args else {}
  return decorator_cls(**args)


@gin.register
class IndexDimByInts(Decorator):
  """Indexes a single dimension by integers loaded from JSON.

  The JSON file is expected to contain a list of integer indices.
  """

  def __init__(self, dim: int | str, json_path: str):
    """Index."""
    super().__init__(context_spec=None)
    self._dim = dim
    self._json_path = json_path

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    indices = json_lib.loads(file.Path(self._json_path).read_text())
    return input_ts[ts.d[self._dim][indices]]


@gin.register
class DfOverF(Decorator):
  """Normalizes fluorescence versus baseline."""

  def __init__(self,
               f0_spec: JsonSpec,
               baseline: float | int,
               context_spec: MutableJsonSpec | None = None):
    """Normalizes fluorescence versus baseline.

    Args:
      f0_spec: Spec of TensorStore containing resting activity, e.g., obtained
        by computing windowed percentiles of low activity.
      baseline: Offset subtracted from f0.
      context_spec: Spec for virtual chunked context overriding its defaults.
    """
    super().__init__(context_spec)
    self._f0_spec = f0_spec
    self._baseline = baseline

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps the input TensorStore with a dF / F0 virtual_chunked."""
    f0_ts = ts.open(self._f0_spec).result()

    def df_over_f_read(domain: ts.IndexDomain, array: np.ndarray,
                       unused_read_params: ts.VirtualChunkedReadParameters):
      f = np.array(input_ts[domain], dtype='float32')
      f0 = np.array(f0_ts[domain], dtype='float32')
      df = f - f0
      f_base = f0 - self._baseline
      f_base[f_base == 0] = 1
      array[...] = df / f_base

    schema = adjust_schema_for_chunksize(
        ts.cast(input_ts, 'float32').schema,
        f0_ts.chunk_layout.read_chunk.shape)
    schema = adjust_schema_for_virtual_chunked(schema)
    return ts.virtual_chunked(
        df_over_f_read, schema=schema, context=self._context)


def _compute_average_evoked_response(
    data: np.ndarray,
    condition_offsets: Sequence[int],
    condition_period_offsets: Sequence[Sequence[int]],
    condition_baselines_exclusive_max: Sequence[int] | None = None,
    axis: int = 0,
    pad_side: str = 'left',
    return_difference: bool = True,
):
  """Computes average evoked response over stimulus repetitions from data.

  Args:
    data: numpy array of shape [timesteps, ...]
    condition_offsets: integers indicating where conditions start and end.
    condition_period_offsets: integers indicating for each condition where
        its repeats start and end.
    condition_baselines_exclusive_max: integers indicating the exclusive max
      applied to periods before baseline computation. For example, this can
      be used to exclude periods in the test set.
    axis: axis over which to subtract average evoked response.
    pad_side: which side of incomplete repeats to pad with nans. Padded nans are
        not in the average but only affect alignment of timesteps.
    return_difference: whether to return the difference between data and evoked
      response, or the evoked response itself.

  Returns:
    average evoked response, or data with average evoked response subtracted.
  """
  if data.shape[axis] != condition_offsets[-1]:
    raise ValueError(
        f'Data length at axis {axis} does not match last condition offset.')
  data = np.swapaxes(data, axis, 0)

  if not condition_baselines_exclusive_max:
    condition_baselines_exclusive_max = [None] * len(condition_period_offsets)
  if len(condition_baselines_exclusive_max) != len(condition_period_offsets):
    raise ValueError(
        'Length of `condition_baselines_exclusive_max` does not match length of'
        ' `condition_period_offsets`.'
    )

  for condition in range(len(condition_offsets) - 1):
    t_start = condition_offsets[condition]
    t_end = condition_offsets[condition + 1]
    period_offsets = condition_period_offsets[condition]
    if not period_offsets:  # no period, subtract mean over condition
      baseline = data[t_start:t_end].mean(axis=0)
      if return_difference:
        data[t_start:t_end] = data[t_start:t_end] - baseline
      else:
        data[t_start:t_end] = baseline
      continue
    assert sum(period_offsets) == t_end - t_start, 'Offset and periods mismatch'

    # collect, pad, and compute period templates for baseline
    periods = []
    t_current = t_start
    period_max = max(period_offsets)
    for period, _ in enumerate(period_offsets):
      period_data = data[t_current:t_current + period_offsets[period]]
      dims = period_data.ndim
      if pad_side == 'left':
        t_padding = (period_max - period_data.shape[0], 0)
      else:
        assert pad_side == 'right'
        t_padding = (0, period_max - period_data.shape[0])
      pad_width = [t_padding] + [(0, 0)] * (dims - 1)
      period_data = np.pad(
          period_data,
          pad_width=pad_width,
          mode='constant',
          constant_values=np.nan
      )
      periods.append(period_data)
      t_current += period_offsets[period]
    assert t_current == t_end
    period = np.nanmean(
        np.stack(periods, axis=0)[
            : condition_baselines_exclusive_max[condition]
        ],
        axis=0,
    )

    # subtract average evoked response from each repeat
    t_current = t_start
    for period_offset in period_offsets:
      baseline = (period[-period_offset:] if pad_side == 'left'
                  else period[:period_offset])
      if return_difference:
        data[t_current:t_current + period_offset] = (
            data[t_current:t_current + period_offset] - baseline)
      else:
        data[t_current:t_current + period_offset] = baseline
      t_current += period_offset
  data = np.swapaxes(data, 0, axis)
  return data


@gin.register
class AverageEvokedResponseFilter(Filter):
  """Computes average evoked response over stimulus repetitions."""

  def __init__(self,
               min_chunksize: Sequence[int] | None = None,
               context_spec: MutableJsonSpec | None = None,
               **filter_args):
    super().__init__(
        filter_fun=_compute_average_evoked_response,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)
