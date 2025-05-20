# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Compute percentiles in a 4D tensor.

Given an XYZT scalar tensor, computes the percentile along the T axis.
"""

import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import gin


def get_window(location, radius, length):
  """Computes the start and end of a window centered at location.

  Args:
    location: location to be centered.
    radius: radius of window.
    length: length of sequence.
  Returns:
    start: starting index.
    end: ending index (exclusive).
  """
  start = None
  end = None
  if location < radius:
    start = 0
    end = min(length, 2 * radius + 1)
    return start, end
  if location + radius >= length - 1:
    start = length - 2 * radius - 1
    end = length
    return start, end
  start = max(0, location - radius)
  end = location + radius + 1
  return start, end


class ComputePercentile(beam.core.DoFn):
  """Computes the percentiles of a TS."""

  def __init__(self, input_spec, output_spec, radius,
               percentiles, max_workers):
    self._input_spec = input_spec
    self._output_spec = output_spec
    self._radius = radius
    self._percentiles = percentiles
    self._num_percentiles = len(percentiles)
    self._max_workers = max_workers

  def setup(self):
    """Sets up the beam bundle."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import tensorstore as ts
    self._ds_in = ts.open(self._input_spec).result()
    self._shape = self._ds_in.domain.shape
    self._dtype = self._ds_in.dtype.numpy_dtype
    self._ds_out = ts.open(self._output_spec).result()

  def process(self, yz):
    """Computes percentiles at a given y and z."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import concurrent.futures
    import numpy as np
    y, z = yz
    # Process entire xt tiles at once.
    tile = self._ds_in[:, y, z, :].read().result()
    staging = np.zeros((self._shape[0], self._shape[3],
                        self._num_percentiles),
                       dtype=self._dtype)
    def compute(t):
      start, end = get_window(t, self._radius, self._shape[3])
      ptile = np.percentile(tile[:, start:end], self._percentiles,
                            axis=1, interpolation="nearest").transpose()
      staging[:, t, :] = ptile
    with concurrent.futures.ThreadPoolExecutor(self._max_workers) as executor:
      _ = executor.map(compute, range(self._shape[3]))
    self._ds_out[:, y, z, :, :] = staging
    yield None


@gin.configurable("compute_percentiles")
def compute_percentiles(pipeline_options=gin.REQUIRED,
                        input_spec=gin.REQUIRED,
                        output_spec=gin.REQUIRED,
                        radius=gin.REQUIRED,
                        percentiles=gin.REQUIRED,
                        max_workers=None):
  """Computes the percentile in a window of T major voxels.

  Args:
    pipeline_options: dictionary of pipeline options
    input_spec: Tensorstore input spec.
    output_spec: Tensorstore output spec.
    radius: the radius over which to compute the percentile.
    percentiles: Percentiles to compute.
    max_workers: Maximum number of worker threads used for computation.
  """
  # pylint: disable=g-import-not-at-top, import-outside-toplevel
  import tensorstore as ts
  pipeline_opt = PipelineOptions.from_dictionary(pipeline_options)
  logging.info(pipeline_opt.get_all_options())
  ds = ts.open(input_spec).result()
  shape = ds.domain.shape
  yz = []
  for y in range(shape[1]):
    for z in range(shape[2]):
      yz.append((y, z))

  with beam.Pipeline(options=pipeline_opt) as p:
    ys = p | beam.Create(yz)
    result = ys | beam.ParDo(ComputePercentile(input_spec, output_spec, radius,
                                               percentiles, max_workers))
  del result
