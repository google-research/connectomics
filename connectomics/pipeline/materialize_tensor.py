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
"""Pipeline for processing TensorStores with transformations applied.

Currently the orientation is around an existing materialized input TensorStore
that is decorated by a series of virtual_chunked TensorStores and then written
to an output TensorStore.

Example usage:
  python materialize_tensor.py --gin_config config.gin

At the moment this binary only supports Beam's DirectRunner;
distributed Beam runners have not been tested.

Example config.gin:
  run.input_spec = {
    "open" : True,
    "driver" : "n5",
    "kvstore" : "gs://my-bucket/input_path",
  }

  # Max projects on z and t axes (dimensions 2 and 3 in the input).
  run.virtual_decorators = [@z/MaxProjection(), @t/MaxProjection(), @Write()]

  z/MaxProjection.projection_dim = 2
  t/MaxProjection.projection_dim = 2  # 3 in input, but 2 after first z-project.

  Write.output_spec_overrides = {
    "create" : True,
    "open": True,
    "driver": "n5",
    "kvstore" : "gs://my-bucket/output_path",
  }
"""

import pprint

# The Mapping and Sequence under collections.abc don't seem to work with the
# version of beam typehints that piggybacks on typing.
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from absl import app
from absl import flags
from absl.testing import flagsaver
import apache_beam as beam
from connectomics.common import beam_utils
from connectomics.common import bounding_box
from connectomics.common import box_generator
from connectomics.common import gin_utils  # pylint: disable=unused-import
from connectomics.common import ts_utils
from connectomics.pipeline.beam import compute_percentiles  # pylint: disable=unused-import
from connectomics.pipeline.beam import reshard_tensor  # pylint: disable=unused-import
import connectomics.segmentation.process  # pylint: disable=unused-import
from connectomics.volume import decorators  # pylint: disable=unused-import
import gin
import numpy as np
# pylint: disable=unused-import
import sofima.decorators.affine
import sofima.decorators.flow
import sofima.decorators.maps
import sofima.decorators.warp
# pylint: enable=unused-import
import tensorstore as ts

FLAGS = flags.FLAGS
_GIN_CONFIG = flags.DEFINE_multi_string(
    'gin_config', [],
    'List of paths to the config files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [], 'Newline separated list of Gin parameter bindings.')
_GIN_SEARCH_PATHS = flags.DEFINE_multi_string(
    'gin_search_paths', [],
    'List of paths to add to Gin\'s search path.')
_DRYRUN = flags.DEFINE_bool(
    'dryrun', False,
    'Just print the input, decorated, and output tensorstore schemas and exit '
    'without running pipeline. Helpful for interactively building an output '
    'spec.')


JsonSpec = Mapping[str, Any]
MutableJsonSpec = MutableMapping[str, Any]


def _counter(name: str) -> beam.metrics.metricbase.Counter:
  return beam.metrics.Metrics.counter('materialize', name)


def _timer_counter(name: str) -> beam.metrics.metricbase.Counter:
  return beam_utils.timer_counter('materialize', name)


def _get_box_generator(decorated_ts: ts.TensorStore,
                       max_box_bytes: int) -> box_generator.BoxGenerator:
  """Get BoxGenerator to process chunks compatible with input and output."""
  box_size = decorated_ts.chunk_layout.read_chunk.shape
  box_bytes = np.prod(box_size) * decorated_ts.dtype.numpy_dtype.itemsize
  if box_bytes > max_box_bytes:
    raise ValueError(f'Too big a box: {box_bytes} > {max_box_bytes} bytes.')
  outer_box = bounding_box.BoundingBox(
      start=decorated_ts.origin, size=decorated_ts.shape)
  return box_generator.BoxGenerator(outer_box, box_size=box_size)


def materialize_subtensor(box_index: int, box_gen: box_generator.BoxGenerator,
                          decorated_ts: ts.TensorStore) -> None:
  """Copies data from decorated input to output within designated box."""
  _, bbox = box_gen.generate(box_index)
  # BoundingBox assumes numpy C-order so needs to be reversed.
  slc = bbox.to_slice_tuple()[::-1]
  with _timer_counter('time-materialize'):
    _ = decorated_ts[slc].read().result()
    _counter('subtensors-materialized').inc()

  for name, counter in tuple(decorators.COUNTER_STORE.get_nonzero()):
    old_value = counter.reset()
    _counter(name).inc(old_value)


@gin.configurable
def run(input_spec: MutableJsonSpec = gin.REQUIRED,
        virtual_decorators: Sequence[decorators.Decorator] = (),
        max_box_bytes: int = int(4e9),
        pipeline_options: Optional[MutableJsonSpec] = None,
        extra_flags: Optional[Sequence[tuple[str, Any]]] = None):
  """Set up and run Beam subtensor distributed materialization pipeline."""
  if not isinstance(virtual_decorators[-1], decorators.Writer):
    raise ValueError('Missing write after last operation.')

  write_encountered = False
  for vd in virtual_decorators:
    if isinstance(vd, decorators.Writer):
      write_encountered = True
    if vd.requires_context and write_encountered:
      raise ValueError('Write chained before an operation requiring context. '
                       'This will cause write amplification.')

  input_ts = ts.open(input_spec).result()
  input_schema_spec = dict(
      driver=input_spec['driver'],
      schema=input_ts.schema.to_json(),
  )
  if _DRYRUN.value:
    print('input_schema_spec:')
    pprint.pprint(input_schema_spec)

  decorated_ts = input_ts
  for vd in virtual_decorators:
    if isinstance(vd, decorators.Writer):
      vd.initialize(input_schema_spec, decorated_ts, _DRYRUN.value)
    decorated_ts = vd.decorate(decorated_ts)
    if _DRYRUN.value:
      print(f'{vd.__class__.__name__} decorated_ts.schema:')
      pprint.pprint(decorated_ts.schema.to_json())
      vd.print_debug_string()

  box_gen = _get_box_generator(decorated_ts, max_box_bytes)
  if _DRYRUN.value:
    print('box_gen:', box_gen)
    return

  def materialize(root) -> None:
    materialize_complete = (
        root
        | 'boxes' >> beam.Create(range(box_gen.num_boxes))
        | beam.Map(materialize_subtensor, box_gen, decorated_ts))

    post_jsons = []
    for vd in virtual_decorators:
      if isinstance(vd, decorators.Writer):
        post_jsons.extend(vd.post_jsons())
    _ = (
        root
        | 'jsons' >> beam.Create(post_jsons)
        | beam_utils.MustFollow(materialize_complete)
        | beam.MapTuple(ts_utils.write_json)
    )

  if pipeline_options is None:
    pipeline_options = {}

  extra_flags = extra_flags if extra_flags is not None else {}
  with flagsaver.flagsaver(**extra_flags):
    with beam.Pipeline(
        options=beam.options.pipeline_options.PipelineOptions.from_dictionary(
            pipeline_options)) as p:
      materialize(p)


@gin.configurable
def multirun(steps: Optional[Any] = None):
  if steps is None:
    steps = [run]
  for n, step in enumerate(steps):
    print(f'step {n+1}/{len(steps)}')
    if _DRYRUN.value and step.__name__ != run.__name__:
      print('dry run; step skipped.')
      continue
    step()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  for path in _GIN_SEARCH_PATHS.value:
    gin.add_config_file_search_path(path)
  gin.parse_config_files_and_bindings(_GIN_CONFIG.value, _GIN_BINDINGS.value)
  multirun()


if __name__ == '__main__':
  app.run(main)
