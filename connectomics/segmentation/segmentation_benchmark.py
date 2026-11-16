# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
"""Microbenchmarks for connectomics.segmentation.labels and rag."""

from collections.abc import Callable
import gc
import tracemalloc

from connectomics.segmentation import labels
from connectomics.segmentation import rag
import google_benchmark
import numpy as np


def _make_block_volume(
    side: int, block: int, dtype: np.dtype, include_zero: bool = True
) -> np.ndarray:
  """Builds a deterministic 3D block segmentation volume."""
  coords = np.arange(side, dtype=np.uint64) // block
  n_blocks = (side + block - 1) // block
  zz, yy, xx = np.meshgrid(coords, coords, coords, indexing='ij')
  vol = zz * (n_blocks * n_blocks) + yy * n_blocks + xx + 1
  if include_zero:
    vol[::block, :, :] = 0
  return vol.astype(dtype)


def _run_with_memory_stats(
    state: google_benchmark.State,
    vol: np.ndarray,
    fn: Callable[[], object],
) -> None:
  """Runs timed benchmark iterations followed by an unmeasured memory pass."""
  while state:
    fn()
  state.items_processed = state.iterations * vol.size
  state.bytes_processed = state.iterations * vol.nbytes

  gc.collect()
  was_tracing = tracemalloc.is_tracing()
  if not was_tracing:
    tracemalloc.start()
  else:
    tracemalloc.reset_peak()
  base_bytes, _ = tracemalloc.get_traced_memory()
  try:
    result = fn()
    end_bytes, peak_bytes = tracemalloc.get_traced_memory()
    out_bytes = max(0, end_bytes - base_bytes)
    peak_delta = max(0, peak_bytes - base_bytes)
    del result
  finally:
    if not was_tracing:
      tracemalloc.stop()

  state.counters['peak_mem'] = google_benchmark.Counter(
      float(peak_delta),
      google_benchmark.Counter.kDefaults,
      google_benchmark.Counter.kIs1024,
  )
  state.counters['temp_mem'] = google_benchmark.Counter(
      float(max(0, peak_delta - out_bytes)),
      google_benchmark.Counter.kDefaults,
      google_benchmark.Counter.kIs1024,
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def relabel_dense_uint32_128(state: google_benchmark.State) -> None:
  vol = _make_block_volume(128, 8, np.dtype(np.uint32))
  orig_ids = np.unique(vol)
  new_ids = (orig_ids * 3) % (orig_ids.max() + 1)
  new_ids[0] = 0
  _run_with_memory_stats(
      state, vol, lambda: labels.relabel(vol, orig_ids, new_ids)
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def relabel_dense_uint64_256(state: google_benchmark.State) -> None:
  vol = _make_block_volume(256, 8, np.dtype(np.uint64))
  orig_ids = np.unique(vol)
  new_ids = (orig_ids * 3) % (orig_ids.max() + 1)
  new_ids[0] = 0
  _run_with_memory_stats(
      state, vol, lambda: labels.relabel(vol, orig_ids, new_ids)
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def relabel_sparse_uint64_128(state: google_benchmark.State) -> None:
  vol = _make_block_volume(128, 8, np.dtype(np.uint64))
  vol[vol > 0] += 1 << 40
  orig_ids = np.unique(vol)
  new_ids = np.arange(orig_ids.size, dtype=np.uint64)
  _run_with_memory_stats(
      state, vol, lambda: labels.relabel(vol, orig_ids, new_ids)
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def split_disconnected_components_uint64_128(
    state: google_benchmark.State,
) -> None:
  vol = _make_block_volume(128, 8, np.dtype(np.uint64), include_zero=True)
  _run_with_memory_stats(
      state, vol, lambda: labels.split_disconnected_components(vol)
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def split_disconnected_components_no_bg_uint64_128(
    state: google_benchmark.State,
) -> None:
  vol = _make_block_volume(128, 8, np.dtype(np.uint64), include_zero=False)
  _run_with_memory_stats(
      state, vol, lambda: labels.split_disconnected_components(vol)
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def rag_from_subvolume_uint16_128(state: google_benchmark.State) -> None:
  vol = _make_block_volume(128, 8, np.dtype(np.uint16))
  _run_with_memory_stats(state, vol, lambda: rag.from_subvolume(vol))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def rag_from_subvolume_uint64_128(state: google_benchmark.State) -> None:
  vol = _make_block_volume(128, 8, np.dtype(np.uint64))
  _run_with_memory_stats(state, vol, lambda: rag.from_subvolume(vol))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def rag_from_subvolume_uint32_256(state: google_benchmark.State) -> None:
  vol = _make_block_volume(256, 8, np.dtype(np.uint32))
  _run_with_memory_stats(state, vol, lambda: rag.from_subvolume(vol))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def rag_from_subvolume_uint64_256(state: google_benchmark.State) -> None:
  vol = _make_block_volume(256, 8, np.dtype(np.uint64))
  _run_with_memory_stats(state, vol, lambda: rag.from_subvolume(vol))


if __name__ == '__main__':
  google_benchmark.main()
