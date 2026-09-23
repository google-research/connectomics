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
"""Segmentation post-processing."""

from typing import Any, MutableMapping

from absl import logging
from connectomics.common import file
from connectomics.common import utils
from connectomics.metrics import ellipticity as metrics_el
from connectomics.segmentation import labels
import gin
import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.measure
import tensorstore as ts
import tifffile

MutableJsonSpec = MutableMapping[str, Any]


def compute_segmentation_props(
    segmentation: np.ndarray,
    compress_image: bool = False
) -> pd.DataFrame:
  """Computes per-segment properties.

  Args:
    segmentation: 3D segmentation with XYZ axis order.
    compress_image: If set, uses `serialize_array` to compress the image
      footprints of segments. The resulting dataframe stores images under the
      `image_gz` key rather than `image`.

  Returns:
    Dataframe with segmentation properties.
  """
  df = pd.DataFrame(skimage.measure.regionprops_table(
      segmentation,
      properties=('label', 'area', 'extent', 'centroid', 'bbox', 'image'),
      extra_properties=(metrics_el.compute_ellipticity,)))
  df = df.rename(columns={
      'bbox-0': 'bbox_x0',
      'bbox-1': 'bbox_y0',
      'bbox-2': 'bbox_z0',
      'bbox-3': 'bbox_x1',
      'bbox-4': 'bbox_y1',
      'bbox-5': 'bbox_z1',
      'centroid-0': 'centroid_x',
      'centroid-1': 'centroid_y',
      'centroid-2': 'centroid_z',
      'compute_ellipticity': 'ellipticity'})
  if compress_image:
    df['image_gz'] = df['image'].apply(
        lambda row: utils.serialize_array(row, compression=-1).decode('latin-1')
    )
    del df['image']
  return df


@gin.configurable
def analyze_segmentation(input_spec: MutableJsonSpec = gin.REQUIRED,
                         output_path: str = gin.REQUIRED):
  """Analyzes segmentation.

  Computes per-segment properties and stores results as DataFrame.
  For now, this is reasonably fast and runs in a single process.

  Args:
    input_spec: TensorStore input spec.
    output_path: Path to output DataFrame.
  """
  ds_in = ts.open(input_spec).result()

  df = compute_segmentation_props(
      ds_in[...].read().result(), compress_image=True)
  logging.info('Dataframe head:')
  logging.info(df.head())

  with file.Path(output_path).open('w') as fh:
    df.to_json(fh)  # pytype: disable=wrong-arg-types  # pandas-drop-duplicates-overloads


def _erode(
    img: np.ndarray,
    num_erosions: int,
    min_count: int = 0) -> np.ndarray:
  """Performs multiple erosions; ensures a minimum count of voxels remains."""
  for _ in range(num_erosions):
    res = scipy.ndimage.binary_erosion(img)
    if np.sum(res) < min_count:
      break
    img = res
  return img


@gin.configurable
def filter_labels(
    input_spec: MutableJsonSpec = gin.REQUIRED,
    output_spec: MutableJsonSpec = gin.REQUIRED,
    query: str = gin.REQUIRED,
    relabel: bool = False,
    num_erosions: int = 0,
    erosions_min_voxels: int = 0,
    mask_spec: MutableJsonSpec | None = None,
    mask_value: int = 1):
  """Filters a labelled segmentation based on regionsprops query/mask.

  Args:
    input_spec: Input segmentation spec.
    output_spec: Output segmentation spec.
    query: Query to apply to the segmentation properties dataframe.
    relabel: If set, relabels the output segmentation to start from 1.
    num_erosions: If >0, performs binary erosion this many times.
    erosions_min_voxels: Minimum number of voxels that remain after erosion.
    mask_spec: If set, applies this mask to the input segmentation before
      filtering.
    mask_value: Value of the mask to keep.
  """
  ds_in = ts.open(input_spec).result()
  seg = ds_in.read().result()

  if mask_spec:
    mask_ds = ts.open(mask_spec).result()
    mask = mask_ds.read().result()
    seg = seg * (mask == mask_value)

  df = compute_segmentation_props(seg)
  keep = df.query(query).reset_index()

  out = np.zeros_like(ds_in)
  for i, row in keep.iterrows():
    img = row['image']
    if num_erosions > 0:
      img = _erode(
          img, num_erosions=num_erosions, min_count=erosions_min_voxels)
    coords = np.array(
        [row['bbox_x0'], row['bbox_y0'], row['bbox_z0']]
    ) + np.argwhere(img)
    out[coords[:, 0], coords[:, 1], coords[:, 2]] = (
        row['label'] if not relabel else (i + 1))  # type: ignore

  ds_out = ts.open(output_spec).result()
  ds_out[...] = out


@gin.configurable
def filter_mask(
    input_spec: MutableJsonSpec = gin.REQUIRED,
    output_spec: MutableJsonSpec = gin.REQUIRED,
    query: str = gin.REQUIRED,
    num_iter_dilation: int = 0):
  """Filters a mask based on regionsprops query."""
  ds_in = ts.open(input_spec).result()
  mask = ds_in.read().result()
  label, _ = scipy.ndimage.label(mask)
  del mask

  df = compute_segmentation_props(label)
  keep = df.query(query)

  out = np.zeros_like(ds_in)
  for _, row in keep.iterrows():
    coords = np.array(
        [row['bbox_x0'], row['bbox_y0'], row['bbox_z0']]
    ) + np.argwhere(row['image'])
    out[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0

  if num_iter_dilation > 0:
    out = scipy.ndimage.binary_dilation(out, iterations=num_iter_dilation)

  ds_out = ts.open(output_spec).result()
  ds_out[...] = out


def recompute_connected_components(
    seg: np.ndarray, offset: int = 0) -> np.ndarray:
  """Recomputes connected components."""
  out = labels.split_disconnected_components(seg)
  out[out > 0] += offset
  return out


@gin.configurable
def ingest_tiff_segmentation(
    tiff_path: str = gin.REQUIRED,
    output_spec: MutableJsonSpec = gin.REQUIRED,
    offset: int = 0,
    transpose: bool = False):
  """Ingests segmentation from TIFF."""
  seg = tifffile.imread(file.Path(tiff_path).open('rb'))

  out = recompute_connected_components(seg, offset=offset)
  if transpose:
    out = out.transpose()

  ds = ts.open(output_spec).result()
  ds[...] = out

  num_unique_labels = len(np.unique(out)[1:])
  logging.info('Wrote segmentation with %d unique labels.', num_unique_labels)


@gin.configurable
def write_boundary_mask_to_tensorstore(
    output_spec: MutableJsonSpec = gin.REQUIRED,
    shape: tuple[int, int, int] = gin.REQUIRED,
    before_xyz: tuple[int, int, int] = (0, 0, 0),
    after_xyz: tuple[int, int, int] = (0, 0, 0)):
  """Writes a boundary mask volume to a TensorStore."""
  out = ts.open(output_spec).result()
  out[...] = labels.create_boundary_mask_volume(shape, before_xyz, after_xyz)
