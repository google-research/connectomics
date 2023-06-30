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
"""OpenCV utilities.

Usage for affine alignment of 2D images:
    _, transform = opencv_utils.optim_transform(fix, mov)
    res = opencv_utils.warp_affine(mov, transform)

Note:
  OpenCV uses [x,y]-convention for images, where x runs horizontal,
  and y vertical, with origin o at the top-left:

    o → x
    ↓ . .
    y . .

  When using OpenCV with other libraries, this may need to be accounted for.
  For instance, skimage (except `skimage.transforms`) uses [r,c]-convention,
  where rows r index into y, and columns c index into x:

    o → c
    ↓ . .
    r . .
"""

from typing import Optional
import warnings

import cv2 as cv
import numpy as np

_BORDER_MODES_ = {
    'constant': cv.BORDER_CONSTANT,
    'replicate': cv.BORDER_REPLICATE,
    'reflect': cv.BORDER_REFLECT,
    'wrap': cv.BORDER_WRAP,
}

_INTERPOLATIONS_FLAGS_ = {
    'nearest': cv.INTER_NEAREST,
    'linear': cv.INTER_LINEAR,
    'cubic': cv.INTER_CUBIC,
    'area': cv.INTER_AREA,
    'lanczos': cv.INTER_LANCZOS4,
    'linear_exact': cv.INTER_LINEAR_EXACT,
}

_MOTION_FLAGS_ = {
    'translation': cv.MOTION_TRANSLATION,
    'euclidean': cv.MOTION_EUCLIDEAN,
    'affine': cv.MOTION_AFFINE,
}


def optim_transform(
    fix: np.ndarray,
    mov: np.ndarray,
    eps: float = 1e-5,
    num_iterations: int = 5000,
    mode: str = 'affine',
    transform_initial: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    warn: bool = False,
) -> tuple[float, np.ndarray]:
  """Finds transform to align 2D images by ECC optimization.

  Uses OpenCV's `cv.findTransformECC` to find a transformation that aligns
  `mov` with `fix`. If `mov` has indices [xm, ym], and `fix` [xf, yf], then
  the resulting transformation matrix T can be used to warp:

    |xf|       |xm|
    |yf| = T * |ym|
               |1 |

  Note that `eps` and `num_iterations` may need problem-specific adjustment.

  Associated publication:
    Evangelidis, Georgios D., and Emmanouil Z. Psarakis. 2008. "Parametric Image
    Alignment Using Enhanced Correlation Coefficient Maximization." IEEE
    Transactions on Pattern Analysis and Machine Intelligence 30 (10): 1858–65.

  Args:
    fix: Fixed 2D image, [x,y]-convention, uint8 or float32.
    mov: Moving 2D image, [x,y]-convention, uint8 or float32.
    eps: Epsilon used for termination criterion.
    num_iterations: Number of iterations.
    mode: One of `translation`, `euclidean`, `affine`.
    transform_initial: Optional initial transform to start optimization from.
      If no value is passed, an identity transform is used.
    mask: Optional mask to indicate valid values of the moving image.
    warn: If `True`, returns NaN correlation coefficient and
      transformation matrix if the optimization fails.

  Returns:
    Correlation coefficient and 2x3 transformation matrix T.

  Raises:
    RuntimeError: If optimisation fails and `warn` is `False`.
  """
  assert fix.ndim == mov.ndim and fix.ndim == 2, (
      '`fix` and `mov` must both be 2D, ' +
      f'but `fix` is {fix.ndim}D and `mov` is {mov.ndim}D.')
  assert fix.shape == mov.shape, (
      'shapes of `fix` and `mov` must be equal, ' +
      f'but `fix` has shape {fix.shape} and `mov` has shape {mov.shape}.')
  assert fix.dtype == mov.dtype and fix.dtype in (np.uint8, np.float32), (
      '`fix` and `mov` must have equal dtypes, either uint8 or float32, ' +
      f'but `fix` has dtype {fix.dtype} and `mov` has dtype {mov.dtype}.')
  assert mode in _MOTION_FLAGS_, (
      f'`mode` must be one of {_MOTION_FLAGS_.keys()}, but is {mode}.')
  assert transform_initial is None or transform_initial.shape == (2, 3), (
      'if passed, `transform_initial` must have shape (2, 3), '
      f'but has shape {transform_initial.shape}.')
  try:
    return cv.findTransformECC(
        templateImage=mov,
        inputImage=fix,
        warpMatrix=transform_initial,
        motionType=_MOTION_FLAGS_[mode],
        criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, num_iterations,
                  eps),
        inputMask=mask)
  except cv.error as e:
    if warn:
      warnings.warn(f'OpenCV exception: {e}. Returning NaNs.')
      return np.nan, np.full((2, 3), np.nan)
    else:
      raise RuntimeError(f'OpenCV exception: {e}.') from e


def warp_affine(
    img: np.ndarray,
    transform: np.ndarray,
    interpolation: str = 'lanczos',
    invert: bool = False,
    border: str = 'constant',
    constant_value: int = 0,
) -> np.ndarray:
  """Applies an affine transformation to a 2D image.

  Args:
    img: Image to transform, [x,y]-convention.
    transform: 2x3 transformation matrix.
    interpolation: Interpolation method, can be `nearest`, `linear`, `cubic`,
      `area`, `lanczos`, or `linear_exact`. See OpenCV docs for
      `cv::InterpolationFlags` for details.
    invert: If set, the transform is inverted with `cv::invertAffineTransform`
      before application.
    border: Defines extrapolation behavior. Can be `constant`, `replicate`,
      `reflect`, or `warp`. See `cv::BorderTypes` for details.
    constant_value: Value used if `border` equals `constant`.

  Returns:
    Transformed version of image, same dtype.
  """
  assert img.ndim == 2, f'`img` must be 2D, but is {img.ndim}D.'
  assert isinstance(transform, np.ndarray), (
      f'`transform` must be numpy array, but is {type(transform)}.')
  assert transform.dtype in (np.float32, np.float64), (
      f'`transform` must be float32 or float64, but is {transform.dtype}.')
  assert interpolation in _INTERPOLATIONS_FLAGS_, (
      f'`interpolation` must be one of {_INTERPOLATIONS_FLAGS_.keys()}, ' +
      f'but is {interpolation}.')
  assert border in _BORDER_MODES_, (
      f'`border` must be one of {_BORDER_MODES_.keys()}, ' +
      f'but is {border}.')
  flags = _INTERPOLATIONS_FLAGS_[interpolation]
  if invert:
    flags += cv.WARP_INVERSE_MAP
  return cv.warpAffine(
      img,
      transform,
      dsize=img.shape[::-1],
      flags=flags,
      borderMode=_BORDER_MODES_[border],
      borderValue=constant_value)
