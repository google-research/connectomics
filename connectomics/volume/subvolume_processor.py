"""Base class encapsulating processing a subvolume to another subvolume."""

import collections
import enum
from typing import Tuple, Optional, Union

from connectomics.common import array
from connectomics.common import bounding_box
import numpy as np


SuggestedXyz = collections.namedtuple('SuggestedXyz', 'x y z')
XyzTuple = array.Tuple3i
TupleOrSuggestedXyz = Union[XyzTuple, SuggestedXyz]  # pylint: disable=invalid-name

ImmutableArray = array.ImmutableArray
MutableArray = array.MutableArray


class OutputNums(enum.Enum):
  SINGLE = 1
  MULTI = 2


class SubvolumeProcessor:
  """Abstract base class for processors.

  The self.process method does the work.  The rest is for documenting input /
  output requirements and naming.
  """

  # Effective subvolume/overlap configuration as set by the framework within
  # which this processor is being executed. This might include, e.g. user
  # overrides supplied via command-line arguments.
  _context = ...  # type: ImmutableArray
  _subvol_size = ...  # type: ImmutableArray
  _overlap = ...  # type: ImmutableArray

  # Whether the output of this processor will be cropped for subvolumes that
  # are adjacent to the input bounding box(es).
  crop_at_borders = True

  # If true, the actual content of input_ndarray doesn't matter. The processor
  # only uses the type and geometry of the array for further processing.
  ignores_input_data = False

  def output_type(self, input_type):
    """Returns Numpy output type of self.process for given input_type.

    Args:
      input_type: A Numpy type, should be one of np.uint8, np.uint64,
        np.float32.
    """
    return input_type

  @property
  def output_num(self) -> OutputNums:
    """Whether self.process produces single output or multiple per input."""
    return OutputNums.SINGLE

  @property
  def name_parts(self) -> Tuple[str]:
    """Returns Tuple[str] to be used in naming jobs, outputs, etc.

    Often useful to include both the name of the processor as well as relevant
    parameters.  The elements are generally joined with '_' or '-' depending on
    the context.
    """
    return type(self).__name__,

  def pixelsize(self, input_psize: array.ArrayLike3d) -> ImmutableArray:
    return input_psize

  def num_channels(self, input_channels: int) -> int:
    return input_channels

  def process(
      self, box: bounding_box.BoundingBoxBase, input_ndarray: MutableArray
  ) -> Tuple[bounding_box.BoundingBoxBase, MutableArray]:
    """Processes the input subvolume.

    Args:
      box: The bounding box of the input_ndarray in the global coordinate space
        of the containing volume.
      input_ndarray: 4d Numpy array with data for the input subvolume.

    Returns:
      The box and input_ndarray, processed appropriately.  If self.context is
      > 0, it is expected that the returned box and input_ndarray will be
      smaller than the input by the context amount.
    """
    raise NotImplementedError

  def subvolume_size(self) -> Optional[TupleOrSuggestedXyz]:
    """Returns the XYZ subvolume size required by self.process.

    Some processors (e.g. TF inference models) may require specific input size.
    If the input size is just a suggestion, should return SuggestedXyz rather
    than raw tuple.  If there is no suggested input size, return None.
    """
    return None

  def context(self) -> Tuple[TupleOrSuggestedXyz, TupleOrSuggestedXyz]:
    """Returns XYZ front/back context needed for processing.

    It is expected that the return from self.process will be smaller than the
    input by this amount in front and back.
    """
    return SuggestedXyz(0, 0, 0), SuggestedXyz(0, 0, 0)

  def overlap(self) -> TupleOrSuggestedXyz:
    """Keep the type of context and sum front and back context."""
    f, b = self.context()
    overlap = f[0] + b[0], f[1] + b[1], f[2] + b[2]
    if isinstance(f, SuggestedXyz) and isinstance(b, SuggestedXyz):
      return SuggestedXyz(*overlap)
    return overlap

  def set_effective_subvol_and_overlap(self, subvol_size, overlap):
    self._subvol_size = array.ImmutableArray(subvol_size)
    self._overlap = array.ImmutableArray(overlap)
    if np.all(self.overlap() == self._overlap):
      self._context = self.context()  # pytype: disable=annotation-type-mismatch
    else:
      pre = self._overlap // 2
      post = self._overlap - pre
      self._context = pre, post  # pytype: disable=annotation-type-mismatch

  def _context_for_box(
      self, box: bounding_box.BoundingBoxBase) -> Tuple[np.ndarray, np.ndarray]:
    front, back = self._context
    front = np.array(front)
    back = np.array(back)
    if not self.crop_at_borders:
      front *= ~box.is_border_start
      back *= ~box.is_border_end

    return front, back

  def crop_box(
      self, box: bounding_box.BoundingBoxBase) -> bounding_box.BoundingBoxBase:
    """Crop box front/back by self.context.

    Args:
      box: BoundingBox to crop.

    Returns:
      Copy of box with bounds reduced by front/back amount given by
      self.context.
    """
    front, back = self._context_for_box(box)
    return box.adjusted_by(start=front, end=-back)

  def crop_box_and_data(
      self, box: bounding_box.BoundingBoxBase,
      data: np.ndarray) -> Tuple[bounding_box.BoundingBoxBase, np.ndarray]:
    """Crop data front/back by self.context.

    Args:
      box: bounding box corresponding to the data array
      data: 4d Numpy array with dimensions channels, Z, Y, X.

    Returns:
      View of data cropped by front/back amount given by self.context.
    """
    cropped_box = self.crop_box(box)
    front, back = self._context_for_box(box)
    fx, fy, fz = front
    bx, by, bz = np.array(data.shape[:0:-1]) - back
    return cropped_box, data[:, fz:bz, fy:by, fx:bx]
