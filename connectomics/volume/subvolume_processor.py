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
"""Base class encapsulating processing a subvolume to another subvolume."""

import collections
import dataclasses
import enum
import importlib
import inspect
import logging
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar, Union

from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.common import counters
from connectomics.common import file
from connectomics.common import utils
from connectomics.volume import descriptor
from connectomics.volume import mask
from connectomics.volume import metadata
from connectomics.volume import subvolume
import dataclasses_json
import numpy as np

Subvolume = subvolume.Subvolume
SuggestedXyz = collections.namedtuple('SuggestedXyz', 'x y z')
TupleOrSuggestedXyz = Union['XyzTuple', SuggestedXyz]  # pylint: disable=invalid-name
XyzTuple = array.Tuple3i
SubvolumeOrMany = Union[Subvolume, List[Subvolume]]

COUNTER_STORE = counters.ThreadsafeCounterStore()
counter = COUNTER_STORE.get_counter
timer_counter = COUNTER_STORE.timer_counter


@dataclasses.dataclass(frozen=True)
class ProcessingConfig(dataclasses_json.DataClassJsonMixin):
  overlap: list[int] | None = None
  # None defaults to the processor's subvolume_size.
  subvolume_size: list[int] | None = None


def dataclass_configuration(cls: ...) -> Optional[Type]:  # pylint:disable=g-bare-generic
  init_params = inspect.signature(cls.__init__).parameters
  if 'config' in init_params:
    # TODO(timblakely): Also check to see if the class is an actual dataclass.
    return init_params['config'].annotation


# TODO(timblakely): Remove this legacy configuration.
@dataclasses.dataclass
class SubvolumeProcessorConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for a given subvolume processor."""
  # Name of class exposed in module_search_path.
  name: str

  # Arguments to SubvolumeProcessor, passed in as kwargs.
  args: Optional[dict[str, Any]] = None

  # Fully.qualified.python.module to search for SubvolumeProcessor `name`.
  module_search_path: str = 'connectomics.volume.processor'


@dataclasses.dataclass
class ProcessVolumeConfig(dataclasses_json.DataClassJsonMixin):
  """User-supplied configuration."""

  # Input volume to process.
  input_volume: descriptor.VolumeDescriptor = dataclasses.field(
      metadata=dataclasses_json.config(
          decoder=file.dataclass_loader(descriptor.VolumeDescriptor)))

  # Output directory to write the volumetric data, inserted automatically into
  # the output_volume's TensorStore spec.
  output_dir: str

  # Processor configuration to apply.
  processor: SubvolumeProcessorConfig

  # Size of each subvolume to process.
  subvolume_size: array.Tuple3i = dataclasses.field(
      metadata=dataclasses_json.config(decoder=tuple))

  # Bounding boxes to process.
  bounding_boxes: Optional[list[bounding_box.BoundingBox]] = None

  # Overlap between neighboring subvolumes. If not specified, will fall back to
  # the overlap determined by the processor.
  overlap: Optional[array.Tuple3i] = dataclasses.field(
      default=None,
      metadata=dataclasses_json.config(
          decoder=lambda x: None if x is None else tuple(x)))

  # Number of bounding boxes to batch together per work item during processing.
  batch_size: int = 1

  # Additional Tensorstore context configuration applied to the input. Useful
  # for limiting parallelism.
  input_ts_context: Optional[dict[str, Any]] = None

  # Additional Tensorstore context configuration applied to the output. Useful
  # for limiting parallelism.
  output_ts_context: Optional[dict[str, Any]] = None

  # TODO(timblakely): Support back shifting edge boxes.


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
  _context: tuple[np.ndarray, np.ndarray]
  _subvol_size: np.ndarray
  _overlap: np.ndarray
  # Whether the output of this processor will be cropped for subvolumes that
  # are adjacent to the input bounding box(es).
  crop_at_borders = True

  # If true, the actual content of input_ndarray doesn't matter. The processor
  # only uses the type and geometry of the array for further processing.
  ignores_input_data = False

  # Namespace to use for counters. Overridable by subclasses; defaults to
  # kebab-case of the class name.
  @property
  def namespace(self) -> str:
    return utils.pascal_to_kebab(type(self).__name__)

  def output_type(self, input_type: Union[np.uint8, np.uint64, np.float32]):
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

  def pixelsize(self, input_psize: array.ArrayLike3d) -> np.ndarray:
    return np.asarray(input_psize)

  def num_channels(self, input_channels: int) -> int:
    return input_channels

  def process(
      self, subvol: subvolume.Subvolume
  ) -> SubvolumeOrMany:
    """Processes the input subvolume.

    Args:
      subvol: Subvolume to process.

    Returns:
      The processed subvolume. If self.context is > 0, it is expected that the
      returned subvolume will be smaller than the input by the context amount.
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

  def set_effective_subvol_and_overlap(self, subvol_size: array.ArrayLike3d,
                                       overlap: array.ArrayLike3d):
    """Assign the effective subvolume and overlap."""
    self._subvol_size = np.asarray(subvol_size)
    self._overlap = np.asarray(overlap)
    if np.all(self.overlap() == self._overlap):
      self._context = tuple([np.asarray(c) for c in self.context()])
    else:
      pre = self._overlap // 2
      post = self._overlap - pre
      self._context = pre, post

  def _context_for_box(
      self, box: bounding_box.BoundingBoxBase) -> Tuple[np.ndarray, np.ndarray]:
    front, back = self._context
    front = np.array(front)
    back = np.array(back)
    if not self.crop_at_borders:
      front *= ~box.is_border_start
      back *= ~box.is_border_end

    return front, back

  def expected_output_box(
      self, box: bounding_box.BoundingBoxBase) -> bounding_box.BoundingBoxBase:
    """Returns the adjusted bounding box after process() is called.

    Note that this is a basic implementation. Subclasses are free to override
    this function.

    Args:
        box: Size of the input subvolume passed to process()

    Returns:
        Bounding box for the output volume.
    """
    scale_factor = 1 / self.pixelsize(np.repeat(1, len(box.size)))
    cropped_box = self.crop_box(box)
    return cropped_box.scale(list(scale_factor))

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
      self,
      box: bounding_box.BoundingBoxBase,
      # TODO(timblakely): Strongly type this as ArrayCZYX
      data: np.ndarray
  ) -> Subvolume:
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
    return Subvolume(data[:, fz:bz, fy:by, fx:bx], cropped_box)

  # TODO(timblakely): Correct the return value from Any.
  def _open_volume(
      self,
      path_or_volume: (
          file.PathLike | metadata.VolumeMetadata | metadata.DecoratedVolume
      ),
  ) -> Any:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def _get_mask_configs(
      self, mask_configs: str | mask.MaskConfigs
  ) -> mask.MaskConfigs:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def _get_metadata(self, path: file.PathLike) -> metadata.VolumeMetadata:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def _build_mask(
      self,
      mask_configs: mask.MaskConfigs,
      box: bounding_box.BoundingBoxBase,
  ) -> Any:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )


def get_processor(config: SubvolumeProcessorConfig) -> SubvolumeProcessor:
  name = config.name
  package = importlib.import_module(config.module_search_path)
  if not hasattr(package, name):
    raise ValueError(f'No processor named {name} in package {package}')
  processor = getattr(package, name)
  args = {} if not config.args else config.args
  return processor(**args)


class DefaultConfigType(enum.Enum):
  EM_2D = 'em_2d'
  # TODO(timblakely): Support additional configuration.
  # EM_3D = 'em_3d'
  # LICONN = 'liconn'


_KNOWN_DEFAULT_PROCESSOR_CONFIGS = {}


T = TypeVar('T', bound=utils.IsDataclass)


def register_default_config(
    config_type: DefaultConfigType,
    config_class: Type[utils.IsDataclass],
    config_fn: Callable[[dict[str, Any] | None], utils.IsDataclass],
):
  """Registers a default configuration for a given config type and class."""
  config_type_map = _KNOWN_DEFAULT_PROCESSOR_CONFIGS.setdefault(config_type, {})
  if config_class in config_type_map:
    logging.warning(
        'Default config for %s already registered for %s overwriting.',
        config_class,
        config_type,
    )
  config_type_map[config_class] = config_fn


def default_config(
    config_class: Type[T],
    config_type: DefaultConfigType | None = None,
    overrides: file.PathLike | dict[str, Any] | None = None,
    fallback_to_em_2d: bool = True,
) -> T:
  """Returns a default configuration for a given config type and class."""
  if overrides and not isinstance(overrides, dict):
    try:
      return file.dataclass_from_serialized(config_class, overrides)
    except KeyError:
      pass
  if not overrides:
    overrides = None
  if isinstance(overrides, file.PathLike):
    overrides = file.load_json(overrides)
  if config_type is None and fallback_to_em_2d:
    logging.warning('No default config type specified, falling back to EM_2D.')
    config_type = DefaultConfigType.EM_2D
  if config_type not in _KNOWN_DEFAULT_PROCESSOR_CONFIGS:
    raise ValueError(f'No default configurations available for {config_type}')
  default_map = _KNOWN_DEFAULT_PROCESSOR_CONFIGS[config_type]
  if config_class not in default_map:
    raise ValueError(f'No default config for {config_class} for {config_type}')
  return default_map[config_class](overrides)
