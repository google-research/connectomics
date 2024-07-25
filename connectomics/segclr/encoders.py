# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
"""Defines JSON encoders."""

import json
from typing import Any
import numpy as np


class NumpyEncoder(json.JSONEncoder):
  """Custom encoder for numpy data types."""

  def default(self, o: Any) -> Any:
    if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                      np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):

      return int(o)

    elif isinstance(o, (np.float16, np.float32, np.float64)):
      return float(o)

    elif isinstance(o, (np.complex64, np.complex128)):
      return {"real": o.real, "imag": o.imag}

    elif isinstance(o, np.ndarray):
      return o.tolist()

    elif isinstance(o, np.bool_):
      return bool(o)

    elif isinstance(o, np.void):
      return None

    return json.JSONEncoder.default(self, o)
