# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Async library for running subprocesses."""

import asyncio
import pathlib
from typing import Sequence

from absl import logging
from connectomics.common import binaries


async def run_task(
    binary: str | pathlib.Path,
    flags: Sequence[str] | binaries.FlagTypes,
    stdout: int = asyncio.subprocess.PIPE,
    stderr: int = asyncio.subprocess.PIPE,
) -> int:
  """Run a binary asynchronously, streaming stdout and stderr.

  Args:
    binary: The binary to run.
    flags: The flags to pass to the binary.
    stdout: The stdout stream to use.
    stderr: The stderr stream to use.

  Returns:
    The exit code of the binary.
  """
  binary = str(binary)
  if isinstance(flags, dict):
    flags = binaries.flags_array(flags)
  proc = await asyncio.create_subprocess_exec(
      binary,
      *flags,
      stdout=stdout,
      stderr=stderr,
  )

  async def _read_stream(
      stream: asyncio.StreamReader, timeout: float = 0.1
  ) -> str | None:
    try:
      while True:
        line = (
            await asyncio.wait_for(stream.readline(), timeout=timeout)
        ).decode('utf-8')
        return line
    except asyncio.TimeoutError:
      return

  complete = False
  final_result = None

  finish = asyncio.create_task(proc.wait(), name='proc.wait')
  stdout = asyncio.create_task(_read_stream(proc.stdout), name='stdout')
  stderr = asyncio.create_task(_read_stream(proc.stderr), name='stderr')

  pending = False
  while not complete:
    done, pending = await asyncio.wait(
        [finish, stdout, stderr], return_when=asyncio.FIRST_COMPLETED
    )
    for task in done:
      if task == finish:
        complete = True
        final_result = finish.result()
        logging.info('Result: %s', final_result)
        continue

      result = task.result()
      if result is not None:
        logging.info('%s: %s', task.get_name(), result)
      if task.get_name() == 'stdout':
        stdout = asyncio.create_task(_read_stream(proc.stdout), name='stdout')
      else:
        stderr = asyncio.create_task(_read_stream(proc.stderr), name='stderr')
  if pending:
    done, pending = await asyncio.wait(pending)
    assert not pending
    for task in done:
      logging.info('%s: %s', task.get_name(), task.result)
  if final_result != 0:
    raise RuntimeError(f'Failed with code {final_result}')
  return final_result
