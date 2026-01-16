# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import coralnpu_v2_sim_pybind
from elftools.elf.elffile import ELFFile
import numpy as np


class CoralNPUV2Simulator:
  """Wrapper for CoralNPUV2SimulatorPy providing helper methods."""

  def __init__(self):
    self.options = coralnpu_v2_sim_pybind.CoralNPUV2SimulatorOptions()
    self.sim = coralnpu_v2_sim_pybind.CoralNPUV2SimulatorPy(self.options)

  def load_program(self, elf_path, entry_point=None):
    """Loads an ELF program.

    If entry_point is None, it's inferred from ELF if not provided.
    """
    # Note: The underlying C++ API might expect an optional or raw value.
    # Based on previous code: LoadProgrampy(path, entry_point)
    self.sim.LoadProgram(elf_path, entry_point)

  def run(self):
    """Runs the simulator."""
    self.sim.Run()

  def wait(self):
    """Waits for the simulator to finish."""
    self.sim.Wait()

  def step(self, num_steps):
    """Steps the simulator."""
    return self.sim.Step(num_steps)

  def get_cycle_count(self):
    """Returns the cycle count."""
    return self.sim.GetCycleCount()

  def read_memory(self, address, length):
    """Reads memory and returns a numpy array of uint8."""
    return self.sim.ReadMemory(address, length)

  def write_memory(self, address, data):
    """Writes data to memory. Data must be a numpy array."""
    if not isinstance(data, np.ndarray):
      raise TypeError('data must be a numpy array')
    if data.dtype != np.uint8:
      data = data.view(np.uint8)
    self.sim.WriteMemory(address, data, len(data))

  def get_elf_entry_and_symbol(self, filename, symbol_names):
    """Returns the entry point and a dictionary of symbol addresses from an ELF file."""
    symbol_map = {}
    with open(filename, 'rb') as f:
      elf_file = ELFFile(f)
      entry_point = elf_file.header['e_entry']
      symtab_section = next(elf_file.iter_sections(type='SHT_SYMTAB'))
      for symbol_name in symbol_names:
        symbol = symtab_section.get_symbol_by_name(symbol_name)
        if symbol:
          symbol_map[symbol_name] = symbol[0].entry['st_value']
        else:
          symbol_map[symbol_name] = 0
      return entry_point, symbol_map
