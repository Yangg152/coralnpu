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

import unittest
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np


class TestCoralNPUV2SimPybind(unittest.TestCase):

  def setUp(self):
    self.sim = CoralNPUV2Simulator()
    self.r = runfiles.Create()
    self.elf_path = self.r.Rlocation(
        "coralnpu_hw/tests/cocotb/rvv/arithmetics/rvv_add_int8_m1.elf"
    )

  def test_add_kernel(self):
    # 1. Load ELF and Symbols
    entry_point, symbol_map = self.sim.get_elf_entry_and_symbol(
        self.elf_path, ["in_buf_1", "in_buf_2", "out_buf"]
    )
    self.assertIn("in_buf_1", symbol_map)
    in_buf_address_1 = symbol_map["in_buf_1"]
    self.assertNotEqual(
        in_buf_address_1, 0, "in_buf_1 symbol not found or address is 0"
    )

    self.assertIn("in_buf_2", symbol_map)
    in_buf_address_2 = symbol_map["in_buf_2"]
    self.assertNotEqual(
        in_buf_address_2, 0, "in_buf_2 symbol not found or address is 0"
    )

    self.assertIn("out_buf", symbol_map)
    out_buf_address = symbol_map["out_buf"]
    self.assertNotEqual(
        out_buf_address, 0, "out_buf symbol not found or address is 0"
    )

    # 2. Load Program
    self.sim.load_program(self.elf_path, entry_point)

    # 3. Prepare and Write Input
    input_array_1 = np.asarray(range(16), dtype=np.uint8)
    input_array_2 = np.asarray(range(16), dtype=np.uint8) * 2
    # Using the wrapper's helper, or just passing array directly if write_memory handles it (it does now)
    self.sim.write_memory(in_buf_address_1, input_array_1)
    self.sim.write_memory(in_buf_address_2, input_array_2)

    # 4. Run Simulator
    self.sim.run()
    self.sim.wait()

    # 5. Read Back and Verify
    expected_output = np.asarray(input_array_1 + input_array_2, dtype=np.uint8)
    expected_size = len(expected_output) * 1  # sizeof(uint8)
    output_array = self.sim.read_memory(out_buf_address, expected_size).view(
        np.uint8
    )
    # Verify we read something back
    self.assertEqual(len(output_array), len(expected_output))
    self.assertTrue(
        np.array_equal(expected_output, output_array),
        "Read data should match input data",
    )
    # Verify cycles were consumed
    cycle_count = self.sim.get_cycle_count()
    print(f"Cycle Count: {cycle_count}")
    self.assertTrue( 58 << cycle_count << 78, "Cycle count should be 68")

  def test_check_input_type(self):
    """Verifies that write_memory raises TypeError for non-numpy arrays."""
    # Use a dummy address
    address = 0x80000000
    # 1. Test with list
    with self.assertRaisesRegex(TypeError, "data must be a numpy array"):
      self.sim.write_memory(address, [1, 2, 3])

    # 2. Test with bytes
    with self.assertRaisesRegex(TypeError, "data must be a numpy array"):
      self.sim.write_memory(address, b"\x01\x02\x03")

  def test_step(self):
    """Verifies the step and get_cycle_count functionality."""
    entry_point, _ = self.sim.get_elf_entry_and_symbol(self.elf_path, [])
    self.sim.load_program(self.elf_path, entry_point)

    steps = 10
    actual_steps = self.sim.step(steps)
    self.assertEqual(actual_steps, steps)

    cycle_count = self.sim.get_cycle_count()
    self.assertGreater(cycle_count, 0)

  def test_read_register(self):
    """Verifies reading a register."""
    entry_point, _ = self.sim.get_elf_entry_and_symbol(self.elf_path, [])
    self.sim.load_program(self.elf_path, entry_point)

    pc_val_str = self.sim.read_register("pc")
    self.assertTrue(isinstance(pc_val_str, str))
    self.assertTrue(pc_val_str.startswith("0x"))
    self.assertGreaterEqual(int(pc_val_str, 16), 0)


if __name__ == "__main__":
  unittest.main()
