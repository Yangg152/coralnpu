# Copyright 2025 Google LLC
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

import cocotb
import numpy as np

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.core_mini_axi_interface import CoreMiniAxiInterface

@cocotb.test()
async def core_mini_axi_tutorial(dut):
    """Testbench to run your CoralNPU program."""
    # Test bench setup
    core_mini_axi = CoreMiniAxiInterface(dut)
    await core_mini_axi.init()
    await core_mini_axi.reset()
    cocotb.start_soon(core_mini_axi.clock.start())

    r = runfiles.Create()
    elf_path = r.Rlocation(
        "coralnpu_hw/tests/cocotb/tutorial/coralnpu_v2_program.elf")
    with open(elf_path, "rb") as f:
      entry_point = await core_mini_axi.load_elf(f)
      inputs1_addr = core_mini_axi.lookup_symbol(f, "input1_buffer")
      inputs2_addr = core_mini_axi.lookup_symbol(f, "input2_buffer")
      outputs_addr = core_mini_axi.lookup_symbol(f, "output_buffer")

    input1_data = np.arange(8, dtype=np.uint32)
    input2_data = 8994 * np.ones(8, dtype=np.uint32)
    await core_mini_axi.write(inputs1_addr, input1_data)
    await core_mini_axi.write(inputs2_addr, input2_data)
    await core_mini_axi.execute_from(entry_point)
    await core_mini_axi.wait_for_halted()

    rdata = (await core_mini_axi.read(outputs_addr, 4 * 8)).view(np.uint32)
    print(f"I got {rdata}")