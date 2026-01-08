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

from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles


@cocotb.test()
async def core_mini_rvv_mobilenet_v1(dut):
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()
    
    # 进度显示协程
    async def show_progress(dut):
        clock_signal = dut.io_aclk 
        print(f"Starting progress monitor on signal: {clock_signal._name}...", flush=True)
        step_cycles = 1_000_000 
        count = 0
        while True:
            await cocotb.triggers.ClockCycles(clock_signal, step_cycles)
            count += 1
            print(f"Simulated {count * step_cycles} cycles...", flush=True)

    elf_files = ['run_mobilenet_v1_025_128_quant_binary.elf']
    
    for elf_file in elf_files:
        # === 关键：加载符号表，增加了 cycles 和 output 变量 ===
        await fixture.load_elf_and_lookup_symbols(
            r.Rlocation('coralnpu_hw/tests/cocotb/tutorial/tfmicro/' + elf_file),
            [
                'inference_status', 
                'inference_status_message', 
                'inference_cycles', # [新增]
                'output_class',     # [新增]
                'output_score'      # [新增]
            ])
        
        progress_task = cocotb.start_soon(show_progress(dut))

        print("Starting execution...", flush=True)
        
        # 运行直到 CPU 停机
        cycle_count = await fixture.run_to_halt(timeout_cycles=2_000_000_000)
        
        progress_task.kill()

        print(f"\nExecution halted at simulation cycle: {cycle_count}", flush=True)

        # === 读取结果 ===
        # 读取状态码 (int8)
        tflite_inference_status = (await fixture.read_signed_byte('inference_status'))
        # 读取状态消息 (64 bytes)
        msg_bytes = await fixture.read('inference_status_message', 64)
        # 去除空字符
        tflite_inference_message = bytes(msg_bytes).split(b'\0')[0].decode(errors='replace')

        # 断言执行成功
        assert tflite_inference_status == 0, f"Inference Failed! Status: {tflite_inference_status}, Msg: {tflite_inference_message}"

        # === 读取性能指标和推理结果 ===
        inference_cycles = (await fixture.read_word('inference_cycles')).item()
        pred_class = (await fixture.read('output_class', 4)).view(np.int32)[0]
        pred_score = (await fixture.read_signed_byte('output_score'))

        print("-" * 40, flush=True)
        print(f"MobileNet V1 Inference SUCCESS", flush=True)
        print(f"Message:     {tflite_inference_message}", flush=True)
        print(f"Time:        {inference_cycles:,} cycles (Firmware measured)", flush=True)
        print(f"Result:      Class {pred_class} (Score: {pred_score})", flush=True)
        print("-" * 40, flush=True)