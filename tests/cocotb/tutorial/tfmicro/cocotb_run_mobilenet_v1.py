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
from cocotb.triggers import ClockCycles

from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles

# C++ 结构体 OpLogEntry { char name[32]; uint32_t cycles; } 的大小
# 32 bytes (name) + 4 bytes (cycles) = 36 bytes
OP_LOG_ENTRY_SIZE = 36 

@cocotb.test()
async def core_mini_rvv_mobilenet_v1(dut):
    # 1. 初始化仿真环境
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()
    
    # 用于存储收集到的算子性能数据，供最后汇总使用
    collected_ops = []

    # === 实时监控协程 (修复版) ===
    async def monitor_profiler(dut, fixture):
        """
        定期检查内存，一旦有新的算子完成，立即输出。
        """
        print(f"\n{'='*60}", flush=True)
        print(f"Streaming Profiler Output (Live)", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"{'ID':<5} | {'Op Name':<25} | {'Cycles':<15}", flush=True)
        print(f"{'-'*60}", flush=True)

        last_count = 0
        
        while True:
            # 每隔 5000 个周期检查一次内存
            await ClockCycles(dut.io_aclk, 5000)

            try:
                raw_status = await fixture.read('inference_status', 1)
                status = int.from_bytes(bytes(raw_status), byteorder='little', signed=True)
                
                if status == 0:
                    pass 
                    print(f"\n[Monitor] Detected inference_status == 0 (Success). Stopping simulation...", flush=True)
                    return # 退出监控，配合主线程逻辑修改
            except Exception:
                pass #     
            
            try:
                # 读取 op_log_count
                raw_count = await fixture.read('op_log_count', 4)
                # [修复] 强制转为 bytes，防止 numpy array 报错
                current_count = int.from_bytes(bytes(raw_count), byteorder='little')
                
                # 如果计数增加，说明有新算子跑完了
                if current_count > last_count:
                    bytes_to_read = current_count * OP_LOG_ENTRY_SIZE
                    
                    # 读取所有日志数据
                    all_logs_data_raw = await fixture.read('op_logs', bytes_to_read)
                    # [修复] 关键步骤：将 numpy array 转为 bytes
                    all_logs_data = bytes(all_logs_data_raw)
                    
                    # 只处理新增的部分
                    for i in range(last_count, current_count):
                        offset = i * OP_LOG_ENTRY_SIZE
                        entry_data = all_logs_data[offset : offset + OP_LOG_ENTRY_SIZE]
                        
                        # 解析 Name (前32字节)
                        raw_name = entry_data[0:32]
                        # split 是 bytes 的方法，这里现在安全了
                        op_name = raw_name.split(b'\0')[0].decode(errors='replace')
                        
                        # 解析 Cycles (后4字节)
                        raw_cycles = entry_data[32:36]
                        op_cycles = int.from_bytes(raw_cycles, byteorder='little')
                        
                        # 1. 实时打印
                        print(f"{i:<5} | {op_name:<25} | {op_cycles:<15,}", flush=True)
                        
                        # 2. 存入列表供最后汇总
                        collected_ops.append({
                            'id': i,
                            'name': op_name,
                            'cycles': op_cycles
                        })
                    
                    last_count = current_count
            except Exception as e:
                # 打印异常详情以便调试
                print(f"Profiler Monitor Warning: {e} (Type: {type(e)})")

    # === 进度打印协程 ===
    async def show_progress(dut):
        step_cycles = 200_000 
        count = 0
        while True:
            await ClockCycles(dut.io_aclk, step_cycles)
            count += 1
            print(f"[Sim] Running... {count * step_cycles} cycles", flush=True)

    # === 主流程 ===
    elf_files = ['run_mobilenet_v1_025_partial_binary.elf'] 
    # elf_files = ['run_mobilenet_v1_025_128_quant_binary.elf'] 
    
    for elf_file in elf_files:
        # 加载 ELF 并查找符号地址
        await fixture.load_elf_and_lookup_symbols(
            r.Rlocation('coralnpu_hw/tests/cocotb/tutorial/tfmicro/' + elf_file),
            [
                'inference_status', 
                'inference_status_message', 
                'inference_cycles',
                'output_class',
                'output_score',
                'op_log_count',  
                'op_logs'     ,
                'debug_log_buffer'  
            ])
        
        print("Starting execution...", flush=True)

        # 启动后台任务
        monitor_task = cocotb.start_soon(monitor_profiler(dut, fixture))
        progress_task = cocotb.start_soon(show_progress(dut))

        # 运行直到 CPU 停机
        halt_cycle_count = await fixture.run_to_halt(timeout_cycles=2_000_000_000)
        
        # 杀死后台任务
        monitor_task.kill()
        progress_task.kill()

        print(f"\nExecution halted at simulation cycle: {halt_cycle_count}", flush=True)
        print(f"{'-'*60}\n", flush=True)

        # ---------------------------------------------------------
        # 仿真结束，读取最终结果
        # ---------------------------------------------------------
        
        # 1. 再次检查是否有遗漏的日志
        raw_count = await fixture.read('op_log_count', 4)
        final_log_count = int.from_bytes(bytes(raw_count), byteorder='little') # [修复] 转 bytes
        
        if final_log_count > len(collected_ops):
            bytes_to_read = final_log_count * OP_LOG_ENTRY_SIZE
            all_logs_data_raw = await fixture.read('op_logs', bytes_to_read)
            all_logs_data = bytes(all_logs_data_raw) # [修复] 转 bytes
            
            for i in range(len(collected_ops), final_log_count):
                offset = i * OP_LOG_ENTRY_SIZE
                entry_data = all_logs_data[offset : offset + OP_LOG_ENTRY_SIZE]
                
                op_name = entry_data[0:32].split(b'\0')[0].decode(errors='replace')
                op_cycles = int.from_bytes(entry_data[32:36], byteorder='little')
                
                collected_ops.append({'id': i, 'name': op_name, 'cycles': op_cycles})

        # 2. 读取标准状态位 (增加 bytes 转换以防万一)
        raw_status = await fixture.read('inference_status', 1)
        tflite_inference_status = int.from_bytes(bytes(raw_status), byteorder='little', signed=True)

        raw_debug_log = await fixture.read('debug_log_buffer', 4096)
        debug_log_str = bytes(raw_debug_log).split(b'\0')[0].decode(errors='replace')

        msg_bytes = await fixture.read('inference_status_message', 64)
        tflite_inference_message = bytes(msg_bytes).split(b'\0')[0].decode(errors='replace')

        raw_cycles = await fixture.read('inference_cycles', 4)
        total_inference_cycles = int.from_bytes(bytes(raw_cycles), byteorder='little', signed=False)

        raw_class = await fixture.read('output_class', 4)
        pred_class = int.from_bytes(bytes(raw_class), byteorder='little', signed=True)
        
        raw_score = await fixture.read('output_score', 1)
        pred_score = int.from_bytes(bytes(raw_score), byteorder='little', signed=True)

        # ---------------------------------------------------------
        # 打印最终总报告
        # ---------------------------------------------------------

        print("=" * 60, flush=True)
        print(f" Status:        {tflite_inference_status} ({'SUCCESS' if tflite_inference_status == 0 else 'FAILED'})", flush=True)
        # 打印详细日志
        if len(debug_log_str) > 0:
            print(f" Debug Log:\n{'-'*20}\n{debug_log_str}\n{'-'*20}", flush=True)
        else:
            print(f" Debug Log:     (Empty)", flush=True)

        print("=" * 60, flush=True)
        print(f"FINAL PERFORMANCE REPORT: MobileNet V1", flush=True)
        print("=" * 60, flush=True)
        
        print(f"{'ID':<5} | {'Op Name':<25} | {'Cycles':<15} | {'% Total':<10}", flush=True)
        print("-" * 60, flush=True)
        
        sum_op_cycles = 0
        for op in collected_ops:
            cycles = op['cycles']
            sum_op_cycles += cycles
            pct = (cycles / total_inference_cycles * 100) if total_inference_cycles > 0 else 0.0
            print(f"{op['id']:<5} | {op['name']:<25} | {cycles:<15,} | {pct:.1f}%", flush=True)
            
        print("-" * 60, flush=True)
        print(f"{'SUM':<5} | {'All Ops':<25} | {sum_op_cycles:<15,}", flush=True)
        
        overhead = total_inference_cycles - sum_op_cycles
        print(f"{'OVHD':<5} | {'Framework/Init':<25} | {overhead:<15,}", flush=True)
        print("=" * 60, flush=True)
        
        print(f" Status:        {tflite_inference_status} ({'SUCCESS' if tflite_inference_status == 0 else 'FAILED'})", flush=True)
        print(f" Status Msg:    {tflite_inference_message}", flush=True)
        print(f" Total Time:    {total_inference_cycles:,} cycles", flush=True)
        
        if tflite_inference_status == 0:
            print(f" Pred Class:    {pred_class}", flush=True)
            print(f" Confidence:    {pred_score} (int8)", flush=True)
        
        print("=" * 60, flush=True)

        assert tflite_inference_status == 0, f"Inference Failed: {tflite_inference_message}"