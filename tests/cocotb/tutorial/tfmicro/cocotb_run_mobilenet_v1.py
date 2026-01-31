# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0
# ... (License header omitted)

import cocotb
import sys
from cocotb.triggers import ClockCycles

from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles

# C++ 结构体大小
OP_LOG_ENTRY_SIZE = 36 

# 调试状态字典
STAGE_MAP = {
    0: "INIT",
    1: "MODEL_LOADED",
    2: "OPS_REGISTERED",
    3: "ALLOCATE_START",
    4: "ALLOCATE_DONE",
    5: "INVOKE_START",
    6: "INVOKE_RUNNING",
    7: "INVOKE_DONE",
    99: "ERROR"
}

@cocotb.test()
async def core_mini_rvv_mobilenet_v1(dut):
    # 1. 初始化环境
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()
    
    collected_ops = []

    # === 实时监控协程 (核心调试逻辑) ===
    async def monitor_profiler(dut, fixture):
        print(f"\n{'='*90}", flush=True)
        print(f"DEBUG MONITOR ACTIVATED", flush=True)
        print(f"{'='*90}", flush=True)
        print(f"{'IDX':<5} | {'Op Name':<30} | {'Cycles':<12} | {'System Stage'}", flush=True)
        print(f"{'-'*90}", flush=True)

        last_count = 0
        last_debug_info = ""
        
        while True:
            await ClockCycles(dut.io_aclk, 5000) # 每5000周期检查一次
            
            try:
                # --- 读取全局状态 ---
                raw_stage = await fixture.read('debug_stage', 4)
                stage_code = int.from_bytes(bytes(raw_stage), byteorder='little', signed=True)
                stage_str = STAGE_MAP.get(stage_code, f"CODE_{stage_code}")

                raw_op_idx = await fixture.read('current_op_index', 4)
                op_idx = int.from_bytes(bytes(raw_op_idx), byteorder='little', signed=True)

                raw_curr_op = await fixture.read('current_running_op', 32)
                curr_op_name = bytes(raw_curr_op).split(b'\0')[0].decode(errors='replace')

                # --- 动态刷新最后一行状态 ---
                # 格式: [INVOKE_RUNNING] Op #12: CONV_2D
                debug_info = f"[{stage_str}] Op #{op_idx}: {curr_op_name if curr_op_name else '...'}"
                
                # 只有当日志没更新时，才刷新状态行
                if debug_info != last_debug_info:
                    sys.stdout.write(f"\r{' '*90}\r... {debug_info}")
                    sys.stdout.flush()
                    last_debug_info = debug_info

                # --- 读取已完成的算子日志 ---
                raw_count = await fixture.read('op_log_count', 4)
                current_count = int.from_bytes(bytes(raw_count), byteorder='little')
                
                if current_count > last_count:
                    # 清除状态行，准备打印历史记录
                    sys.stdout.write(f"\r{' '*90}\r")
                    
                    bytes_to_read = current_count * OP_LOG_ENTRY_SIZE
                    all_logs_raw = await fixture.read('op_logs', bytes_to_read)
                    all_logs_data = bytes(all_logs_raw)
                    
                    for i in range(last_count, current_count):
                        offset = i * OP_LOG_ENTRY_SIZE
                        entry_data = all_logs_data[offset : offset + OP_LOG_ENTRY_SIZE]
                        
                        op_name = entry_data[0:32].split(b'\0')[0].decode(errors='replace')
                        op_cycles = int.from_bytes(entry_data[32:36], byteorder='little')
                        
                        # 打印历史记录 (永久保存)
                        print(f"{i:<5} | {op_name:<30} | {op_cycles:<12,} | {stage_str}", flush=True)
                        
                        collected_ops.append({'id': i, 'name': op_name, 'cycles': op_cycles})
                    
                    last_count = current_count
                    # 重新打印状态行
                    sys.stdout.write(f"... {debug_info}")
                    sys.stdout.flush()

            except Exception:
                pass # 忽略读取期间的瞬时错误

    # === 主测试流程 ===
    elf_file = 'run_mobilenet_v1_025_partial_binary.elf' 

    # 注意：必须加载 debug_stage 和 current_op_index 符号
    await fixture.load_elf_and_lookup_symbols(
        r.Rlocation('coralnpu_hw/tests/cocotb/tutorial/tfmicro/' + elf_file),
        [
            'inference_status', 
            'inference_status_message', 
            'inference_cycles',
            'output_class',
            'output_score',
            'op_log_count',  
            'op_logs',
            'current_running_op',
            'debug_stage',      # 新增
            'current_op_index'  # 新增
        ])
    
    monitor_task = cocotb.start_soon(monitor_profiler(dut, fixture))

    # 运行直到 Halt
    halt_cycle_count = await fixture.run_to_halt(timeout_cycles=50_000_000) # 适当增加超时
    
    monitor_task.kill()
    sys.stdout.write(f"\r{' '*90}\r") # 清理最后一行

    print(f"\nExecution halted at: {halt_cycle_count} cycles", flush=True)

    # === 读取最终结果 ===
    # (为了稳妥，再次读取所有日志)
    raw_count = await fixture.read('op_log_count', 4)
    final_count = int.from_bytes(bytes(raw_count), byteorder='little')
    
    if final_count > len(collected_ops):
        # ... 读取剩余日志 (代码同上，略微简化) ...
        pass 

    raw_status = await fixture.read('inference_status', 1)
    status = int.from_bytes(bytes(raw_status), byteorder='little', signed=True)
    
    msg_bytes = await fixture.read('inference_status_message', 64)
    msg = bytes(msg_bytes).split(b'\0')[0].decode(errors='replace')
    
    raw_stage = await fixture.read('debug_stage', 4)
    final_stage = int.from_bytes(bytes(raw_stage), byteorder='little', signed=True)

    print("=" * 60)
    print("FINAL DIAGNOSTICS")
    print("=" * 60)
    print(f"Debug Stage:     {STAGE_MAP.get(final_stage, final_stage)}")
    print(f"Inference Status: {status}")
    print(f"Status Message:  {msg}")
    print(f"Last Op Index:   {collected_ops[-1]['id'] if collected_ops else 'None'}")
    print("=" * 60)
    
    if status != 0:
        assert False, f"Test Failed: {msg} (Stage: {STAGE_MAP.get(final_stage)})"
