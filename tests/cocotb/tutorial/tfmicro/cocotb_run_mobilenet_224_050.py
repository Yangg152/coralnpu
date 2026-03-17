# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

import cocotb
import sys
import struct
from cocotb.triggers import ClockCycles
from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles

# [DebugStage 映射]
STAGE_MAP = {
    0: "INIT",
    1: "MAGIC_CHECK",
    2: "STRUCTURE_CHECK",
    10: "PRE_CHECK",        
    3: "OPS_REGISTERED",
    4: "ALLOCATE_START",
    5: "ALLOCATE_DONE",
    6: "INVOKE_START",
    8: "INVOKE_DONE",
    99: "ERROR"
}

@cocotb.test()
async def core_mini_rvv_mobilenet_224_050(dut):
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()
    
    # 辅助读取函数
    async def read_u32(symbol):
        raw = await fixture.read(symbol, 4)
        return int.from_bytes(bytes(raw), byteorder='little')

    async def read_str(symbol, max_len=64):
        raw = await fixture.read(symbol, max_len)
        return bytes(raw).split(b'\0')[0].decode(errors='replace')

    # 监控协程
        # 监控协程
    async def monitor_profiler():
        print(f"\n{'='*90}", flush=True)
        print(f"DEBUG MONITOR ACTIVATED (Performance Tracing Mode)", flush=True)
        print(f"{'='*90}", flush=True)
        
        last_log_str = "" 
        last_status_msg = ""
        last_stage = -1
        stage_stuck_counter = 0 
        
        # 你的采样周期
        SAMPLE_PERIOD = 20000 
        # 新增：总累计周期计数器
        total_monitored_cycles = 0 

        while True:
            await ClockCycles(dut.io_aclk, SAMPLE_PERIOD) 
            
            # --- ✅ 新增代码开始: 每 500万 周期打印一次 ---
            total_monitored_cycles += SAMPLE_PERIOD
            if total_monitored_cycles % 5_000_000 == 0:
                # 先清除当前行的状态栏，防止文字重叠
                sys.stdout.write(f"\r{' '*90}\r")
                # 打印永久日志（会自动换行，保留在屏幕上）
                print(f"[HEARTBEAT] Simulation Running... Total Cycles: {total_monitored_cycles/1_000_000:.1f} M")
            # --- ✅ 新增代码结束 ---

            try:
                stage_code = await read_u32('debug_stage')
                if stage_code > 0x7FFFFFFF: stage_code -= 0x100000000
                
                op_idx = await read_u32('current_op_index')
                if op_idx > 0x7FFFFFFF: op_idx -= 0x100000000

                status_msg = await read_str('inference_status_message')
                current_op_name = await read_str('current_running_op', 32)
                
                if stage_code == last_stage:
                    stage_stuck_counter += 1
                else:
                    stage_stuck_counter = 0
                    last_stage = stage_code

                # 内存分配提示
                if stage_code == 4: 
                    if stage_stuck_counter % 200 == 0 and stage_stuck_counter > 0:
                         elapsed_cycles = stage_stuck_counter * SAMPLE_PERIOD
                         sys.stdout.write(f"\r{' '*90}\r") 
                         print(f"[INFO] Memory Allocating... {elapsed_cycles/1000000:.1f}M cycles elapsed")

                # 预检提示
                elif stage_code == 10: 
                    if status_msg != last_status_msg:
                        sys.stdout.write(f"\r{' '*90}\r") 
                        print(f"[PRE-CHECK] {status_msg}")
                        last_status_msg = status_msg

                # 状态更新
                elif status_msg != last_status_msg and len(status_msg) > 0:
                     sys.stdout.write(f"\r{' '*90}\r")
                     print(f"[STATUS] {STAGE_MAP.get(stage_code, 'UNKNOWN')}: {status_msg}")
                     last_status_msg = status_msg

                # 底部状态栏 - [修复] 逻辑：只在正确阶段显示 OpIdx
                stage_str = STAGE_MAP.get(stage_code, f"CODE_{stage_code}")
                spinner = "|/-\\"[(stage_stuck_counter // 5) % 4]
                
                op_display = " " * 20 # 默认留空
                
                if stage_code == 6: # INVOKE_START
                    op_display = f"Op:{op_idx} [{current_op_name}]"
                elif stage_code == 10: # PRE_CHECK
                    op_display = f"ChkOp:{op_idx}"
                
                sys.stdout.write(f"\r {spinner} [{stage_str}] {op_display} | Msg: {status_msg[:30]}")
                sys.stdout.flush()

            except Exception as e:
                pass

    elf_file = 'run_mobilenet_v1_050_224_quant_binary.elf' 
    elf_path = r.Rlocation('coralnpu_hw/tests/cocotb/tutorial/tfmicro/' + elf_file)
    
    await fixture.load_elf_and_lookup_symbols(
        elf_path,
        [
            'inference_status', 'inference_status_message', 
            'debug_stage', 'current_op_index', 'current_running_op',
            'inference_cycles',
            'op_logs', 'op_log_count', 'debug_log_buffer'
        ])
    
    monitor_task = cocotb.start_soon(monitor_profiler())

    # 运行到结束
    halt_cycle_count = await fixture.run_to_halt(timeout_cycles=500_000_000)
    monitor_task.kill()
    print(f"\n\nExecution halted at: {halt_cycle_count} cycles", flush=True)

    # 结果检查
    raw_status = await fixture.read('inference_status', 1)
    status = int.from_bytes(bytes(raw_status), byteorder='little', signed=True)
    msg = await read_str('inference_status_message')
    total_cycles = await read_u32('inference_cycles')

    # ==============================================================================
    # 打印算子性能表
    # ==============================================================================
    if status == 0:
        log_count = await read_u32('op_log_count')
        print("\n" + "="*60)
        print(f" PER-OPERATOR PERFORMANCE REPORT (Total: {log_count} ops)")
        print("="*60)
        print(f"{'Idx':<4} | {'Operator Name':<25} | {'Cycles':>12} | {'% Total':>7}")
        print("-" * 60)

        STRUCT_SIZE = 36 
        
        raw_logs = await fixture.read('op_logs', STRUCT_SIZE * log_count)
        
        sum_cycles = 0
        for i in range(log_count):
            offset = i * STRUCT_SIZE
            entry_bytes = raw_logs[offset : offset + STRUCT_SIZE]
            op_name_bytes, cycles = struct.unpack('<32sI', entry_bytes)
            
            op_name = op_name_bytes.split(b'\0')[0].decode(errors='replace')
            sum_cycles += cycles
            
            pct = (cycles / total_cycles * 100) if total_cycles > 0 else 0
            print(f"{i:<4} | {op_name:<25} | {cycles:>12,} | {pct:>6.1f}%")
            
        print("-" * 60)
        print(f"{'SUM':<4} | {'All Operators':<25} | {sum_cycles:>12,} |")
        print(f"{'TOT':<4} | {'Total Inference':<25} | {total_cycles:>12,} |")
        print("=" * 60)
    
    print(f"FINAL STATUS: {status} ({msg})")
    
    if status != 0:
        assert False, f"Test Failed with status {status}: {msg}"
