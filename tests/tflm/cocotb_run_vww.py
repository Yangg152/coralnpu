# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

import cocotb
import sys
import struct
from cocotb.triggers import ClockCycles
from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles

STAGE_MAP = {
    0:  "INIT",
    1:  "MAGIC_CHECK",
    2:  "STRUCTURE_CHECK",
    3:  "OPS_REGISTERED",
    4:  "ALLOCATE_START",
    5:  "ALLOCATE_DONE",
    6:  "INVOKE_START",
    8:  "INVOKE_DONE",
    99: "ERROR",
}

CLASS_NAMES = {0: "no_person", 1: "person"}


@cocotb.test()
async def core_mini_rvv_vww_full(dut):
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()

    async def read_u32(symbol):
        raw = await fixture.read(symbol, 4)
        return int.from_bytes(bytes(raw), byteorder='little')

    async def read_i32(symbol):
        val = await read_u32(symbol)
        return val if val <= 0x7FFFFFFF else val - 0x100000000

    async def read_str(symbol, max_len=64):
        raw = await fixture.read(symbol, max_len)
        return bytes(raw).split(b'\0')[0].decode(errors='replace')

    async def monitor_profiler():
        print(f"\n{'='*90}", flush=True)
        print(f"DEBUG MONITOR ACTIVATED  —  VWW Inference", flush=True)
        print(f"{'='*90}", flush=True)

        last_status_msg  = ""
        last_stage       = -1
        stage_stuck_counter = 0

        SAMPLE_PERIOD          = 20_000
        total_monitored_cycles = 0

        while True:
            await ClockCycles(dut.io_aclk, SAMPLE_PERIOD)
            total_monitored_cycles += SAMPLE_PERIOD

            if total_monitored_cycles % 5_000_000 == 0:
                sys.stdout.write(f"\r{' '*90}\r")
                print(f"[HEARTBEAT] Simulation Running... "
                      f"Total Cycles: {total_monitored_cycles/1_000_000:.1f} M",
                      flush=True)

            try:
                stage_code = await read_u32('debug_stage')
                if stage_code > 0x7FFFFFFF:
                    stage_code -= 0x100000000

                op_idx = await read_u32('current_op_index')
                if op_idx > 0x7FFFFFFF:
                    op_idx -= 0x100000000

                status_msg      = await read_str('inference_status_message')
                current_op_name = await read_str('current_running_op', 32)

                if stage_code == last_stage:
                    stage_stuck_counter += 1
                else:
                    stage_stuck_counter = 0
                    last_stage = stage_code

                # 内存分配提示
                if stage_code == 4:
                    if stage_stuck_counter % 200 == 0 and stage_stuck_counter > 0:
                        elapsed = stage_stuck_counter * SAMPLE_PERIOD
                        sys.stdout.write(f"\r{' '*90}\r")
                        print(f"[INFO] Memory Allocating... "
                              f"{elapsed/1_000_000:.1f}M cycles elapsed")

                elif status_msg != last_status_msg and len(status_msg) > 0:
                    sys.stdout.write(f"\r{' '*90}\r")
                    stage_str = STAGE_MAP.get(stage_code, f'CODE_{stage_code}')
                    print(f"[STATUS] {stage_str}: {status_msg}")
                    last_status_msg = status_msg

                # 底部状态栏
                stage_str  = STAGE_MAP.get(stage_code, f"CODE_{stage_code}")
                spinner    = "|/-\\"[(stage_stuck_counter // 5) % 4]
                op_display = " " * 20

                if stage_code == 6:
                    op_display = f"Op:{op_idx} [{current_op_name}]"

                sys.stdout.write(
                    f"\r {spinner} [{stage_str}] {op_display} | "
                    f"Msg: {status_msg[:30]}"
                )
                sys.stdout.flush()

            except Exception:
                pass

    elf_file = 'run_vww_binary.elf'
    elf_path = r.Rlocation('coralnpu_hw/tests/tflm/' + elf_file)

    await fixture.load_elf_and_lookup_symbols(
        elf_path,
        [
            'inference_status',
            'inference_status_message',
            'debug_stage',
            'current_op_index',
            'current_running_op',
            'inference_cycles',
            'output_class',
            'output_score',
            'op_logs',
            'op_log_count',
            'debug_log_buffer',
        ])

    monitor_task = cocotb.start_soon(monitor_profiler())

    halt_cycle_count = await fixture.run_to_halt(timeout_cycles=500_000_000)
    monitor_task.kill()
    print(f"\n\nExecution halted at: {halt_cycle_count} cycles", flush=True)

    # ── 读取结果 ──────────────────────────────────────────────
    raw_status = await fixture.read('inference_status', 1)
    status     = int.from_bytes(bytes(raw_status), byteorder='little', signed=True)
    msg        = await read_str('inference_status_message')
    total_cycles = await read_u32('inference_cycles')

    out_class = await read_i32('output_class')
    raw_score = await fixture.read('output_score', 1)
    out_score = int.from_bytes(bytes(raw_score), byteorder='little', signed=True)

    # ── 算子性能表 ────────────────────────────────────────────
    if status == 0:
        log_count = await read_u32('op_log_count')
        print("\n" + "="*65)
        print(f" VWW PER-OPERATOR PERFORMANCE REPORT  (Total: {log_count} ops)")
        print("="*65)
        print(f"{'Idx':<4} | {'Operator Name':<25} | {'Cycles':>12} | {'% Total':>7}")
        print("-"*65)

        STRUCT_SIZE = 36
        raw_logs = await fixture.read('op_logs', STRUCT_SIZE * log_count)

        sum_cycles = 0
        for i in range(log_count):
            offset     = i * STRUCT_SIZE
            entry_bytes = raw_logs[offset: offset + STRUCT_SIZE]
            op_name_bytes, cycles = struct.unpack('<32sI', entry_bytes)
            op_name    = op_name_bytes.split(b'\0')[0].decode(errors='replace')
            sum_cycles += cycles
            pct        = (cycles / total_cycles * 100) if total_cycles > 0 else 0
            print(f"{i:<4} | {op_name:<25} | {cycles:>12,} | {pct:>6.1f}%")

        print("-"*65)
        print(f"{'SUM':<4} | {'All Operators':<25} | {sum_cycles:>12,} |")
        print(f"{'TOT':<4} | {'Total Inference':<25} | {total_cycles:>12,} |")
        print("="*65)

        class_name = CLASS_NAMES.get(out_class, f"unknown({out_class})")
        print(f"\n Prediction : {class_name}  (raw score = {out_score})")

    print(f"\nFINAL STATUS: {status} ({msg})")

    if status != 0:
        debug_log = await read_str('debug_log_buffer', max_len=2048)
        print(f"\n{'='*70}")
        print("TFLM INTERNAL ERROR LOG (critical for diagnosis):")
        print(debug_log)
        print('='*70)
        assert False, f"Test Failed with status {status}: {msg}"