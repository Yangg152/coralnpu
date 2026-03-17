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
from coralnpu_test_utils.sim_test_fixture import Fixture


# ==============================================================================
# Hardcoded Quantization Parameters (from depthwise_conv_test.cc)
# output_multiplier and output_shift arrays repeat every 32 entries.
# ==============================================================================
_OUTPUT_MULTIPLIER_BASE = [
    1215836872, 1185328645, 2021922132, 1293877495, 1998015473, 1311410766,
    1264495606, 1275161392, 1117173051, 1337281014, 1333632221, 1344518575,
    2123709422, 1204213014, 2004037101, 1403589537, 1353512509, 1184963313,
    1213021689, 1414462760, 1222505637, 1256031519, 1133123072, 1150781908,
    1255231012, 1152753331, 1238298425, 1294933879, 1111570159, 1309309916,
    1251381292, 1218702370,
]

_OUTPUT_SHIFT_BASE = [
    -7, -7, -8, -7, -8, -7, -7, -7, -7, -7, -7, -7, -8, -7, -8, -7,
    -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7,
]

# Input/output quantization constants (from depthwise_conv_test.cc)
_INPUT_OFFSET  =  128
_OUTPUT_OFFSET = -128
_ACT_MIN       = -128
_ACT_MAX       =  127


# ==============================================================================
# TFLite Quantization Math Helpers
# ==============================================================================

def _sat_rounding_doubling_high_mul(a_i32: np.ndarray, b_scalar: int) -> np.ndarray:
    """Replicates TFLite SaturatingRoundingDoublingHighMul (int32 x int32 -> int32)."""
    a64 = a_i32.astype(np.int64)
    b64 = np.int64(b_scalar)
    nudge = np.int64(1 << 30)
    return ((a64 * b64 + nudge) >> 31).astype(np.int64)


def _rounding_right_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """Arithmetic rounding right shift (shift >= 0)."""
    if shift == 0:
        return x
    threshold = np.int64(1) << (shift - 1)
    return (x + threshold) >> shift


def _requantize(acc: np.ndarray, oc: int) -> np.ndarray:
    """Apply per-channel requantization matching TFLite reference kernel."""
    mult  = _OUTPUT_MULTIPLIER_BASE[oc % 32]
    shift = _OUTPUT_SHIFT_BASE[oc % 32]

    acc64 = _sat_rounding_doubling_high_mul(acc, mult)

    if shift < 0:
        acc64 = _rounding_right_shift(acc64, -shift)
    else:
        acc64 = acc64 * (np.int64(1) << shift)

    acc64 += _OUTPUT_OFFSET
    return np.clip(acc64, _ACT_MIN, _ACT_MAX).astype(np.int8)


# ==============================================================================
# Python Reference Implementation for Depthwise Conv
# Matches TFLite reference_integer_ops::DepthwiseConvPerChannel exactly.
# padding = 1 (fixed, from C file)
# ==============================================================================

def py_depthwise_conv2d(input_data: np.ndarray,
                         filter_data: np.ndarray,
                         bias_data: np.ndarray,
                         in_shape, f_shape, out_shape,
                         stride: int, dm: int) -> np.ndarray:
    """
    Pure-Python / NumPy depthwise conv matching the C golden reference.

    Shapes (NHWC / TFLite convention):
      in_shape  : [1, in_h,  in_w,  in_d]
      f_shape   : [1, 3,     3,     out_d]   out_d = in_d * dm
      out_shape : [1, out_h, out_w, out_d]
    padding = 1 (hardcoded in C file)
    """
    _, in_h,  in_w,  in_d  = in_shape
    _, f_h,   f_w,   out_d  = f_shape
    _, out_h, out_w, _      = out_shape

    inp  = input_data.reshape(in_shape).astype(np.int32)   # [1,H,W,C]
    flt  = filter_data.reshape(f_shape).astype(np.int32)   # [1,3,3,OC]
    bias = bias_data.astype(np.int32)                       # [OC]

    # Padding=1: pad value must be -input_offset so that
    # (pad_val + input_offset) == 0, contributing nothing to the sum.
    pad_val = -_INPUT_OFFSET
    inp_padded = np.pad(
        inp,
        ((0, 0), (1, 1), (1, 1), (0, 0)),
        mode='constant',
        constant_values=pad_val,
    )  # [1, in_h+2, in_w+2, in_d]

    out = np.zeros((1, out_h, out_w, out_d), dtype=np.int8)

    for oh in range(out_h):
        for ow in range(out_w):
            patch = inp_padded[0,
                               oh * stride : oh * stride + f_h,
                               ow * stride : ow * stride + f_w,
                               :]  # [f_h, f_w, in_d]

            for ic in range(in_d):
                for m in range(dm):
                    oc = ic * dm + m

                    # Accumulate: (input + input_offset) * weight,  int32
                    acc_val = np.int32(0)
                    for fh in range(f_h):
                        for fw in range(f_w):
                            acc_val += (
                                (np.int32(patch[fh, fw, ic]) + _INPUT_OFFSET)
                                * np.int32(flt[0, fh, fw, oc])
                            )

                    acc_val += bias[oc]

                    # Per-channel requantization
                    out[0, oh, ow, oc] = _requantize(
                        np.array([acc_val], dtype=np.int32), oc
                    )[0]

    return out.flatten()


# ==============================================================================
# Hardcoded Reference Cycle Counts (from earlier scalar-kernel runs)
# ==============================================================================
HARDCODED_REF_CYCLES = {
    # (in_d, dm, stride, out_h, out_w) -> ref_cycles
    (8,   1, 1, 4, 4):   65_462,
    (8,   1, 2, 4, 4):   70_873,
    (32,  1, 1, 4, 4):  259_016,
    (32,  1, 2, 4, 4):  280_495,
    (64,  1, 1, 4, 4):  517_496,
    (64,  1, 2, 4, 4):  560_331,
    (16,  2, 2, 4, 4):  273_558,
}


# ==============================================================================
# Global Storage for Final Summary
# ==============================================================================
TEST_RESULTS = []


def print_global_summary(log):
    header = (
        f"\n{'='*120}\n"
        f"{'In Shape':<22} | {'Filter':<12} | {'DM':>4} | {'Stride':>6} | "
        f"{'Ref Cycles':>12} | {'Opt Cycles':>12} | {'Speedup':>8} | {'Status':>8}\n"
        f"{'-'*120}"
    )
    rows = []
    for res in TEST_RESULTS:
        opt_cyc_str = f"{res['opt']:12d}"       if res['opt']     is not None else f"{'N/A':>12}"
        speedup_str = f"{res['speedup']:7.2f}x"  if res['speedup'] is not None else f"{'N/A':>8}"
        rows.append(
            f"{res['in_shape']:<22} | {res['filter']:<12} | {res['dm']:>4} | {res['stride']:>6} | "
            f"{res['ref']:12d} | {opt_cyc_str} | {speedup_str} | {res['status']:>8}"
        )
    table = "\n".join([header] + rows + [f"{'='*120}\n"])
    log.info(table)


# ==============================================================================
# Helper
# ==============================================================================

def tolerate(target: int, tolerance: float = 1.2) -> int:
    return int(target * tolerance)


class DepthwiseConvTest:
    """
    Test harness for depthwise conv.

    Correctness: Python reference (py_depthwise_conv2d) vs run_optimized output.
    Speedup: hardcoded ref cycles / measured opt cycles.
    """

    def __init__(self, dut, in_d, dm=1, stride=1, out_h=4, out_w=4):
        self.dut    = dut
        self.dm     = dm
        self.stride = stride

        out_d  = in_d * dm
        in_h   = out_h * stride
        in_w   = out_w * stride

        self.in_shape   = np.array([1, in_h, in_w, in_d], dtype=np.uint32)
        self.f_shape    = np.array([1, 3, 3, out_d],       dtype=np.uint32)
        self.bias_shape = np.array([out_d],                dtype=np.uint32)
        self.out_shape  = np.array([1, out_h, out_w, out_d], dtype=np.uint32)
        self.out_size   = int(np.prod(self.out_shape))

        self._in_shape_str = f"[1,{in_h},{in_w},{in_d}]"
        self._filter_str   = f"[1,3,3,{out_d}]"
        self._ref_key      = (in_d, dm, stride, out_h, out_w)

        r = runfiles.Create()
        self.elf_file = r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/depthwise_conv_test.elf')
        self.fixture = None

        # Saved for Python reference
        self._input_data  = None
        self._filter_data = None
        self._bias_data   = None

    async def load_and_populate_input(self):
        self.fixture = await Fixture.Create(self.dut, highmem=True)
        await self.fixture.load_elf_and_lookup_symbols(
            self.elf_file,
            [
                'impl', 'run_ref', 'run_optimized',
                'dm', 'stride',
                'filter_shape', 'filter_data',
                'bias_shape',   'bias_data',
                'input_shape',  'input_data',
                'output_shape', 'output_data',
            ]
        )

        rng = np.random.default_rng()
        self._filter_data = rng.integers(-128, 128, self.f_shape,   dtype=np.int8).flatten()
        self._bias_data   = rng.integers(-5000, 5000, self.out_shape[3], dtype=np.int32)
        self._input_data  = rng.integers(-128, 128, self.in_shape,  dtype=np.int8).flatten()

        await self.fixture.write_word('stride', self.stride)
        await self.fixture.write_word('dm', self.dm)
        await self.fixture.write('filter_shape', self.f_shape)
        await self.fixture.write('filter_data',  self._filter_data)
        await self.fixture.write('bias_shape',   self.bias_shape)
        await self.fixture.write('bias_data',    self._bias_data)
        await self.fixture.write('input_shape',  self.in_shape)
        await self.fixture.write('input_data',   self._input_data)
        await self.fixture.write('output_shape', self.out_shape)

    async def _run_kernel(self, func_ptr: str, timeout_cycles: int):
        await self.fixture.write_ptr('impl', func_ptr)
        await self.fixture.write('output_data', np.zeros([self.out_size], dtype=np.int8))
        cycles  = await self.fixture.run_to_halt(timeout_cycles=timeout_cycles)
        outputs = (await self.fixture.read('output_data', self.out_size)).view(np.int8)
        return outputs, cycles

    async def test(self, opt_target: int):
        """
        1. Compute Python golden reference.
        2. Run run_optimized on hardware.
        3. Compare and report.
        Speedup uses hardcoded ref cycle counts (no need to run run_ref).
        """
        # --- Python golden reference ---
        py_ref = py_depthwise_conv2d(
            self._input_data, self._filter_data, self._bias_data,
            self.in_shape, self.f_shape, self.out_shape,
            self.stride, self.dm,
        )

        # --- Optimized kernel ---
        opt_output, opt_cycles = await self._run_kernel(
            'run_optimized', tolerate(opt_target))

        # --- Verify ---
        match  = (opt_output == py_ref).all()
        status = "PASS" if match else "FAIL"

        # --- Speedup ---
        ref_cycles = HARDCODED_REF_CYCLES.get(self._ref_key, 0)
        speedup    = ref_cycles / opt_cycles if opt_cycles > 0 else 0.0

        # --- Record ---
        TEST_RESULTS.append({
            "in_shape": self._in_shape_str,
            "filter":   self._filter_str,
            "dm":       self.dm,
            "stride":   self.stride,
            "ref":      ref_cycles,
            "opt":      opt_cycles,
            "speedup":  speedup,
            "status":   status,
        })

        print_global_summary(self.dut._log)

        if not match:
            mismatch_idx = np.where(opt_output != py_ref)[0]
            n = min(16, len(mismatch_idx))
            self.dut._log.warning(
                f"Output mismatch! Python ref vs opt differ at "
                f"{len(mismatch_idx)} positions.\n"
                f"  py_ref[:16]:  {py_ref[:16]}\n"
                f"  opt_output[:16]: {opt_output[:16]}\n"
                f"  First {n} mismatch indices: {mismatch_idx[:n]}\n"
                f"  py_ref  at those: {py_ref[mismatch_idx[:n]]}\n"
                f"  opt_out at those: {opt_output[mismatch_idx[:n]]}"
            )
            assert False, f"Output mismatch at {len(mismatch_idx)} positions."

    async def benchmark(self, opt_target: int):
        """Run only the optimized kernel (benchmark mode) — no correctness check."""
        _, opt_cycles = await self._run_kernel(
            'run_optimized', tolerate(opt_target))

        ref_cycles = HARDCODED_REF_CYCLES.get(self._ref_key, 0)
        speedup    = (ref_cycles / opt_cycles
                      if (opt_cycles > 0 and ref_cycles > 0) else None)

        TEST_RESULTS.append({
            "in_shape": self._in_shape_str,
            "filter":   self._filter_str,
            "dm":       self.dm,
            "stride":   self.stride,
            "ref":      ref_cycles,
            "opt":      opt_cycles,
            "speedup":  speedup,
            "status":   "BENCH",
        })
        print_global_summary(self.dut._log)


# ==============================================================================
# Tests
# Cycle count targets from `-c dbg` runs (DCHECKs enabled, slower than opt).
# ==============================================================================

@cocotb.test()
async def test_dwconv8to8stride1(dut):
    t = DepthwiseConvTest(dut, in_d=8)
    await t.load_and_populate_input()
    await t.test(opt_target=26_600)

@cocotb.test()
async def test_dwconv8to8stride2(dut):
    t = DepthwiseConvTest(dut, in_d=8, stride=2)
    await t.load_and_populate_input()
    await t.test(opt_target=26_400)

@cocotb.test()
async def test_dwconv32to32stride1(dut):
    t = DepthwiseConvTest(dut, in_d=32)
    await t.load_and_populate_input()
    await t.test(opt_target=30_600)

@cocotb.test()
async def test_dwconv32to32stride2(dut):
    t = DepthwiseConvTest(dut, in_d=32, stride=2)
    await t.load_and_populate_input()
    await t.test(opt_target=28_500)

@cocotb.test()
async def test_dwconv64to64stride1(dut):
    t = DepthwiseConvTest(dut, in_d=64)
    await t.load_and_populate_input()
    await t.test(opt_target=49_300)

@cocotb.test()
async def test_dwconv64to64stride2(dut):
    t = DepthwiseConvTest(dut, in_d=64, stride=2)
    await t.load_and_populate_input()
    await t.test(opt_target=45_700)

@cocotb.test()
async def test_dwconv16to32stride2(dut):
    t = DepthwiseConvTest(dut, in_d=16, dm=2, stride=2)
    await t.load_and_populate_input()
    await t.test(opt_target=41_600)


# ==============================================================================
# Benchmarks — skipped by default.
# Run with COCOTB_TESTCASE=<name>
# Cycle count targets from `-c opt` runs.
# ==============================================================================

@cocotb.test(skip=True)
async def benchmark_dwconv8to8(dut):
    t = DepthwiseConvTest(dut, in_d=8, out_h=112, out_w=112)
    await t.load_and_populate_input()
    await t.benchmark(opt_target=2_600_000)

@cocotb.test(skip=True)
async def benchmark_dwconv32to32(dut):
    t = DepthwiseConvTest(dut, in_d=32, out_h=56, out_w=56)
    await t.load_and_populate_input()
    await t.benchmark(opt_target=974_000)

@cocotb.test(skip=True)
async def benchmark_dwconv64to64(dut):
    t = DepthwiseConvTest(dut, in_d=64, out_h=28, out_w=28)
    await t.load_and_populate_input()
    await t.benchmark(opt_target=528_000)

@cocotb.test(skip=True)
async def benchmark_dwconv128to128(dut):
    t = DepthwiseConvTest(dut, in_d=128, out_h=14, out_w=14)
    await t.load_and_populate_input()
    await t.benchmark(opt_target=301_000)

@cocotb.test(skip=True)
async def benchmark_dwconv256to256(dut):
    t = DepthwiseConvTest(dut, in_d=256, out_h=7, out_w=7)
    await t.load_and_populate_input()
    await t.benchmark(opt_target=180_000)
