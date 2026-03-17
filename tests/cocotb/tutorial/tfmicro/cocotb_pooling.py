# Copyright 2025 Google LLC
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

# tests/cocotb/tutorial/tfmicro/cocotb_pooling.py

import cocotb
import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

# ==============================================================================
# Global Storage for Final Summary
# ==============================================================================
TEST_RESULTS = []


def print_global_summary(log):
    """Prints a cumulative table of all test results collected so far."""
    header = (
        f"\n{'='*130}\n"
        f"{'Op':<10} | {'Shape (N,H,W,C)':<22} | {'Filter':<10} | {'Stride':<8} | "
        f"{'Pad':<6} | {'Ref Cycles':>12} | {'Opt Cycles':>12} | {'Speedup':>8} | {'Status':>8}\n"
        f"{'-'*130}"
    )
    rows = []
    for r in TEST_RESULTS:
        opt_str     = f"{r['opt']:12d}" if r['opt'] is not None else f"{'-':>12}"
        speedup_str = f"{r['speedup']:8.2f}x" if r['speedup'] is not None else f"{'-':>8}"
        rows.append(
            f"{r['op']:<10} | {r['shape']:<22} | {r['filter']:<10} | {r['stride']:<8} | "
            f"{r['pad']:<6} | {r['ref']:12d} | {opt_str} | {speedup_str} | {r['status']:>8}"
        )
    table = "\n".join([header] + rows + [f"{'='*130}\n"])
    log.info(table)


def tolerate(target: int, tolerance: float = 1.5) -> int:
    return int(target * tolerance)


# ==============================================================================
# Reference helpers (pure-Python / NumPy)
# ==============================================================================

def _ref_avgpool_int8(input_nhwc: np.ndarray,
                      filter_h: int, filter_w: int,
                      stride_h: int, stride_w: int,
                      pad_h: int, pad_w: int,
                      act_min: int = -128, act_max: int = 127) -> np.ndarray:
    """
    Scalar reference for int8 AveragePool matching TFLite quantization contract:
      output = clip( round_half_up( mean( window ) ), act_min, act_max )
    where mean() is computed on raw int8 values (no zero-point shift needed
    when input_scale == output_scale, see kernel comment).
    """
    n, in_h, in_w, c = input_nhwc.shape
    out_h = (in_h + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - filter_w) // stride_w + 1

    # Pad with zeros (neutral for averaging — zero-point is 0 in our tests)
    padded = np.pad(input_nhwc.astype(np.int32),
                    ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode='constant', constant_values=0)

    output = np.empty((n, out_h, out_w, c), dtype=np.int8)
    for b in range(n):
        for oy in range(out_h):
            for ox in range(out_w):
                window = padded[b,
                                oy * stride_h: oy * stride_h + filter_h,
                                ox * stride_w: ox * stride_w + filter_w, :]
                s = window.sum(axis=(0, 1))          # shape (C,)
                count = window.shape[0] * window.shape[1]
                # round-half-up: (sum + count//2) // count
                avg = (s + count // 2) // count
                output[b, oy, ox, :] = np.clip(avg, act_min, act_max).astype(np.int8)
    return output


def _ref_maxpool_int8(input_nhwc: np.ndarray,
                      filter_h: int, filter_w: int,
                      stride_h: int, stride_w: int,
                      pad_h: int, pad_w: int,
                      act_min: int = -128, act_max: int = 127) -> np.ndarray:
    """Scalar reference for int8 MaxPool."""
    n, in_h, in_w, c = input_nhwc.shape
    out_h = (in_h + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - filter_w) // stride_w + 1

    padded = np.pad(input_nhwc.astype(np.int32),
                    ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode='constant', constant_values=-128)

    output = np.empty((n, out_h, out_w, c), dtype=np.int8)
    for b in range(n):
        for oy in range(out_h):
            for ox in range(out_w):
                window = padded[b,
                                oy * stride_h: oy * stride_h + filter_h,
                                ox * stride_w: ox * stride_w + filter_w, :]
                mx = window.max(axis=(0, 1))
                output[b, oy, ox, :] = np.clip(mx, act_min, act_max).astype(np.int8)
    return output


# ==============================================================================
# Test fixture class
# ==============================================================================

class PoolingOpTest:
    """
    Manages one pooling test case:
      - Loads the ELF, writes parameters & input data
      - Runs ref and opt kernels, compares outputs
      - Appends results to TEST_RESULTS and prints the running summary
    """

    SYMBOL_LIST = [
        'impl',
        'input_shape', 'output_shape',
        'stride_height', 'stride_width',
        'filter_height', 'filter_width',
        'pad_height', 'pad_width',
        'activation_min', 'activation_max',
        'input_data', 'output_data',
        # ref/opt entry points
        'run_ref_avgpool', 'run_opt_avgpool',
        'run_ref_maxpool', 'run_opt_maxpool',
    ]

    def __init__(self, dut, op: str,
                 input_shape,
                 filter_h: int, filter_w: int,
                 stride_h: int = 1, stride_w: int = 1,
                 pad_h: int = 0, pad_w: int = 0,
                 act_min: int = -128, act_max: int = 127):
        """
        Args:
            op:           'avg' or 'max'
            input_shape:  [N, H, W, C]
            filter_h/w:   pooling window
            stride_h/w:   stride
            pad_h/w:      SAME-style symmetric padding
            act_min/max:  quantized activation clipping range
        """
        assert op in ('avg', 'max'), "op must be 'avg' or 'max'"
        self.dut = dut
        self.op = op
        self.input_shape  = np.array(input_shape, dtype=np.int32)
        self.filter_h     = filter_h
        self.filter_w     = filter_w
        self.stride_h     = stride_h
        self.stride_w     = stride_w
        self.pad_h        = pad_h
        self.pad_w        = pad_w
        self.act_min      = act_min
        self.act_max      = act_max

        n, h, w, c = input_shape
        out_h = (h + 2 * pad_h - filter_h) // stride_h + 1
        out_w = (w + 2 * pad_w - filter_w) // stride_w + 1
        self.output_shape = np.array([n, out_h, out_w, c], dtype=np.int32)
        self.flat_out_size = int(np.prod(self.output_shape))

        r = runfiles.Create()
        self.elf_file = r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/pooling_test.elf')
        self.fixture  = None
        self.in_data  = None

    async def setup(self):
        self.fixture = await Fixture.Create(self.dut, highmem=True)
        await self.fixture.load_elf_and_lookup_symbols(
            self.elf_file, self.SYMBOL_LIST)

        # Random input
        rng = np.random.default_rng(42)
        self.in_data = rng.integers(-50, 50,
                                    size=tuple(self.input_shape),
                                    dtype=np.int8)

        # Write shapes
        await self.fixture.write('input_shape',  self.input_shape)
        await self.fixture.write('output_shape', self.output_shape)

        # Write pooling parameters (each as a single int32)
        for name, val in [
            ('stride_height', self.stride_h),
            ('stride_width',  self.stride_w),
            ('filter_height', self.filter_h),
            ('filter_width',  self.filter_w),
            ('pad_height',    self.pad_h),
            ('pad_width',     self.pad_w),
            ('activation_min', self.act_min),
            ('activation_max', self.act_max),
        ]:
            await self.fixture.write(name, np.array([val], dtype=np.int32))

        # Write input data (NHWC flat)
        await self.fixture.write('input_data', self.in_data.flatten())

    async def _run_kernel(self, func_name: str, timeout: int):
        await self.fixture.write_ptr('impl', func_name)
        await self.fixture.write('output_data',
                                 np.zeros(self.flat_out_size, dtype=np.int8))
        cycles = await self.fixture.run_to_halt(timeout_cycles=timeout)
        raw    = await self.fixture.read('output_data', self.flat_out_size)
        return raw.view(np.int8), cycles

    async def run_compare(self,
                          ref_cycles_limit: int,
                          opt_cycles_limit: int,
                          check_opt: bool = True):
        ref_func = f'run_ref_{self.op}pool'
        opt_func = f'run_opt_{self.op}pool'

        # ---- Run Reference kernel ----
        ref_out, ref_cycles = await self._run_kernel(
            ref_func, tolerate(ref_cycles_limit))

        opt_cycles = None
        speedup    = None
        status     = 'PASS'

        if check_opt:
            # ---- Run Optimized kernel ----
            opt_out, opt_cycles = await self._run_kernel(
                opt_func, tolerate(opt_cycles_limit))

            # ---- Verify against pure-Python reference ----
            if self.op == 'avg':
                golden = _ref_avgpool_int8(
                    self.in_data,
                    self.filter_h, self.filter_w,
                    self.stride_h, self.stride_w,
                    self.pad_h, self.pad_w,
                    self.act_min, self.act_max).flatten()
            else:
                golden = _ref_maxpool_int8(
                    self.in_data,
                    self.filter_h, self.filter_w,
                    self.stride_h, self.stride_w,
                    self.pad_h, self.pad_w,
                    self.act_min, self.act_max).flatten()

            # Allow ±1 rounding tolerance
            diff_opt_ref    = np.abs(opt_out.astype(int) - ref_out.astype(int))
            diff_opt_golden = np.abs(opt_out.astype(int) - golden.astype(int))
            max_diff        = int(np.max(diff_opt_ref))

            if max_diff > 1:
                status = 'FAIL'
                fail_idx = np.where(diff_opt_ref > 1)[0]
                first    = fail_idx[0]
                self.dut._log.error(
                    f"MISMATCH at flat index {first}: "
                    f"ref={ref_out[first]}  opt={opt_out[first]}  "
                    f"golden={golden[first]}")
                self.dut._log.error(
                    f"  Max |opt-ref| diff: {max_diff}  "
                    f"  Max |opt-golden| diff: {int(np.max(diff_opt_golden))}")

            speedup = ref_cycles / opt_cycles if opt_cycles > 0 else 0.0
        else:
            status = 'REF_ONLY'

        n, h, w, c = self.input_shape
        TEST_RESULTS.append({
            'op':      self.op.upper() + 'Pool',
            'shape':   f'[{n},{h},{w},{c}]',
            'filter':  f'{self.filter_h}x{self.filter_w}',
            'stride':  f'{self.stride_h}x{self.stride_w}',
            'pad':     f'{self.pad_h}x{self.pad_w}',
            'ref':     ref_cycles,
            'opt':     opt_cycles,
            'speedup': speedup,
            'status':  status,
        })

        print_global_summary(self.dut._log)

        if status == 'FAIL':
            assert False, (
                f"{self.op.upper()}Pool mismatch > 1 LSB. "
                f"Max diff: {max_diff}")


# ==============================================================================
# Test Cases — AveragePool
# ==============================================================================

@cocotb.test()
async def test_01_avgpool_no_pad_2x2(dut):
    """AvgPool 2×2/s1, no padding, small tensor [1,4,4,8]"""
    t = PoolingOpTest(dut, 'avg',
                      input_shape=[1, 4, 4, 8],
                      filter_h=2, filter_w=2,
                      stride_h=1, stride_w=1,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=50_000,
                        opt_cycles_limit=10_000)


@cocotb.test()
async def test_02_avgpool_no_pad_2x2_stride2(dut):
    """AvgPool 2×2/s2, no padding [1,8,8,16]"""
    t = PoolingOpTest(dut, 'avg',
                      input_shape=[1, 8, 8, 16],
                      filter_h=2, filter_w=2,
                      stride_h=2, stride_w=2,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=200_000,
                        opt_cycles_limit=40_000)


@cocotb.test()
async def test_03_avgpool_same_pad_3x3(dut):
    """AvgPool 3×3/s1, SAME padding [1,8,8,32]"""
    # SAME padding for 3×3: pad = (3-1)//2 = 1
    t = PoolingOpTest(dut, 'avg',
                      input_shape=[1, 8, 8, 32],
                      filter_h=3, filter_w=3,
                      stride_h=1, stride_w=1,
                      pad_h=1, pad_w=1)
    await t.setup()
    await t.run_compare(ref_cycles_limit=500_000,
                        opt_cycles_limit=80_000)


@cocotb.test()
async def test_04_avgpool_global_small(dut):
    """AvgPool global (filter==input): [1,4,4,8] → [1,1,1,8]"""
    t = PoolingOpTest(dut, 'avg',
                      input_shape=[1, 4, 4, 8],
                      filter_h=4, filter_w=4,
                      stride_h=1, stride_w=1,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=50_000,
                        opt_cycles_limit=10_000)


@cocotb.test()
async def test_05_avgpool_batch2(dut):
    """AvgPool 2×2/s2, batch=2 [2,4,4,8]"""
    t = PoolingOpTest(dut, 'avg',
                      input_shape=[2, 4, 4, 8],
                      filter_h=2, filter_w=2,
                      stride_h=2, stride_w=2,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=100_000,
                        opt_cycles_limit=20_000)


@cocotb.test()
async def test_06_avgpool_mobilenet_gap(dut):
    """
    MobileNet Global Average Pooling: [1,4,4,256] → [1,1,1,256]
    Typical final pooling before the classifier head.
    """
    t = PoolingOpTest(dut, 'avg',
                      input_shape=[1, 4, 4, 256],
                      filter_h=4, filter_w=4,
                      stride_h=1, stride_w=1,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=2_000_000,
                        opt_cycles_limit=200_000)


@cocotb.test()
async def test_07_avgpool_depthwise_wide(dut):
    """AvgPool 2×2/s2 on wide channel tensor [1,8,8,128]"""
    t = PoolingOpTest(dut, 'avg',
                      input_shape=[1, 8, 8, 128],
                      filter_h=2, filter_w=2,
                      stride_h=2, stride_w=2,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=1_000_000,
                        opt_cycles_limit=100_000)


# ==============================================================================
# Test Cases — MaxPool
# ==============================================================================

@cocotb.test()
async def test_08_maxpool_no_pad_2x2(dut):
    """MaxPool 2×2/s1, no padding, small tensor [1,4,4,8]"""
    t = PoolingOpTest(dut, 'max',
                      input_shape=[1, 4, 4, 8],
                      filter_h=2, filter_w=2,
                      stride_h=1, stride_w=1,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=50_000,
                        opt_cycles_limit=10_000)


@cocotb.test()
async def test_09_maxpool_no_pad_2x2_stride2(dut):
    """MaxPool 2×2/s2, no padding [1,8,8,16]"""
    t = PoolingOpTest(dut, 'max',
                      input_shape=[1, 8, 8, 16],
                      filter_h=2, filter_w=2,
                      stride_h=2, stride_w=2,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=200_000,
                        opt_cycles_limit=40_000)


@cocotb.test()
async def test_10_maxpool_same_pad_3x3(dut):
    """MaxPool 3×3/s1, SAME padding [1,8,8,32]"""
    t = PoolingOpTest(dut, 'max',
                      input_shape=[1, 8, 8, 32],
                      filter_h=3, filter_w=3,
                      stride_h=1, stride_w=1,
                      pad_h=1, pad_w=1)
    await t.setup()
    await t.run_compare(ref_cycles_limit=500_000,
                        opt_cycles_limit=80_000)


@cocotb.test()
async def test_11_maxpool_batch2(dut):
    """MaxPool 2×2/s2, batch=2 [2,4,4,8]"""
    t = PoolingOpTest(dut, 'max',
                      input_shape=[2, 4, 4, 8],
                      filter_h=2, filter_w=2,
                      stride_h=2, stride_w=2,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=100_000,
                        opt_cycles_limit=20_000)


@cocotb.test()
async def test_12_maxpool_wide_channel(dut):
    """MaxPool 2×2/s2, wide channels [1,8,8,128]"""
    t = PoolingOpTest(dut, 'max',
                      input_shape=[1, 8, 8, 128],
                      filter_h=2, filter_w=2,
                      stride_h=2, stride_w=2,
                      pad_h=0, pad_w=0)
    await t.setup()
    await t.run_compare(ref_cycles_limit=1_000_000,
                        opt_cycles_limit=80_000)


@cocotb.test()
async def test_13_maxpool_mobilenet_style(dut):
    """
    MobileNet-style MaxPool 3×3/s2 with SAME padding [1,8,8,256]
    pad = (3-1)//2 = 1 to keep spatial dims halved.
    """
    t = PoolingOpTest(dut, 'max',
                      input_shape=[1, 8, 8, 256],
                      filter_h=3, filter_w=3,
                      stride_h=2, stride_w=2,
                      pad_h=1, pad_w=1)
    await t.setup()
    await t.run_compare(ref_cycles_limit=5_000_000,
                        opt_cycles_limit=500_000)


@cocotb.test()
async def test_14_maxpool_act_clip(dut):
    """MaxPool with non-default activation clipping [-64, 63], [1,4,4,16]"""
    t = PoolingOpTest(dut, 'max',
                      input_shape=[1, 4, 4, 16],
                      filter_h=2, filter_w=2,
                      stride_h=1, stride_w=1,
                      pad_h=0, pad_w=0,
                      act_min=-64, act_max=63)
    await t.setup()
    await t.run_compare(ref_cycles_limit=100_000,
                        opt_cycles_limit=20_000)


# ==============================================================================
# Ref-only smoke tests (verify harness wiring without opt comparison)
# ==============================================================================

@cocotb.test()
async def test_15_ref_avgpool_smoke(dut):
    """Ref-only smoke: AvgPool [1,16,16,32] 3×3/s2/pad1"""
    t = PoolingOpTest(dut, 'avg',
                      input_shape=[1, 16, 16, 32],
                      filter_h=3, filter_w=3,
                      stride_h=2, stride_w=2,
                      pad_h=1, pad_w=1)
    await t.setup()
    await t.run_compare(ref_cycles_limit=2_000_000,
                        opt_cycles_limit=0,
                        check_opt=False)


@cocotb.test()
async def test_16_ref_maxpool_smoke(dut):
    """Ref-only smoke: MaxPool [1,16,16,32] 3×3/s2/pad1"""
    t = PoolingOpTest(dut, 'max',
                      input_shape=[1, 16, 16, 32],
                      filter_h=3, filter_w=3,
                      stride_h=2, stride_w=2,
                      pad_h=1, pad_w=1)
    await t.setup()
    await t.run_compare(ref_cycles_limit=2_000_000,
                        opt_cycles_limit=0,
                        check_opt=False)
