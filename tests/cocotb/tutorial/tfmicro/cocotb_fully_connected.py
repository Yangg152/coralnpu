# tests/cocotb/tutorial/tfmicro/fc_test.py

import cocotb
import numpy as np

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture


def tolerate(target: int, tolerance: float = 1.5) -> int:
    return int(target * tolerance)


# ==============================================================================
# TFLite Quantization Math Helpers
# ==============================================================================

def sat_rounding_doubling_high_mul(a, b):
    """Replicates TFLite SaturatingRoundingDoublingHighMul."""
    a_64 = a.astype(np.int64)
    b_64 = np.int64(b)
    nudge = np.int64(1 << 30)
    return (a_64 * b_64 + nudge) >> 31


def rounding_right_shift(x, shift):
    """Arithmetic rounding right shift."""
    if shift == 0:
        return x
    threshold = 1 << (shift - 1)
    return (x + threshold) >> shift


# ==============================================================================
# FC Test Harness
# ==============================================================================

class FCTest:
    all_results = []

    def __init__(self, in_features: int, out_features: int, batch: int = 1):
        self.in_features = in_features
        self.out_features = out_features
        self.batch = batch
        self.gamma = in_features * out_features
        self.shape_str = f"({batch},{in_features})->{out_features}"

        # TFLite-like shape layout
        # input_shape  = [batch, in_features]
        # filter_shape = [out_features, in_features]
        # output_shape = [batch, out_features]
        self.in_shape = np.array([batch, in_features], dtype=np.uint32)
        self.f_shape = np.array([out_features, in_features], dtype=np.uint32)
        self.bias_shape = np.array([out_features], dtype=np.uint32)
        self.out_shape = np.array([batch, out_features], dtype=np.uint32)
        self.out_size = batch * out_features

        r = runfiles.Create()
        self.elf_file = r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/fully_connected_test.elf'
        )
        self.fixture = None

        self.input_vals = None
        self.filter_vals = None
        self.bias_vals = None

        # Fixed quantization params (aligned with C++ harness)
        self.input_offset = 128
        self.output_offset = -128
        self.output_mult = 1215836872
        self.output_shift = -7
        self.act_min = -128
        self.act_max = 127

    async def load_and_populate_input(self, dut):
        self.fixture = await Fixture.Create(dut, highmem=True)
        await self.fixture.load_elf_and_lookup_symbols(
            self.elf_file,
            [
                'impl', 'run_ref', 'run_optimized',
                'filter_shape', 'filter_data',
                'bias_shape', 'bias_data',
                'input_shape', 'input_data',
                'output_shape', 'output_data'
            ]
        )

        rng = np.random.default_rng(42)
        self.filter_vals = rng.integers(
            -127, 127, self.f_shape, dtype=np.int8
        )
        self.bias_vals = rng.integers(
            -5000, 5000, self.out_features, dtype=np.int32
        )
        self.input_vals = rng.integers(
            -128, 127, self.in_shape, dtype=np.int8
        )

        await self.fixture.write('filter_shape', self.f_shape)
        await self.fixture.write('filter_data', self.filter_vals.flatten())
        await self.fixture.write('bias_shape', self.bias_shape)
        await self.fixture.write('bias_data', self.bias_vals)
        await self.fixture.write('input_shape', self.in_shape)
        await self.fixture.write('input_data', self.input_vals.flatten())
        await self.fixture.write('output_shape', self.out_shape)

    async def run(self, func_ptr: str, timeout_cycles: int):
        await self.fixture.write_ptr('impl', func_ptr)
        await self.fixture.write(
            'output_data', np.zeros(self.out_size, dtype=np.int8)
        )
        cycles = await self.fixture.run_to_halt(timeout_cycles=timeout_cycles)
        outputs = (await self.fixture.read('output_data', self.out_size)).view(np.int8)
        return outputs, cycles

    def compute_python_ref(self) -> np.ndarray:
        """
        Python Golden for TFLite FullyConnectedPerChannel.

        For each (b, out_c):
            acc = sum_d(filter[out_c, d] * (input[b, d] + input_offset)) + bias[out_c]
            -> SaturatingRoundingDoublingHighMul(acc, mult)
            -> RoundingRightShift(..., -shift)   (when shift < 0)
            -> + output_offset
            -> clamp(act_min, act_max)
        """
        inp = self.input_vals.reshape(self.batch, self.in_features).astype(np.int32)
        inp += self.input_offset

        flt = self.filter_vals.reshape(
            self.out_features, self.in_features
        ).astype(np.int32)

        acc = inp @ flt.T
        acc += self.bias_vals  # broadcast over batch

        acc_scaled = sat_rounding_doubling_high_mul(acc, self.output_mult)

        if self.output_shift < 0:
            acc_scaled = rounding_right_shift(acc_scaled, -self.output_shift)
        else:
            acc_scaled = acc_scaled * (1 << self.output_shift)

        acc_final = acc_scaled + self.output_offset
        output = np.clip(acc_final, self.act_min, self.act_max).astype(np.int8)
        return output.flatten()

    async def test(
        self,
        ref_target: int,
        opt_target: int,
        check_python: bool = True,
    ):
        """
        Run optimized path + C++ reference path in the same RTL environment.
        Optionally cross-check Python golden.
        """
        print(f"\n[Start] FC {self.shape_str} (Γ={self.gamma})", flush=True)

        opt_output, opt_cycles = await self.run(
            'run_optimized', tolerate(opt_target)
        )
        print(f"  -> Optimized Done: {opt_cycles} cycles", flush=True)

        ref_output, ref_cycles = await self.run(
            'run_ref', tolerate(ref_target, 2.0)
        )
        print(f"  -> Reference Done: {ref_cycles} cycles", flush=True)

        speedup = ref_cycles / opt_cycles if opt_cycles > 0 else 0.0
        status = "PASS"
        mismatch = False

        if check_python:
            py_ref = self.compute_python_ref()
            if not (py_ref == ref_output).all():
                print("[WARNING] Python reference disagrees with C++ reference!")

        if not (opt_output == ref_output).all():
            mismatch = True
            status = "FAIL"

        FCTest.all_results.append({
            "shape": self.shape_str,
            "batch": self.batch,
            "in": self.in_features,
            "out": self.out_features,
            "gamma": self.gamma,
            "ref": ref_cycles,
            "opt": opt_cycles,
            "speedup": speedup,
            "status": status,
        })

        if mismatch:
            idx = np.where(opt_output != ref_output)[0]
            print(f"\n[ERROR] Output mismatch in FC {self.shape_str}!", flush=True)
            print(f"Mismatch indices (first 5): {idx[:5]}")
            print(f"Ref (Expected): {ref_output[idx[:5]]}")
            print(f"Opt (Actual)  : {opt_output[idx[:5]]}")
            diff = opt_output[idx[:5]].astype(int) - ref_output[idx[:5]].astype(int)
            print(f"Diff          : {diff}")
            assert False, "Output mismatch detected!"

    async def benchmark_fixed(
        self,
        fixed_ref_cycles: int,
        opt_target: int,
        check_python: bool = True,
    ):
        """
        Optional benchmark mode:
        only run optimized path in current test;
        use externally measured reference cycles.
        """
        print(f"\n[Benchmark] FC {self.shape_str} (Γ={self.gamma})", flush=True)

        opt_output, opt_cycles = await self.run(
            'run_optimized', tolerate(opt_target)
        )
        print(f"  -> Optimized Done: {opt_cycles} cycles", flush=True)

        status = "FIXED"
        if check_python:
            py_ref = self.compute_python_ref()
            # In fixed benchmark mode we don't have current C++ ref output,
            # so only check optimized against Python golden.
            if not (opt_output == py_ref).all():
                status = "FAIL (Py)"
                idx = np.where(opt_output != py_ref)[0]
                print(f"\n[ERROR] Output mismatch in FC {self.shape_str}!", flush=True)
                print(f"Mismatch indices (first 5): {idx[:5]}")
                print(f"Py Ref (Expected): {py_ref[idx[:5]]}")
                print(f"Opt    (Actual)  : {opt_output[idx[:5]]}")
                diff = opt_output[idx[:5]].astype(int) - py_ref[idx[:5]].astype(int)
                print(f"Diff             : {diff}")
                assert False, "Output mismatch detected!"

        speedup = fixed_ref_cycles / opt_cycles if opt_cycles > 0 else 0.0

        FCTest.all_results.append({
            "shape": self.shape_str,
            "batch": self.batch,
            "in": self.in_features,
            "out": self.out_features,
            "gamma": self.gamma,
            "ref": fixed_ref_cycles,
            "opt": opt_cycles,
            "speedup": speedup,
            "status": status,
        })

    @classmethod
    def print_final_summary(cls):
        print("\n" + "=" * 120)
        print(f"{'FC BENCHMARK SUMMARY':^120}")
        print("=" * 120)
        print(
            f"{'Shape':<22} | {'Batch':>5} | {'In':>5} | {'Out':>5} | "
            f"{'Γ=D×Cout':>10} | {'Ref Cyc':>10} | {'Opt Cyc':>10} | "
            f"{'Speedup':>8} | {'Status':<10}"
        )
        print("-" * 120)

        for r in sorted(
            cls.all_results,
            key=lambda x: (x["gamma"], x["in"], x["out"], x["batch"])
        ):
            print(
                f"{r['shape']:<22} | {r['batch']:>5} | {r['in']:>5} | {r['out']:>5} | "
                f"{r['gamma']:>10} | {r['ref']:>10} | {r['opt']:>10} | "
                f"{r['speedup']:>7.2f}x | {r['status']:<10}"
            )

        print("-" * 120 + "\n", flush=True)


# ==============================================================================
# Small helper
# ==============================================================================

async def _run_case(
    dut,
    in_features: int,
    out_features: int,
    batch: int,
    ref_target: int,
    opt_target: int,
    check_python: bool = True,
):
    t = FCTest(
        in_features=in_features,
        out_features=out_features,
        batch=batch,
    )
    await t.load_and_populate_input(dut)
    await t.test(
        ref_target=ref_target,
        opt_target=opt_target,
        check_python=check_python,
    )


# ==============================================================================
# Test set
# 目标：
# 1. 小规模/极小规模 correctness
# 2. 阈值附近（<512, =512, >512）
# 3. remainder / odd channels
# 4. 同 Γ 不同比例
# 5. VWW 分类头
# 6. 中大规模收益
# ==============================================================================

@cocotb.test()
async def test_fc_tiny_4_to_4(dut):
    """极小规模 sanity check: Γ=16"""
    await _run_case(
        dut,
        in_features=4,
        out_features=4,
        batch=1,
        ref_target=5_000,
        opt_target=5_000,
    )


@cocotb.test()
async def test_fc_small_16_to_8(dut):
    """小规模：Γ=128"""
    await _run_case(
        dut,
        in_features=16,
        out_features=8,
        batch=1,
        ref_target=10_000,
        opt_target=8_000,
    )


@cocotb.test()
async def test_fc_odd_channels_13_to_7(dut):
    """非对齐通道：Γ=91，验证 remainder 路径"""
    await _run_case(
        dut,
        in_features=13,
        out_features=7,
        batch=1,
        ref_target=10_000,
        opt_target=8_000,
    )


@cocotb.test()
async def test_fc_threshold_below_31_to_16(dut):
    """阈值下方：Γ=496"""
    await _run_case(
        dut,
        in_features=31,
        out_features=16,
        batch=1,
        ref_target=20_000,
        opt_target=8_000,
    )


@cocotb.test()
async def test_fc_threshold_exact_32_to_16(dut):
    """阈值命中：Γ=512"""
    await _run_case(
        dut,
        in_features=32,
        out_features=16,
        batch=1,
        ref_target=20_000,
        opt_target=8_000,
    )


@cocotb.test()
async def test_fc_threshold_above_33_to_16(dut):
    """阈值上方：Γ=528，同时验证 in=32+1 的 remainder"""
    await _run_case(
        dut,
        in_features=33,
        out_features=16,
        batch=1,
        ref_target=25_000,
        opt_target=10_000,
    )


@cocotb.test()
async def test_fc_same_gamma_32_to_8(dut):
    """同 Γ=256，不同比例之一：32->8"""
    await _run_case(
        dut,
        in_features=32,
        out_features=8,
        batch=1,
        ref_target=12_000,
        opt_target=8_000,
    )


@cocotb.test()
async def test_fc_same_gamma_8_to_32(dut):
    """同 Γ=256，不同比例之二：8->32"""
    await _run_case(
        dut,
        in_features=8,
        out_features=32,
        batch=1,
        ref_target=15_000,
        opt_target=10_000,
    )


@cocotb.test()
async def test_fc_batch4_32_to_16(dut):
    """批处理展平验证：batch=4"""
    await _run_case(
        dut,
        in_features=32,
        out_features=16,
        batch=4,
        ref_target=60_000,
        opt_target=20_000,
    )


@cocotb.test()
async def test_fc_vww_head_256_to_2(dut):
    """VWW 二分类头：Γ=512，贴近真实负载"""
    await _run_case(
        dut,
        in_features=256,
        out_features=2,
        batch=1,
        ref_target=40_000,
        opt_target=15_000,
    )


@cocotb.test()
async def test_fc_medium_64_to_32(dut):
    """中等规模：Γ=2048"""
    await _run_case(
        dut,
        in_features=64,
        out_features=32,
        batch=1,
        ref_target=80_000,
        opt_target=20_000,
    )


@cocotb.test()
async def test_fc_large_256_to_64(dut):
    """较大规模：Γ=16384"""
    await _run_case(
        dut,
        in_features=256,
        out_features=64,
        batch=1,
        ref_target=500_000,
        opt_target=80_000,
    )


# ==============================================================================
# Optional benchmarks (disabled by default)
# ==============================================================================

@cocotb.test(skip=True)
async def benchmark_fc_128_to_128(dut):
    """可选 benchmark：Γ=16384"""
    t = FCTest(in_features=128, out_features=128, batch=1)
    await t.load_and_populate_input(dut)
    await t.benchmark_fixed(
        fixed_ref_cycles=0,   # 先跑一次 run_ref 后填入
        opt_target=120_000,
        check_python=True,
    )


@cocotb.test(skip=True)
async def benchmark_fc_256_to_1000(dut):
    """可选 benchmark：大输出分类头"""
    t = FCTest(in_features=256, out_features=1000, batch=1)
    await t.load_and_populate_input(dut)
    await t.benchmark_fixed(
        fixed_ref_cycles=0,   # 先跑一次 run_ref 后填入
        opt_target=1_500_000,
        check_python=True,
    )


# ==============================================================================
# Final report
# ==============================================================================

@cocotb.test()
async def z_final_report(dut):
    FCTest.print_final_summary()
