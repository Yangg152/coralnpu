# tests/cocotb/tutorial/tfmicro/cocotb_softmax.py
import cocotb
import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

# ==============================================================================
# Global Storage for Final Summary
# ==============================================================================
TEST_RESULTS = []

def print_global_summary(log):
    header = (
        f"\n{'='*80}\n"
        f"{'OP':<8} | {'Shape':<16} | {'Ref Cycles':>12} | {'Opt Cycles':>12} | {'Speedup':>8} | {'Status':>6}\n"
        f"{'-'*80}"
    )
    rows = []
    for res in TEST_RESULTS:
        rows.append(
            f"{res['op']:<8} | {res['shape']:<16} | "
            f"{res['ref']:12d} | {res['opt']:12d} | {res['speedup']:8.2f}x | {res['status']:>6}"
        )
    table = "\n".join([header] + rows + [f"{'='*80}\n"])
    log.info(table)

def tolerate(target: int, tolerance = 1.5) -> int:
    return int(target * tolerance)

# ==============================================================================
# TFLite Quantization Helper
# ==============================================================================
def get_softmax_quant_params(beta=1.0):
    # 模拟 TFLite Prepare 阶段计算的参数
    # 这里硬编码一组典型的 int8 Softmax 参数
    # Input Scale 0.05, Beta 1.0 -> Multiplier/Shift for gemmlowp
    
    # 1. Input Multiplier (FixedPoint Q.31)
    # 对应 value = 0.5625 * 2^31
    input_multiplier = 1207959552 
    # 2. Input Left Shift
    input_left_shift = -5
    # 3. Diff Min (cutoff)
    diff_min = -4000 
    
    return input_multiplier, input_left_shift, diff_min

# ==============================================================================
# Test Class
# ==============================================================================
class SoftmaxTest:
    def __init__(self, dut, shape):
        self.dut = dut
        self.shape = np.array([1, shape[0], shape[1], shape[2]], dtype=np.int32)
        self.flat_size = int(np.prod(self.shape))
        self.shape_tuple = tuple(shape)
        
        r = runfiles.Create()
        # 确保这里的 ELF 文件名与 BUILD 文件中 data 属性一致
        self.elf_file = r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/softmax_test.elf')
        self.fixture = None

    async def setup(self):
        self.fixture = await Fixture.Create(self.dut, highmem=True)
        await self.fixture.load_elf_and_lookup_symbols(
            self.elf_file,
            [
                'impl', 'run_ref_softmax', 'run_opt_softmax',
                'shape_dims', 'dims_count',
                'input_data', 'output_data',
                'input_multiplier', 'input_left_shift', 'diff_min', 'beta'
            ]
        )
        
        # Generate Data
        rng = np.random.default_rng(42)
        self.in_data = rng.integers(-128, 127, size=tuple(self.shape), dtype=np.int8)
        
        mult, shift, dmin = get_softmax_quant_params()
        
        await self.fixture.write('shape_dims', self.shape)
        await self.fixture.write('input_multiplier', np.array([mult], dtype=np.int32))
        await self.fixture.write('input_left_shift', np.array([shift], dtype=np.int32))
        await self.fixture.write('diff_min', np.array([dmin], dtype=np.int32))
        await self.fixture.write('input_data', self.in_data.flatten())

    async def run_kernel(self, func_name, timeout):
        await self.fixture.write_ptr('impl', func_name)
        # Clear output
        await self.fixture.write('output_data', np.zeros(self.flat_size, dtype=np.int8))
        
        cycles = await self.fixture.run_to_halt(timeout_cycles=timeout)
        output = await self.fixture.read('output_data', self.flat_size)
        return output.view(np.int8), cycles

    async def run_compare(self, ref_limit, opt_limit):
        # 1. Ref
        ref_out, ref_cycles = await self.run_kernel('run_ref_softmax', tolerate(ref_limit))
        
        # 2. Opt
        opt_out, opt_cycles = await self.run_kernel('run_opt_softmax', tolerate(opt_limit))

        # 3. Verify
        # Softmax 允许 +/- 1 的误差 (由于计算顺序和定点精度差异)
        diff = np.abs(ref_out.astype(np.int16) - opt_out.astype(np.int16))
        fail_indices = np.where(diff > 1)[0]
        
        status = "PASS"
        if len(fail_indices) > 0:
            status = "FAIL"
            idx = fail_indices[0]
            self.dut._log.error(f"Mismatch at {idx}: Ref={ref_out[idx]}, Opt={opt_out[idx]}")
        
        speedup = ref_cycles / opt_cycles if opt_cycles > 0 else 0.0
        
        TEST_RESULTS.append({
            "op": "SOFTMAX",
            "shape": str(self.shape_tuple),
            "ref": ref_cycles,
            "opt": opt_cycles,
            "speedup": speedup,
            "status": status
        })
        
        print_global_summary(self.dut._log)
        
        if status == "FAIL":
            assert False, "Softmax output mismatch > 1"

# ==============================================================================
# Cocotb Test Cases (必须是顶层函数)
# ==============================================================================

@cocotb.test()
async def test_softmax_small_depth(dut):
    """Softmax: Small Depth (1, 1, 10)"""
    test = SoftmaxTest(dut, shape=(1, 1, 10))
    await test.setup()
    await test.run_compare(50_000, 20_0000)

@cocotb.test()
async def test_softmax_large_depth(dut):
    """Softmax: Large Depth (1, 1, 256)"""
    test = SoftmaxTest(dut, shape=(1, 1, 256))
    await test.setup()
    await test.run_compare(500_000, 100_000)

# @cocotb.test()
# async def test_softmax_batch(dut):
#     """Softmax: Batch (10, 10, 64)"""
#     test = SoftmaxTest(dut, shape=(10, 10, 64))
#     await test.setup()
#     await test.run_compare(5_000_000, 1_000_000)

@cocotb.test()
async def test_softmax_1x1000_reshaped(dut):
    """Softmax: 1x1000 (Simulating Reshape from 1x1x1x1000)"""
    # 对应图片中的场景：1x1000 的 Softmax
    # 传入 (1, 1, 1000) -> 实际 Shape [1, 1, 1, 1000]
    # Depth = 1000, Outer Loops = 1
    test = SoftmaxTest(dut, shape=(1, 1, 1000))
    await test.setup()
    await test.run_compare(ref_limit=2_000_000, opt_limit=500_000)

# ==============================================================================
# 性能扫描测试 (用于寻找标量/向量盈亏平衡点)
# ==============================================================================
@cocotb.test()
async def test_softmax_threshold_sweep(dut):
    """
    Sweep Softmax depth to find the optimal threshold (break-even point).
    """
    # 密集扫描 8 到 128 的范围，寻找交叉点
    depths_to_test = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128, 256]
    
    dut._log.info("==================================================")
    dut._log.info("      Starting Softmax Threshold Sweep Test       ")
    dut._log.info("==================================================")
    
    for d in depths_to_test:
        test = SoftmaxTest(dut, shape=(1, 1, d))
        await test.setup()
        # 将 timeout 放宽，因为较大的 depth 可能会跑较长时间
        await test.run_compare(ref_limit=10_000_000, opt_limit=10_000_000)
        
    dut._log.info("Sweep Test Completed! Please check the Global Summary Table.")

