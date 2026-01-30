# tests/cocotb/tutorial/tfmicro/cocotb_mean.py
import cocotb
import numpy as np
import math
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

# ==============================================================================
# Global Storage for Final Summary
# ==============================================================================
TEST_RESULTS = []

def print_global_summary(log):
    """Prints a cumulative table of all test results so far."""
    header = (
        f"\n{'='*110}\n"
        f"{'Shape':<16} | {'Axis':<10} | {'Ref Cycles':>12} | {'Opt Cycles':>12} | {'Speedup':>8} | {'Status':>6}\n"
        f"{'-'*110}"
    )
    rows = []
    for res in TEST_RESULTS:
        # 格式化 Opt Cycles，如果是 N/A 显示为 -
        opt_cyc_str = f"{res['opt']:12d}" if res['opt'] is not None else f"{'-':>12}"
        speedup_str = f"{res['speedup']:8.2f}x" if res['speedup'] is not None else f"{'-':>8}"
        
        rows.append(
            f"{res['shape']:<16} | {res['axis']:<10} | "
            f"{res['ref']:12d} | {opt_cyc_str} | {speedup_str} | {res['status']:>6}"
        )
    
    table = "\n".join([header] + rows + [f"{'='*110}\n"])
    log.info(table)

# ==============================================================================
# Helper Class & Functions
# ==============================================================================

def tolerate(target: int, tolerance = 1.5) -> int:
    return int(target * tolerance)

class MeanOpTest:
    def __init__(self, dut, input_shape, axis, keep_dims=False):
        self.dut = dut
        self.input_shape = np.array(input_shape, dtype=np.int32)
        self.axis = np.array(axis, dtype=np.int32)
        self.keep_dims = keep_dims
        
        self.shape_tuple = tuple(input_shape)
        self.axis_tuple = tuple(axis)
        
        # Calculate expected output shape for buffer sizing
        dummy_in = np.zeros(input_shape)
        dummy_out = np.mean(dummy_in, axis=tuple(axis), keepdims=keep_dims)
        self.output_shape = np.array(dummy_out.shape, dtype=np.int32)
        self.flat_out_size = int(np.prod(self.output_shape))
        
        r = runfiles.Create()
        self.elf_file = r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/mean_test.elf')
        self.fixture = None
        self.in_data = None

    def quantize_multiplier(self, real_multiplier):
        if real_multiplier == 0: return 0, 0
        q, shift = math.frexp(real_multiplier)
        q_fixed = int(round(q * (1 << 31)))
        if q_fixed == (1 << 31):
            q_fixed //= 2
            shift += 1
        return q_fixed, shift

    async def setup(self):
        self.fixture = await Fixture.Create(self.dut, highmem=True)
        await self.fixture.load_elf_and_lookup_symbols(
            self.elf_file,
            [
                'impl', 'run_ref_mean', 'run_opt_mean',
                'input_shape', 'output_shape', 'axis_data',
                'input_dims_count', 'output_dims_count', 'axis_count', 'keep_dims',
                'input_data', 'output_data',
                'input_zero_point', 'output_zero_point', 
                'output_multiplier', 'output_shift'
            ]
        )
        
        # Random Input Generation
        rng = np.random.default_rng(42)
        self.in_data = rng.integers(-50, 50, size=tuple(self.input_shape), dtype=np.int8)
        
        # Setup Quantization Parameters
        # Using 1.0 real multiplier logic (InputScale == OutputScale)
        # Reference implementation handles the 1/N division internally by modifying these values.
        real_multiplier = 1.0 
        significand, shift = self.quantize_multiplier(real_multiplier)
        
        zp_in = 0
        zp_out = 0

        # Write Parameters to Memory
        await self.fixture.write('input_shape', self.input_shape)
        await self.fixture.write('output_shape', self.output_shape)
        await self.fixture.write('axis_data', self.axis)
        
        await self.fixture.write('input_dims_count', np.array([len(self.input_shape)], dtype=np.int32))
        await self.fixture.write('output_dims_count', np.array([len(self.output_shape)], dtype=np.int32))
        await self.fixture.write('axis_count', np.array([len(self.axis)], dtype=np.int32))
        await self.fixture.write('keep_dims', np.array([1 if self.keep_dims else 0], dtype=np.int32))
        
        await self.fixture.write('input_zero_point', np.array([zp_in], dtype=np.int32))
        await self.fixture.write('output_zero_point', np.array([zp_out], dtype=np.int32))
        await self.fixture.write('output_multiplier', np.array([significand], dtype=np.int32))
        await self.fixture.write('output_shift', np.array([shift], dtype=np.int32))
        
        await self.fixture.write('input_data', self.in_data.flatten())

    async def run_kernel(self, func_name, timeout):
        await self.fixture.write_ptr('impl', func_name)
        # Clear output buffer
        await self.fixture.write('output_data', np.zeros(self.flat_out_size, dtype=np.int8))
        
        cycles = await self.fixture.run_to_halt(timeout_cycles=timeout)
        output = await self.fixture.read('output_data', self.flat_out_size)
        return output.view(np.int8), cycles

    async def run_compare(self, ref_cycles_limit, opt_cycles_limit, check_opt=True):
        # 1. Run Reference
        ref_out, ref_cycles = await self.run_kernel('run_ref_mean', tolerate(ref_cycles_limit))
        
        opt_cycles = None
        speedup = None
        status = "PASS"

        if check_opt:
            # 2. Run Optimized
            opt_out, opt_cycles = await self.run_kernel('run_opt_mean', tolerate(opt_cycles_limit))

            # 3. Verify
            # Allow +/- 1 difference due to integer arithmetic rounding differences
            diff = ref_out.astype(int) - opt_out.astype(int)
            abs_diff = np.abs(diff)
            max_diff = np.max(abs_diff)
            
            if max_diff > 1:
                status = "FAIL"
                fail_indices = np.where(abs_diff > 1)[0]
                first_fail = fail_indices[0]
                
                self.dut._log.error(f"CRITICAL MISMATCH at index {first_fail}")
                self.dut._log.error(f"  Ref: {ref_out[first_fail]} vs Opt: {opt_out[first_fail]}")
                self.dut._log.error(f"  Max Diff: {max_diff}")
            
            speedup = ref_cycles / opt_cycles if opt_cycles > 0 else 0.0
        else:
            status = "REF_ONLY"

        # 4. Add to Global Results
        TEST_RESULTS.append({
            "shape": str(self.shape_tuple),
            "axis": str(self.axis_tuple),
            "ref": ref_cycles,
            "opt": opt_cycles,
            "speedup": speedup,
            "status": status
        })

        # 5. Print Global Summary
        print_global_summary(self.dut._log)
        
        if status == "FAIL":
             assert False, f"Mean Output Mismatch > 1. Max diff: {max_diff}"

# ==============================================================================
# Test Cases
# ==============================================================================

@cocotb.test()
async def test_01_global_pooling_small(dut):
    """Mean: Global Pooling Small (Batch=1, H=4, W=4, C=8), Axis=[1,2]"""
    test = MeanOpTest(dut, input_shape=[1, 4, 4, 8], axis=[1, 2], keep_dims=False)
    await test.setup()
    await test.run_compare(ref_cycles_limit=50_000, opt_cycles_limit=10_000, check_opt=True)

@cocotb.test()
async def test_02_global_pooling_batch(dut):
    """Mean: Global Pooling Batched (Batch=2, H=4, W=4, C=8), Axis=[1,2]"""
    test = MeanOpTest(dut, input_shape=[2, 4, 4, 8], axis=[1, 2], keep_dims=False)
    await test.setup()
    await test.run_compare(ref_cycles_limit=100_000, opt_cycles_limit=20_000, check_opt=True)

@cocotb.test()
async def test_model_mobilenet_gap(dut):
    """
    Real World Case: MobileNet Global Average Pooling
    From user uploaded image.
    Input:  [Batch=1, Height=4, Width=4, Channels=256]
    Axis:   [1, 2] (Reduce H and W)
    Output: [1, 1, 1, 256] (Keep Dims = True)
    """
    # 1. 设置参数
    # Batch=1, H=4, W=4, C=256
    input_shape = [1, 4, 4, 256]
    axis = [1, 2]
    
    # 2. 初始化测试
    # 你的优化算子逻辑是扁平化写入 [Batch, Channels]，
    # 这与 [Batch, 1, 1, Channels] 的内存布局是完全一致的，所以 check_opt=True 应该能通过。
    test = MeanOpTest(dut, input_shape=input_shape, axis=axis, keep_dims=True)
    await test.setup()

    # 3. 运行对比
    # 计算量估算:
    # 元素总数 = 4 * 4 * 256 = 4096 个 int8 加法。
    # Ref 实现: 标量循环，大约需要 4096 * 10 ~ 20 cycles = 40k ~ 80k cycles。
    # Opt 实现: RVV 向量化 (假设 VLEN=256bit/512bit, LMUL=2/4)，并行度很高。
    #          H*W=16，循环 16 次累加，外层循环 Channels/VL。
    #          预计非常快，可能在 5k-10k cycles 左右。
    # 我们设置比较宽裕的 limit 以防万一。
    await test.run_compare(ref_cycles_limit=200_000_00, opt_cycles_limit=50_000_00, check_opt=True)


