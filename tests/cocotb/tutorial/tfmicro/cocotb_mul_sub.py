# tests/cocotb/tutorial/tfmicro/cocotb_mul_sub.py
import cocotb
import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

# ==============================================================================
# Global Storage for Final Summary
# ==============================================================================
TEST_RESULTS = []

def print_global_summary(log):
    """Prints a cumulative table of all test results so far."""
    header = (
        f"\n{'='*100}\n"
        f"{'OP':<5} | {'Mode':<10} | {'Shape':<16} | {'Ref Cycles':>12} | {'Opt Cycles':>12} | {'Speedup':>8} | {'Status':>6}\n"
        f"{'-'*100}"
    )
    rows = []
    for res in TEST_RESULTS:
        rows.append(
            f"{res['op']:<5} | {res['mode']:<10} | {res['shape']:<16} | "
            f"{res['ref']:12d} | {res['opt']:12d} | {res['speedup']:8.2f}x | {res['status']:>6}"
        )
    
    table = "\n".join([header] + rows + [f"{'='*100}\n"])
    log.info(table)

# ==============================================================================
# Helper Class & Functions
# ==============================================================================

def tolerate(target: int, tolerance = 1.5) -> int:
    return int(target * tolerance)

class BinaryOpTest:
    def __init__(self, dut, shape, broadcast_mode='none'):
        self.dut = dut
        self.shape = np.array([1, shape[0], shape[1], shape[2]], dtype=np.int32)
        self.flat_size = int(np.prod(self.shape))
        self.mode = broadcast_mode
        self.shape_tuple = tuple(shape) # Store for summary
        
        if broadcast_mode == 'none':
            self.input1_shape = self.shape
            self.input2_shape = self.shape
            self.desc = f"Element-wise {self.shape_tuple}"
        elif broadcast_mode == 'input2':
            self.input1_shape = self.shape
            self.input2_shape = np.array([1, 1, 1, 1], dtype=np.int32)
            self.desc = f"Broadcast In2(Scalar) -> In1{self.shape_tuple}"
        elif broadcast_mode == 'input1':
            self.input1_shape = np.array([1, 1, 1, 1], dtype=np.int32)
            self.input2_shape = self.shape
            self.desc = f"Broadcast In1(Scalar) -> In2{self.shape_tuple}"
        else:
            raise ValueError(f"Unknown broadcast mode: {broadcast_mode}")

        self.output_shape = self.shape
        
        r = runfiles.Create()
        self.elf_file = r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/mul_sub_test.elf')
        self.fixture = None

    async def setup(self):
        self.fixture = await Fixture.Create(self.dut, highmem=True)
        await self.fixture.load_elf_and_lookup_symbols(
            self.elf_file,
            [
                'impl', 
                'run_ref_mul', 'run_opt_mul',
                'run_ref_sub', 'run_opt_sub',
                'input1_shape', 'input2_shape', 'output_shape',
                'input1_data', 'input2_data', 'output_data',
                'input1_zero_point', 'input2_zero_point', 
                'output_zero_point', 'output_multiplier', 
                'output_shift', 'output_min', 'output_max'
            ]
        )
        
        rng = np.random.default_rng(42)
        self.in1 = rng.integers(-100, 100, size=tuple(self.input1_shape), dtype=np.int8)
        self.in2 = rng.integers(-100, 100, size=tuple(self.input2_shape), dtype=np.int8)
        
        mult = 1500000000 
        shift = -7
        
        await self.fixture.write('input1_shape', self.input1_shape)
        await self.fixture.write('input2_shape', self.input2_shape)
        await self.fixture.write('output_shape', self.output_shape)
        
        await self.fixture.write('input1_zero_point', np.array([0], dtype=np.int32))
        await self.fixture.write('input2_zero_point', np.array([0], dtype=np.int32))
        await self.fixture.write('output_zero_point', np.array([0], dtype=np.int32))
        await self.fixture.write('output_multiplier', np.array([mult], dtype=np.int32))
        await self.fixture.write('output_shift', np.array([shift], dtype=np.int32))
        await self.fixture.write('output_min', np.array([-128], dtype=np.int32))
        await self.fixture.write('output_max', np.array([127], dtype=np.int32))
        
        await self.fixture.write('input1_data', self.in1.flatten())
        await self.fixture.write('input2_data', self.in2.flatten())

    async def run_kernel(self, func_name, timeout):
        await self.fixture.write_ptr('impl', func_name)
        await self.fixture.write('output_data', np.zeros(self.flat_size, dtype=np.int8))
        
        cycles = await self.fixture.run_to_halt(timeout_cycles=timeout)
        output = await self.fixture.read('output_data', self.flat_size)
        return output.view(np.int8), cycles

    async def run_compare(self, op_type, ref_cycles_limit, opt_cycles_limit):
        if op_type == "MUL":
            ref_func = 'run_ref_mul'
            opt_func = 'run_opt_mul'
        elif op_type == "SUB":
            ref_func = 'run_ref_sub'
            opt_func = 'run_opt_sub'
        else:
            raise ValueError(f"Unknown op_type: {op_type}")

        # 1. Run Reference
        ref_out, ref_cycles = await self.run_kernel(ref_func, tolerate(ref_cycles_limit))
        
        # 2. Run Optimized
        opt_out, opt_cycles = await self.run_kernel(opt_func, tolerate(opt_cycles_limit))

        # 3. Verify
        diff = ref_out.astype(np.int16) - opt_out.astype(np.int16)
        abs_diff = np.abs(diff)
        
        fail_indices = np.where(abs_diff > 1)[0]
        status = "PASS"
        
        if len(fail_indices) > 0:
            status = "FAIL"
            first_fail = fail_indices[0]
            idx1 = 0 if np.prod(self.input1_shape) == 1 else first_fail
            idx2 = 0 if np.prod(self.input2_shape) == 1 else first_fail
            
            self.dut._log.error(f"CRITICAL MISMATCH at index {first_fail}")
            self.dut._log.error(f"  Ref: {ref_out[first_fail]} vs Opt: {opt_out[first_fail]}")
            self.dut._log.error(f"  In1: {self.in1.flatten()[idx1]}")
            self.dut._log.error(f"  In2: {self.in2.flatten()[idx2]}")
            # Don't assert here if we want to see the summary of other tests, 
            # but usually we want to fail fast. 
            # Let's assert AFTER adding to summary list so at least it's recorded if we catch it later.
        
        speedup = ref_cycles / opt_cycles if opt_cycles > 0 else 0.0

        # 4. Add to Global Results
        TEST_RESULTS.append({
            "op": op_type,
            "mode": self.mode,
            "shape": str(self.shape_tuple),
            "ref": ref_cycles,
            "opt": opt_cycles,
            "speedup": speedup,
            "status": status
        })

        # 5. Print Global Summary (Cumulative)
        print_global_summary(self.dut._log)
        
        if status == "FAIL":
             assert False, f"{op_type} Output Mismatch > 1"

# ==============================================================================
# Test Cases
# ==============================================================================

@cocotb.test()
async def test_01_mul_small(dut):
    """MUL: Small (4,4,3)"""
    test = BinaryOpTest(dut, shape=(4, 4, 3), broadcast_mode='none') 
    await test.setup()
    await test.run_compare("MUL", 50_000, 10_000)

@cocotb.test()
async def test_02_sub_small(dut):
    """SUB: Small (4,4,3)"""
    test = BinaryOpTest(dut, shape=(4, 4, 3), broadcast_mode='none')
    await test.setup()
    await test.run_compare("SUB", 50_000, 10_000)

@cocotb.test()
async def test_03_mul_elementwise(dut):
    """MUL: Large (128,128,3)"""
    test = BinaryOpTest(dut, shape=(128, 128, 3), broadcast_mode='none')
    await test.setup()
    await test.run_compare("MUL", 20_000_000, 500_000)

@cocotb.test()
async def test_04_mul_broadcast(dut):
    """MUL: Broadcast (128,128,3)"""
    test = BinaryOpTest(dut, shape=(128, 128, 3), broadcast_mode='input2')
    await test.setup()
    await test.run_compare("MUL", 20_000_000, 500_000)

@cocotb.test()
async def test_05_sub_elementwise(dut):
    """SUB: Large (128,128,3)"""
    test = BinaryOpTest(dut, shape=(128, 128, 3), broadcast_mode='none')
    await test.setup()
    await test.run_compare("SUB", 20_000_000, 500_000)

@cocotb.test()
async def test_06_sub_bcast_vec_scalar(dut):
    """SUB: Broadcast Vector-Scalar"""
    test = BinaryOpTest(dut, shape=(128, 128, 3), broadcast_mode='input2')
    await test.setup()
    await test.run_compare("SUB", 20_000_000, 500_000)

@cocotb.test()
async def test_07_sub_bcast_scalar_vec(dut):
    """SUB: Broadcast Scalar-Vector"""
    test = BinaryOpTest(dut, shape=(128, 128, 3), broadcast_mode='input1')
    await test.setup()
    await test.run_compare("SUB", 20_000_000, 500_000)
