import cocotb
import numpy as np

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

def tolerate(target: int, tolerance = 1.5) -> int:
    return int(target * tolerance)

class MeanTest:
    def __init__(self, batch, h, w, ch):
        self.batch = batch
        self.h = h
        self.w = w
        self.ch = ch
        
        # NHWC Layout
        self.in_shape = np.array([batch, h, w, ch], dtype=np.uint32)
        # Global Pooling Output: [batch, 1, 1, ch]
        self.out_shape = np.array([batch, 1, 1, ch], dtype=np.uint32)
        
        self.in_size = int(np.prod(self.in_shape))
        self.out_size = int(np.prod(self.out_shape))

        r = runfiles.Create()
        self.elf_file = r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/mean_test.elf')
        self.fixture = None
        self.dut = None  # <--- 新增：用于保存 DUT 引用
        
        self.test_data = {}

    async def setup_and_generate_data(self, dut):
        self.dut = dut  # <--- 新增：保存 DUT
        self.fixture = await Fixture.Create(dut, highmem=True)
        await self.fixture.load_elf_and_lookup_symbols(
            self.elf_file,
            [
                'impl',
                'run_ref',
                'run_optimized',
                'g_sync_flag',
                'input_shape_dims',
                'output_shape_dims',
                'input_data',
                'output_data',
                'input_zero_point',
                'output_zero_point',
                'multiplier',
                'shift',
                'axis_data',
                'axis_count'
            ]
        )

        rng = np.random.default_rng(42)
        
        self.test_data['input_data'] = rng.integers(-128, 127, self.in_shape, dtype=np.int8).flatten()
        
        num_elements = self.h * self.w
        real_multiplier = 1.0 / num_elements if num_elements > 0 else 0.0
        significand, shift_val = self.quantize_multiplier(real_multiplier)
        
        self.test_data['input_zp'] = int(rng.integers(-20, 20))
        self.test_data['output_zp'] = int(rng.integers(-20, 20))
        self.test_data['multiplier'] = int(significand)
        self.test_data['shift'] = int(shift_val)
        self.test_data['axis_data'] = np.array([1, 2], dtype=np.int32)
        self.test_data['axis_count'] = 2

    def quantize_multiplier(self, double_multiplier):
        if double_multiplier == 0.:
            return 0, 0
        q, shift = np.frexp(double_multiplier)
        q_fixed = int(round(q * (1 << 31)))
        if q_fixed == (1 << 31):
            q_fixed //= 2
            shift += 1
        return q_fixed, shift

    async def _write_data_to_dut(self):
        await self.fixture.write('input_shape_dims', self.in_shape)
        await self.fixture.write('output_shape_dims', self.out_shape)
        await self.fixture.write('input_data', self.test_data['input_data'])
        
        await self.fixture.write_word('input_zero_point', self.test_data['input_zp'])
        await self.fixture.write_word('output_zero_point', self.test_data['output_zp'])
        await self.fixture.write_word('multiplier', self.test_data['multiplier'])
        await self.fixture.write_word('shift', self.test_data['shift'])
        
        await self.fixture.write('axis_data', self.test_data['axis_data'])
        await self.fixture.write_word('axis_count', self.test_data['axis_count'])
        
        await self.fixture.write('output_data', np.zeros([self.out_size], dtype=np.int8))

    async def run(self, func_ptr_name: str, timeout_cycles):
        run_task = cocotb.start_soon(self.fixture.run_to_halt(timeout_cycles=timeout_cycles))
        
        # 【握手阶段 1】
        while True:
            raw_bytes = await self.fixture.read('g_sync_flag', 4)
            val = raw_bytes.view(np.int32)[0]
            if val == 1:
                break
            # 修改：使用 self.dut.clk 而不是 self.fixture.dut.clk
            await cocotb.triggers.ClockCycles(self.dut.clk, 100)
        
        # 【写入数据】
        await self._write_data_to_dut()
        await self.fixture.write_ptr('impl', func_ptr_name)
        
        # 【握手阶段 2】
        await self.fixture.write_word('g_sync_flag', 2)
        
        cycles = await run_task
        
        outputs = (await self.fixture.read('output_data', self.out_size)).view(np.int8)
        return outputs, cycles

    async def test(self, ref_target, opt_target):
        print(f"Running Reference (Timeout: {tolerate(ref_target)})...", flush=True)
        ref_output, ref_cycles = await self.run('run_ref', tolerate(ref_target))
        print(f'Done. ref_cycles={ref_cycles}', flush=True)
        
        print(f"Running Optimized (Timeout: {tolerate(opt_target)})...", flush=True)
        opt_output, opt_cycles = await self.run('run_optimized', tolerate(opt_target))
        print(f'Done. opt_cycles={opt_cycles}', flush=True)

        diff = np.abs(opt_output.astype(int) - ref_output.astype(int))
        if np.any(diff > 1):
            mismatch_idx = np.where(diff > 1)[0]
            print(f"Mismatch found at indices: {mismatch_idx[:10]}", flush=True)
            print(f"Ref: {ref_output[mismatch_idx[:10]]}", flush=True)
            print(f"Opt: {opt_output[mismatch_idx[:10]]}", flush=True)
            print(f"Diff: {diff[mismatch_idx[:10]]}", flush=True)
            assert False, "Output mismatch > 1!"
        
        if np.any(diff > 0):
            print(f"Warning: {np.sum(diff > 0)} outputs differ by 1 (acceptable rounding diff).")

        print(f"Speedup: {ref_cycles / opt_cycles:.2f}x", flush=True)

# === 测试用例 ===

@cocotb.test()
async def test_mean_small(dut):
    t = MeanTest(batch=1, h=4, w=4, ch=16)
    await t.setup_and_generate_data(dut)
    await t.test(ref_target=50_000, opt_target=10_000)

@cocotb.test()
async def test_mean_mobilenet_style(dut):
    t = MeanTest(batch=1, h=7, w=7, ch=1024)
    await t.setup_and_generate_data(dut)
    await t.test(ref_target=500_000, opt_target=50_000)

@cocotb.test()
async def test_mean_large_spatial(dut):
    t = MeanTest(batch=1, h=32, w=32, ch=32)
    await t.setup_and_generate_data(dut)
    await t.test(ref_target=600_000, opt_target=100_000)

@cocotb.test()
async def test_mean_odd_channel(dut):
    t = MeanTest(batch=1, h=4, w=4, ch=33)
    await t.setup_and_generate_data(dut)
    await t.test(ref_target=60_000, opt_target=15_000)
