import cocotb
import numpy as np

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

def tolerate(target: int, tolerance = 1.5) -> int:
    return int(target * tolerance)

class ConvTest:
    def __init__(self, in_ch, out_ch, h=4, w=4, kernel_size=1, stride=1):
        self.stride = stride
        self.in_shape = np.array([1, h, w, in_ch], dtype=np.uint32)
        # TFLite Conv2D Weights: [Out, H, W, In]
        self.f_shape = np.array([out_ch, kernel_size, kernel_size, in_ch], dtype=np.uint32)
        self.bias_shape = np.array([out_ch], dtype=np.uint32)
        
        out_h = h // stride 
        out_w = w // stride
        self.out_shape = np.array([1, out_h, out_w, out_ch], dtype=np.uint32)
        self.out_size = int(np.prod(self.out_shape))

        r = runfiles.Create()
        self.elf_file = r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/conv_test.elf')
        self.fixture = None

    async def load_and_populate_input(self, dut):
        self.fixture = await Fixture.Create(dut, highmem=True)
        await self.fixture.load_elf_and_lookup_symbols(
            self.elf_file,
            [
                'impl',
                'run_ref',
                'run_optimized',
                'stride_width',
                'stride_height',
                'filter_shape',
                'filter_data',
                'bias_shape',
                'bias_data',
                'input_shape',
                'input_data',
                'output_shape',
                'output_data',
            ]
        )

        rng = np.random.default_rng(42)
        
        # 随机生成数据
        filter_data = rng.integers(-127, 127, self.f_shape, dtype=np.int8).flatten()
        bias_data = rng.integers(-5000, 5000, self.out_shape[3], dtype=np.int32)
        input_data = rng.integers(-128, 127, self.in_shape, dtype=np.int8).flatten()

        await self.fixture.write_word('stride_width', self.stride)
        await self.fixture.write_word('stride_height', self.stride)
        await self.fixture.write('filter_shape', self.f_shape)
        await self.fixture.write('filter_data', filter_data)
        await self.fixture.write('bias_shape', self.bias_shape)
        await self.fixture.write('bias_data', bias_data)
        await self.fixture.write('input_shape', self.in_shape)
        await self.fixture.write('input_data', input_data)
        await self.fixture.write('output_shape', self.out_shape)

    async def run(self, func_ptr: str, timeout_cycles):
        await self.fixture.write_ptr('impl', func_ptr)
        # 清零输出
        await self.fixture.write('output_data', np.zeros([self.out_size], dtype=np.int8))
        
        cycles = await self.fixture.run_to_halt(timeout_cycles=timeout_cycles)
        outputs = (await self.fixture.read('output_data', self.out_size)).view(np.int8)
        return outputs, cycles

    async def test(self, ref_target, opt_target):
        print(f"Running Reference (Timeout: {tolerate(ref_target)})...", flush=True)
        ref_output, ref_cycles = await self.run('run_ref', tolerate(ref_target))
        print(f'Done. ref_cycles={ref_cycles}', flush=True)
        
        print(f"Running Optimized (Timeout: {tolerate(opt_target)})...", flush=True)
        opt_output, opt_cycles = await self.run('run_optimized', tolerate(opt_target))
        print(f'Done. opt_cycles={opt_cycles}', flush=True)

        # 验证正确性
        if not (opt_output == ref_output).all():
            mismatch_idx = np.where(opt_output != ref_output)[0]
            print(f"Mismatch found at indices: {mismatch_idx[:10]}", flush=True)
            print(f"Ref: {ref_output[mismatch_idx[:10]]}", flush=True)
            print(f"Opt: {opt_output[mismatch_idx[:10]]}", flush=True)
            assert False, "Output mismatch!"
            
        print(f"Speedup: {ref_cycles / opt_cycles:.2f}x", flush=True)

# === 测试用例 ===

# 1. 冒烟测试
@cocotb.test()
async def test_conv1x1_mini(dut):
    t = ConvTest(in_ch=8, out_ch=8, h=4, w=4)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=60_000, opt_target=15_000)

# 2. 向量化测试: 32通道
# 增加了 ref_target 到 800,000
@cocotb.test()
async def test_conv1x1_standard(dut):
    t = ConvTest(in_ch=32, out_ch=32, h=4, w=4)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=800_000, opt_target=100_000)

# 3. 奇数通道测试
@cocotb.test()
async def test_conv1x1_odd(dut):
    t = ConvTest(in_ch=15, out_ch=15, h=2, w=2)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=60_000, opt_target=15_000)
