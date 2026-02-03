import cocotb
import numpy as np
import sys

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

def tolerate(target: int, tolerance = 1.5) -> int:
    return int(target * tolerance)

class ConvTest:
    # 收集测试结果
    all_results = []

    def __init__(self, in_ch, out_ch, h=32, w=32, kernel_size=1, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        
        self.op_name = "CONV2D"
        self.mode_str = f"K{kernel_size}x{kernel_size} S{stride} P{padding}"
        self.shape_str = f"({h},{w},{in_ch})->{out_ch}"

        self.in_shape = np.array([1, h, w, in_ch], dtype=np.uint32)
        self.f_shape = np.array([out_ch, kernel_size, kernel_size, in_ch], dtype=np.uint32)
        self.bias_shape = np.array([out_ch], dtype=np.uint32)
        
        # 计算 Output Shape
        out_h = (h + 2 * padding - kernel_size) // stride + 1
        out_w = (w + 2 * padding - kernel_size) // stride + 1
        
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
                'impl', 'run_ref', 'run_optimized',
                'stride_width', 'stride_height',
                'padding_width', 'padding_height',
                'filter_shape', 'filter_data',
                'bias_shape', 'bias_data',
                'input_shape', 'input_data',
                'output_shape', 'output_data',
            ]
        )

        rng = np.random.default_rng(42)
        filter_data = rng.integers(-127, 127, self.f_shape, dtype=np.int8).flatten()
        bias_data = rng.integers(-5000, 5000, self.out_shape[3], dtype=np.int32)
        input_data = rng.integers(-128, 127, self.in_shape, dtype=np.int8).flatten()

        await self.fixture.write_word('stride_width', self.stride)
        await self.fixture.write_word('stride_height', self.stride)
        await self.fixture.write_word('padding_width', self.padding)
        await self.fixture.write_word('padding_height', self.padding)
        
        await self.fixture.write('filter_shape', self.f_shape)
        await self.fixture.write('filter_data', filter_data)
        await self.fixture.write('bias_shape', self.bias_shape)
        await self.fixture.write('bias_data', bias_data)
        await self.fixture.write('input_shape', self.in_shape)
        await self.fixture.write('input_data', input_data)
        await self.fixture.write('output_shape', self.out_shape)

    async def run(self, func_ptr: str, timeout_cycles):
        await self.fixture.write_ptr('impl', func_ptr)
        # 清除输出 buffer
        await self.fixture.write('output_data', np.zeros([self.out_size], dtype=np.int8))
        
        cycles = await self.fixture.run_to_halt(timeout_cycles=timeout_cycles)
        outputs = (await self.fixture.read('output_data', self.out_size)).view(np.int8)
        return outputs, cycles

    # 增加 run_ref 参数，允许手动开启 Reference
    async def test(self, ref_target, opt_target, run_ref=False):
        print(f"\n[Start] {self.mode_str} {self.shape_str}", flush=True)

        # 1. 运行 Optimized
        opt_output, opt_cycles = await self.run('run_optimized', tolerate(opt_target))
        print(f"  -> Optimized Done: {opt_cycles} cycles", flush=True)

        # 2. 运行 Reference
        ref_cycles = 0
        status = "SKIP"
        speedup = 0.0
        mismatch = False

        if run_ref:
            print("  -> Running Reference...", flush=True)
            ref_output, ref_cycles = await self.run('run_ref', tolerate(ref_target, 2.0))
            print(f"  -> Reference Done: {ref_cycles} cycles", flush=True)
            
            if not (opt_output == ref_output).all():
                mismatch = True
                status = "FAIL"
            else:
                status = "PASS"
            
            if opt_cycles > 0:
                speedup = ref_cycles / opt_cycles
        else:
            status = "SKIP"

        result_entry = {
            "mode": self.mode_str,
            "shape": self.shape_str,
            "ref": ref_cycles,
            "opt": opt_cycles,
            "speedup": speedup,
            "status": status
        }
        ConvTest.all_results.append(result_entry)

        if mismatch:
            print(f"\n[ERROR] Output mismatch in {self.mode_str}!", flush=True)
            # 简单打印前几个不匹配的数据方便调试
            mismatch_idx = np.where(opt_output != ref_output)[0]
            print(f"Mismatch indices (first 5): {mismatch_idx[:5]}")
            print(f"Ref: {ref_output[mismatch_idx[:5]]}")
            print(f"Opt: {opt_output[mismatch_idx[:5]]}")
            assert False, "Output mismatch detected!"

    @classmethod
    def print_final_summary(cls):
        print("\n" + "="*110)
        print(f"{'BENCHMARK SUMMARY':^110}")
        print("="*110)
        print(f"{'Mode':<20} | {'Shape':<25} | {'Ref Cyc':<12} | {'Opt Cyc':<12} | {'Speedup':<8} | {'Status':<6}")
        print("-" * 110)
        
        for res in cls.all_results:
            mode = res['mode']
            shape = res['shape']
            ref = res['ref']
            opt = res['opt']
            sp = res['speedup']
            st = res['status']
            
            print(f"{mode:<20} | {shape:<25} | {ref:<12} | {opt:<12} | {sp:<7.2f}x | {st:<6}")

        print("-" * 110 + "\n", flush=True)


# === 1. 快速功能性测试 (Small Sanity Checks) ===

@cocotb.test()
async def test_conv1x1_mini(dut):
    """最小 1x1 测试: 检查基本数据流"""
    t = ConvTest(in_ch=8, out_ch=8, h=4, w=4)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=60_000, opt_target=15_000, run_ref=True)

@cocotb.test()
async def test_conv3x3_basic(dut):
    """基本 3x3 测试: 检查滑动窗口"""
    t = ConvTest(in_ch=2, out_ch=4, h=5, w=5, kernel_size=3, stride=1, padding=0)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=500_000, opt_target=100_000, run_ref=True)

@cocotb.test()
async def test_conv3x3_stride2(dut):
    """Stride 2 测试: 检查步长跳跃"""
    t = ConvTest(in_ch=2, out_ch=4, h=7, w=7, kernel_size=3, stride=2, padding=0)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=400_000, opt_target=80_000, run_ref=True)

@cocotb.test()
async def test_conv1x1_odd_channels(dut):
    """奇数通道测试: 检查 RVV Masking/Tail 处理"""
    t = ConvTest(in_ch=3, out_ch=5, h=2, w=2, kernel_size=1, stride=1, padding=0)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=200_000, opt_target=50_000, run_ref=True)


# === 2. 真实负载性能测试 (Benchmarks) ===

# Case 1: Op 0 (First Layer) - 耗时较长
@cocotb.test()
async def benchmark_op0_layer1(dut):
    # MBV1 0.25 Layer 0: 128x128x3 -> 64x64x8 (Stride 2)
    t = ConvTest(in_ch=3, out_ch=8, h=128, w=128, kernel_size=3, stride=2, padding=1)
    await t.load_and_populate_input(dut)
    
    # 标量预测需要 30M+ cycles，请耐心等待
    await t.test(ref_target=60_000_000, opt_target=6_000_000, run_ref=True)


# Case 2: Representative Pointwise Layer (Middle of Network)
@cocotb.test()
async def benchmark_pointwise_mid(dut):
    # 32x32x16 -> 32x32x32
    t = ConvTest(in_ch=16, out_ch=32, h=32, w=32, kernel_size=1, stride=1, padding=0)
    await t.load_and_populate_input(dut)
    
    await t.test(ref_target=30_000_000, opt_target=3_000_000, run_ref=True)


# === 3. 最终报告 ===
@cocotb.test()
async def z_final_report(dut):
    ConvTest.print_final_summary()
