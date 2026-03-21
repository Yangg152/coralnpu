import cocotb
import numpy as np
import sys

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

def tolerate(target: int, tolerance = 1.5) -> int:
    return int(target * tolerance)

# === TFLite Quantization Math Helpers ===
def sat_rounding_doubling_high_mul(a, b):
    """模拟 TFLite 的 SaturatingRoundingDoublingHighMul"""
    # a 是 numpy array (int32), b 是 Python int (scalar)
    a_64 = a.astype(np.int64)
    b_64 = np.int64(b) 
    
    nudge = np.int64(1 << 30)
    # (a * b + (1<<30)) >> 31
    return (a_64 * b_64 + nudge) >> 31

def rounding_right_shift(x, shift):
    """模拟 TFLite 的 Rounding Divide / Shift"""
    # shift 必须为正数，表示向右移多少位
    if shift == 0: return x
    mask = (1 << shift) - 1
    threshold = (1 << (shift - 1))
    # Python 的 >> 对于负数行为是算术右移，符合 C++ 行为
    return (x + threshold) >> shift

class ConvTest:
    all_results = []

    def __init__(self, in_ch, out_ch, h=32, w=32, kernel_size=1, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.h = h
        self.w = w
        self.kernel_size = kernel_size
        
        self.mode_str = f"K{kernel_size}x{kernel_size} S{stride} P{padding}"
        self.shape_str = f"({h},{w},{in_ch})->{out_ch}"

        self.in_shape = np.array([1, h, w, in_ch], dtype=np.uint32)
        self.f_shape = np.array([out_ch, kernel_size, kernel_size, in_ch], dtype=np.uint32)
        self.bias_shape = np.array([out_ch], dtype=np.uint32)
        
        out_h = (h + 2 * padding - kernel_size) // stride + 1
        out_w = (w + 2 * padding - kernel_size) // stride + 1
        
        self.out_shape = np.array([1, out_h, out_w, out_ch], dtype=np.uint32)
        self.out_size = int(np.prod(self.out_shape))
        
        r = runfiles.Create()
        self.elf_file = r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/conv_test.elf')
        self.fixture = None

        # 保存生成的数据以便 Python 计算
        self.input_vals = None
        self.filter_vals = None
        self.bias_vals = None

        # 固定的量化参数 (参考 C++ 代码)
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
            ['impl', 'run_ref', 'run_optimized',
             'stride_width', 'stride_height', 'padding_width', 'padding_height',
             'filter_shape', 'filter_data', 'bias_shape', 'bias_data',
             'input_shape', 'input_data', 'output_shape', 'output_data']
        )

        rng = np.random.default_rng(42)
        # 生成数据并保存原始形状以便 Python 计算
        self.filter_vals = rng.integers(-127, 127, self.f_shape, dtype=np.int8)
        self.bias_vals = rng.integers(-5000, 5000, self.out_shape[3], dtype=np.int32)
        self.input_vals = rng.integers(-128, 127, self.in_shape, dtype=np.int8)

        # 写入仿真器
        await self.fixture.write_word('stride_width', self.stride)
        await self.fixture.write_word('stride_height', self.stride)
        await self.fixture.write_word('padding_width', self.padding)
        await self.fixture.write_word('padding_height', self.padding)
        
        await self.fixture.write('filter_shape', self.f_shape)
        await self.fixture.write('filter_data', self.filter_vals.flatten())
        await self.fixture.write('bias_shape', self.bias_shape)
        await self.fixture.write('bias_data', self.bias_vals)
        await self.fixture.write('input_shape', self.in_shape)
        await self.fixture.write('input_data', self.input_vals.flatten())
        await self.fixture.write('output_shape', self.out_shape)

    async def run(self, func_ptr: str, timeout_cycles):
        await self.fixture.write_ptr('impl', func_ptr)
        await self.fixture.write('output_data', np.zeros([self.out_size], dtype=np.int8))
        cycles = await self.fixture.run_to_halt(timeout_cycles=timeout_cycles)
        outputs = (await self.fixture.read('output_data', self.out_size)).view(np.int8)
        return outputs, cycles

    def compute_python_ref(self):
        """在 Python 中计算 Golden Output，使用 NumPy 向量化加速"""
        print("  -> Computing Python Reference...", flush=True)
        
        # 1. Padding
        inp = self.input_vals[0] # [H, W, C]
        
        if self.padding > 0:
            # [FIXED] Padding 逻辑修正
            # TFLite 卷积公式: sum((Input + InputOffset) * Filter)
            # Padding 区域应当贡献 0，即 (PadVal + InputOffset) = 0
            # 所以 PadVal = -InputOffset
            pad_val = -self.input_offset
            
            # Pad H and W dimensions. 
            inp = np.pad(inp, 
                         ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 
                         'constant',
                         constant_values=pad_val) # 使用正确的填充值
        
        # 2. Im2Col / Sliding Window
        out_h = self.out_shape[1]
        out_w = self.out_shape[2]
        
        k = self.kernel_size
        s = self.stride
        
        # 提取窗口
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(inp, (k, k), axis=(0, 1)) 
        windows = windows[::s, ::s, ...] 
        windows = windows.transpose(0, 1, 3, 4, 2) # [OutH, OutW, K, K, C]
        
        # 3. 核心计算 (Int32 Accumulation)
        # 注意：这里会加上 offset，如果 padding 填的是 -offset，加完正好是 0
        input_i32 = windows.astype(np.int32) + self.input_offset
        weights_i32 = self.filter_vals.astype(np.int32) # [OutCh, K, K, InCh]
        
        # MatMul / TensorDot
        acc = np.tensordot(input_i32, weights_i32, axes=([2,3,4], [1,2,3]))
        
        # 4. Add Bias
        acc += self.bias_vals 

        # 5. Quantization Output Stage
        acc_scaled = sat_rounding_doubling_high_mul(acc, self.output_mult)
        
        if self.output_shift < 0:
            acc_scaled = rounding_right_shift(acc_scaled, -self.output_shift)
        else:
            acc_scaled = acc_scaled * (1 << self.output_shift)
            
        acc_final = acc_scaled + self.output_offset
        
        # 6. Clamping
        output = np.clip(acc_final, self.act_min, self.act_max).astype(np.int8)
        
        return output.flatten()


    async def test(self, ref_target, opt_target, run_ref=False, fixed_ref_cycles=None, check_python=False):
        print(f"\n[Start] {self.mode_str} {self.shape_str}", flush=True)

        # 1. 运行 Optimized (Simulator)
        opt_output, opt_cycles = await self.run('run_optimized', tolerate(opt_target))
        print(f"  -> Optimized Done: {opt_cycles} cycles", flush=True)

        ref_cycles = 0
        status = "SKIP"
        speedup = 0.0
        mismatch = False
        ref_output = None

        # 2. 处理 Reference 逻辑
        if fixed_ref_cycles is not None:
            # 使用固定周期数，不跑 C++ Ref
            ref_cycles = fixed_ref_cycles
            status = "FIXED"
            if opt_cycles > 0:
                speedup = ref_cycles / opt_cycles
            
            # 如果要求验证，则跑 Python Ref
            if check_python:
                ref_output = self.compute_python_ref()
                if not (opt_output == ref_output).all():
                    mismatch = True
                    status = "FAIL (Py)"
                else:
                    status = "PASS (Py)"
        
        elif run_ref:
            # 跑 C++ Ref (Simulator)
            print("  -> Running C++ Reference...", flush=True)
            ref_output, ref_cycles = await self.run('run_ref', tolerate(ref_target, 2.0))
            print(f"  -> Reference Done: {ref_cycles} cycles", flush=True)
            
            # 双重验证
            if check_python:
                py_ref = self.compute_python_ref()
                if not (py_ref == ref_output).all():
                    print("[WARNING] Python Ref disagrees with C++ Ref!")
            
            if not (opt_output == ref_output).all():
                mismatch = True
                status = "FAIL"
            else:
                status = "PASS"
            
            if opt_cycles > 0:
                speedup = ref_cycles / opt_cycles

        # 记录结果
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
            mismatch_idx = np.where(opt_output != ref_output)[0]
            print(f"Mismatch indices (first 5): {mismatch_idx[:5]}")
            print(f"Ref (Expected): {ref_output[mismatch_idx[:5]]}")
            print(f"Opt (Actual)  : {opt_output[mismatch_idx[:5]]}")
            diff = opt_output[mismatch_idx[:5]].astype(int) - ref_output[mismatch_idx[:5]].astype(int)
            print(f"Diff          : {diff}")
            assert False, "Output mismatch detected!"

    @classmethod
    def print_final_summary(cls):
        print("\n" + "="*120)
        print(f"{'BENCHMARK SUMMARY':^120}")
        print("="*120)
        print(f"{'Mode':<20} | {'Shape':<25} | {'Ref Cyc':<12} | {'Opt Cyc':<12} | {'Speedup':<8} | {'Status':<10}")
        print("-" * 120)
        
        for res in cls.all_results:
            mode = res['mode']
            shape = res['shape']
            ref = res['ref']
            opt = res['opt']
            sp = res['speedup']
            st = res['status']
            
            print(f"{mode:<20} | {shape:<25} | {ref:<12} | {opt:<12} | {sp:<7.2f}x | {st:<10}")

        print("-" * 120 + "\n", flush=True)


# ==============================================================================
# Tests
# ==============================================================================

@cocotb.test()
async def benchmark_mxu_vww_first_layer(dut):
    """Benchmark: VWW 首层 3x3 conv, stride 2, RGB input"""
    t = ConvTest(in_ch=3, out_ch=8, h=96, w=96, kernel_size=3, stride=2, padding=1)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=40_000_000, opt_target=5_000_000, run_ref=False, check_python=True)

@cocotb.test()
async def test_mxu_exact_1_tile(dut):
    """【完美对齐】1个 16x16 Tile: Pixels=16, Out_C=16, In_C=16"""
    t = ConvTest(in_ch=16, out_ch=16, h=4, w=4, kernel_size=1)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=200_000, opt_target=50_000, run_ref=True, check_python=True)

@cocotb.test()
async def test_mxu_unaligned_small(dut):
    """【不足对齐】Pixels 和 Channels 都不足 16"""
    t = ConvTest(in_ch=11, out_ch=7, h=3, w=3, kernel_size=1)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=100_000, opt_target=30_000, run_ref=True, check_python=True)

@cocotb.test()
async def test_mxu_tk_greater_than_16(dut):
    """【Tk 深度跨 Tile】输入深度超过 16"""
    t = ConvTest(in_ch=32, out_ch=16, h=5, w=5, kernel_size=1)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=500_000, opt_target=80_000, run_ref=True, check_python=True)

@cocotb.test()
async def test_mxu_large_channel(dut):
    """【大通道切换】Tk 最大化边界测试"""
    t = ConvTest(in_ch=128, out_ch=32, h=4, w=4, kernel_size=1)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=1_000_000, opt_target=150_000, run_ref=True, check_python=True)

@cocotb.test()
async def test_fallback_3x3_sanity(dut):
    """【退化回退测试】验证 3x3 会安全回到纯 RVV 而不走 MXU"""
    t = ConvTest(in_ch=4, out_ch=8, h=5, w=5, kernel_size=3, stride=1, padding=1)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=500_000, opt_target=200_000, run_ref=True, check_python=True)

@cocotb.test()
async def benchmark_mxu_mobilenet_v1_pw1(dut):
    """Benchmark: MobileNetV1 第一个 Pointwise Conv (1x1)"""
    t = ConvTest(in_ch=32, out_ch=64, h=64, w=64, kernel_size=1)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=80_000_000, opt_target=5_000_000, run_ref=True, check_python=True)

@cocotb.test()
async def benchmark_mxu_vww_bottleneck(dut):
    """Benchmark: VWW 网络的特征压缩瓶颈层"""
    t = ConvTest(in_ch=128, out_ch=32, h=16, w=16, kernel_size=1)
    await t.load_and_populate_input(dut)
    await t.test(ref_target=40_000_000, opt_target=1_500_000, run_ref=True, check_python=True)

@cocotb.test()
async def z_final_report(dut):
    ConvTest.print_final_summary()
