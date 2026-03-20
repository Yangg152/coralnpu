import cocotb
import numpy as np
import sys

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

def tolerate(target: int, tolerance = 1.5) -> int:
    return int(target * tolerance)

# === TFLite Quantization Math Helpers ===
def sat_rounding_doubling_high_mul(a, b):
    a_64 = a.astype(np.int64)
    b_64 = np.int64(b)
    nudge = np.int64(1 << 30)
    return (a_64 * b_64 + nudge) >> 31

def rounding_right_shift(x, shift):
    if shift == 0: return x
    threshold = (1 << (shift - 1))
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
            'coralnpu_hw/tests/mxu/mxu_conv_test.elf')
        self.fixture = None

        self.input_vals = None
        self.filter_vals = None
        self.bias_vals = None

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
        self.filter_vals = rng.integers(-127, 127, self.f_shape, dtype=np.int8)
        self.bias_vals = rng.integers(-5000, 5000, self.out_shape[3], dtype=np.int32)
        self.input_vals = rng.integers(-128, 127, self.in_shape, dtype=np.int8)

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
        inp = self.input_vals[0]
        if self.padding > 0:
            pad_val = -self.input_offset
            inp = np.pad(inp,
                         ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                         'constant',
                         constant_values=pad_val)

        k = self.kernel_size
        s = self.stride

        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(inp, (k, k), axis=(0, 1))
        windows = windows[::s, ::s, ...]
        windows = windows.transpose(0, 1, 3, 4, 2)

        input_i32 = windows.astype(np.int32) + self.input_offset
        weights_i32 = self.filter_vals.astype(np.int32)

        acc = np.tensordot(input_i32, weights_i32, axes=([2, 3, 4], [1, 2, 3]))
        acc += self.bias_vals

        acc_scaled = sat_rounding_doubling_high_mul(acc, self.output_mult)
        if self.output_shift < 0:
            acc_scaled = rounding_right_shift(acc_scaled, -self.output_shift)
        else:
            acc_scaled = acc_scaled * (1 << self.output_shift)

        acc_final = acc_scaled + self.output_offset
        output = np.clip(acc_final, self.act_min, self.act_max).astype(np.int8)
        return output.flatten()

    def _compute_intermediate_ref(self):
        inp = self.input_vals[0]
        if self.padding > 0:
            pad_val = -self.input_offset
            inp = np.pad(inp,
                         ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                         'constant',
                         constant_values=pad_val)

        k = self.kernel_size
        s = self.stride

        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(inp, (k, k), axis=(0, 1))
        windows = windows[::s, ::s, ...]
        windows = windows.transpose(0, 1, 3, 4, 2)

        input_i32 = windows.astype(np.int32) + self.input_offset
        weights_i32 = self.filter_vals.astype(np.int32)

        acc_with_offset = np.tensordot(input_i32, weights_i32, axes=([2, 3, 4], [1, 2, 3]))
        acc_with_bias = acc_with_offset + self.bias_vals

        input_raw = windows.astype(np.int32)
        acc_raw = np.tensordot(input_raw, weights_i32, axes=([2, 3, 4], [1, 2, 3]))

        filter_sum = self.filter_vals.astype(np.int64).reshape(self.out_ch, -1).sum(axis=1)
        offset_comp = (filter_sum * self.input_offset).astype(np.int64)

        return {
            'acc_with_offset': acc_with_offset,
            'acc_with_bias': acc_with_bias,
            'acc_raw': acc_raw,
            'offset_comp': offset_comp,
            'filter_sum': filter_sum,
        }

    def _diagnose_stuck_pixel(self, opt_output, ref_output):
        """诊断 MXU 是否对所有 pixel 使用了同一个 pixel 的 activation 数据。
        
        方法：对每个 pixel，反推 MXU 实际使用的 activation 向量，
        然后检查是否所有 pixel 都匹配 pixel 0 的 activation。
        """
        out_h, out_w, out_ch = int(self.out_shape[1]), int(self.out_shape[2]), int(self.out_shape[3])
        k = self.kernel_size
        s = self.stride

        inp = self.input_vals[0]
        if self.padding > 0:
            pad_val = -self.input_offset
            inp = np.pad(inp,
                         ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                         'constant',
                         constant_values=pad_val)

        opt_3d = opt_output.reshape(out_h, out_w, out_ch)
        ref_3d = ref_output.reshape(out_h, out_w, out_ch)

        weights_i32 = self.filter_vals.astype(np.int32)  # (OC, KH, KW, IC)

        print(f"\n{'='*80}", flush=True)
        print(f"  STUCK-PIXEL DIAGNOSIS", flush=True)
        print(f"{'='*80}", flush=True)

        # For 1x1 conv, each pixel's activation is just inp[y, x, :] + offset
        # acc[y,x,oc] = sum_ic( (inp[y,x,ic] + offset) * weight[oc,0,0,ic] ) + bias[oc]
        # If MXU uses pixel P instead of pixel (y,x):
        # acc_wrong[y,x,oc] = sum_ic( (inp[Py,Px,ic] + offset) * weight[oc,0,0,ic] ) + bias[oc]

        # Strategy: compute what the output WOULD be if MXU always used pixel 0's activation
        pixel0_act = (inp[0, 0, :].astype(np.int32) + self.input_offset)  # (IC,)
        acc_pixel0 = np.dot(pixel0_act, weights_i32.reshape(out_ch, -1).T) + self.bias_vals  # (OC,)

        # Quantize pixel0 result
        acc_scaled = sat_rounding_doubling_high_mul(acc_pixel0, self.output_mult)
        if self.output_shift < 0:
            acc_scaled = rounding_right_shift(acc_scaled, -self.output_shift)
        else:
            acc_scaled = acc_scaled * (1 << self.output_shift)
        pixel0_quantized = np.clip(acc_scaled + self.output_offset, self.act_min, self.act_max).astype(np.int8)

        # Check: does opt output match pixel0's result for ALL pixels?
        pixel0_match_count = 0
        pixel0_mismatch_count = 0
        total_pixels = out_h * out_w

        for y in range(out_h):
            for x in range(out_w):
                if np.array_equal(opt_3d[y, x, :], pixel0_quantized):
                    pixel0_match_count += 1
                else:
                    pixel0_mismatch_count += 1

        print(f"  Hypothesis: MXU uses pixel(0,0) activation for ALL pixels", flush=True)
        print(f"    Pixels matching pixel(0,0) result: {pixel0_match_count}/{total_pixels}", flush=True)
        print(f"    Pixels NOT matching             : {pixel0_mismatch_count}/{total_pixels}", flush=True)

        if pixel0_match_count == total_pixels:
            print(f"  >>> CONFIRMED: MXU output is identical to pixel(0,0) for every pixel!", flush=True)
            print(f"  >>> Root cause: MLOAD_A always loads the same address (pixel 0).", flush=True)
            print(f"  >>> The activation base pointer is NOT advancing between pixels.", flush=True)
        elif pixel0_match_count > total_pixels * 0.8:
            print(f"  >>> LIKELY: Most pixels match pixel(0,0). Partial stuck behavior.", flush=True)

        # Also try: does each pixel match some FIXED pixel (not necessarily pixel 0)?
        # For each pixel in opt, find which source pixel's activation would produce that output
        print(f"\n  [Per-Pixel Source Identification]", flush=True)
        print(f"  For each output pixel, find which input pixel the MXU actually used:", flush=True)

        # Precompute quantized output for every possible source pixel
        all_pixel_results = np.zeros((out_h, out_w, out_ch), dtype=np.int8)
        for sy in range(out_h):
            for sx in range(out_w):
                iy = sy * s
                ix = sx * s
                act = inp[iy:iy+k, ix:ix+k, :].astype(np.int32).flatten() + self.input_offset
                acc = np.dot(act, weights_i32.reshape(out_ch, -1).T) + self.bias_vals
                sc = sat_rounding_doubling_high_mul(acc, self.output_mult)
                if self.output_shift < 0:
                    sc = rounding_right_shift(sc, -self.output_shift)
                else:
                    sc = sc * (1 << self.output_shift)
                all_pixel_results[sy, sx, :] = np.clip(sc + self.output_offset,
                                                        self.act_min, self.act_max).astype(np.int8)

        source_map = np.full((out_h, out_w), -1, dtype=np.int32)
        for y in range(out_h):
            for x in range(out_w):
                opt_pixel = opt_3d[y, x, :]
                for sy in range(out_h):
                    for sx in range(out_w):
                        if np.array_equal(opt_pixel, all_pixel_results[sy, sx, :]):
                            source_map[y, x] = sy * out_w + sx
                            break
                    if source_map[y, x] >= 0:
                        break

        print(f"\n  Source pixel map (linear index, -1=unknown):", flush=True)
        print(f"  Expected: each pixel maps to itself (diagonal pattern)", flush=True)
        for y in range(min(out_h, 8)):
            row_str = " ".join(f"{source_map[y, x]:3d}" for x in range(min(out_w, 16)))
            expected_str = " ".join(f"{y * out_w + x:3d}" for x in range(min(out_w, 16)))
            print(f"    row {y}: actual=[{row_str}]  expected=[{expected_str}]", flush=True)

        # Analyze the source map pattern
        unique_sources = np.unique(source_map)
        if len(unique_sources) == 1 and unique_sources[0] >= 0:
            src = unique_sources[0]
            sy, sx = src // out_w, src % out_w
            print(f"\n  >>> ALL pixels sourced from pixel ({sy},{sx}) = linear {src}", flush=True)
            print(f"  >>> This is a stuck activation pointer bug.", flush=True)
        elif len(unique_sources) <= 4:
            print(f"\n  >>> Only {len(unique_sources)} unique sources: {unique_sources}", flush=True)
            print(f"  >>> Activation pointer advances too slowly or wraps incorrectly.", flush=True)
        else:
            # Check if it's a shifted pattern
            correct_count = 0
            for y in range(out_h):
                for x in range(out_w):
                    expected_src = y * out_w + x
                    if source_map[y, x] == expected_src:
                        correct_count += 1
            print(f"\n  >>> Correctly sourced pixels: {correct_count}/{total_pixels}", flush=True)
            if correct_count < total_pixels:
                # Check for systematic offset
                offsets = []
                for y in range(out_h):
                    for x in range(out_w):
                        if source_map[y, x] >= 0:
                            expected = y * out_w + x
                            offsets.append(source_map[y, x] - expected)
                if offsets:
                    offsets = np.array(offsets)
                    unique_offsets, offset_counts = np.unique(offsets, return_counts=True)
                    print(f"  >>> Source offset distribution:", flush=True)
                    for off, cnt in zip(unique_offsets[:10], offset_counts[:10]):
                        print(f"      offset={off:+4d}  count={cnt}", flush=True)

        print(f"{'='*80}\n", flush=True)

    def _analyze_mismatch(self, opt_output, ref_output):
        out_h, out_w, out_ch = int(self.out_shape[1]), int(self.out_shape[2]), int(self.out_shape[3])

        opt_3d = opt_output.reshape(out_h, out_w, out_ch)
        ref_3d = ref_output.reshape(out_h, out_w, out_ch)
        diff_3d = opt_3d.astype(np.int32) - ref_3d.astype(np.int32)

        mismatch_mask = (opt_3d != ref_3d)
        total_mismatch = int(mismatch_mask.sum())
        total_elements = opt_output.size

        print(f"\n{'='*80}")
        print(f"  MISMATCH ANALYSIS: {self.mode_str} {self.shape_str}")
        print(f"{'='*80}")
        print(f"  Total elements : {total_elements}")
        print(f"  Mismatches     : {total_mismatch} ({100.0*total_mismatch/total_elements:.2f}%)")
        print(f"  Output shape   : ({out_h}, {out_w}, {out_ch})")

        flat_diff = diff_3d[mismatch_mask].flatten()
        if len(flat_diff) > 0:
            print(f"\n  [Diff Distribution]")
            print(f"    min={flat_diff.min()}, max={flat_diff.max()}, "
                  f"mean={flat_diff.mean():.2f}, median={np.median(flat_diff):.1f}")
            unique_diffs, counts = np.unique(flat_diff, return_counts=True)
            top_n = min(10, len(unique_diffs))
            sorted_idx = np.argsort(-counts)[:top_n]
            print(f"    Top {top_n} diff values:")
            for i in sorted_idx:
                print(f"      diff={unique_diffs[i]:+4d}  count={counts[i]}")

        per_ch_mismatch = mismatch_mask.sum(axis=(0, 1))
        print(f"\n  [Per-Channel Mismatch Count]")
        for oc in range(out_ch):
            if per_ch_mismatch[oc] > 0:
                print(f"    oc={oc:3d}: {per_ch_mismatch[oc]:6d} mismatches "
                      f"({100.0*per_ch_mismatch[oc]/(out_h*out_w):.1f}%)")

        per_row_mismatch = mismatch_mask.sum(axis=(1, 2))
        per_col_mismatch = mismatch_mask.sum(axis=(0, 2))
        print(f"\n  [Spatial Distribution]")
        print(f"    Per-row (out_y) mismatch: min={per_row_mismatch.min()}, "
              f"max={per_row_mismatch.max()}, nonzero_rows={np.count_nonzero(per_row_mismatch)}/{out_h}")
        print(f"    Per-col (out_x) mismatch: min={per_col_mismatch.min()}, "
              f"max={per_col_mismatch.max()}, nonzero_cols={np.count_nonzero(per_col_mismatch)}/{out_w}")

        mismatch_flat_idx = np.where(opt_output != ref_output)[0]
        pixel_idx = mismatch_flat_idx // out_ch
        channel_idx = mismatch_flat_idx % out_ch

        tile_boundary_count = 0
        tile_interior_count = 0
        for pidx in pixel_idx[:200]:
            pos_in_tile = pidx % 16
            if pos_in_tile == 0 or pos_in_tile == 15:
                tile_boundary_count += 1
            else:
                tile_interior_count += 1

        print(f"\n  [16-Pixel Tile Boundary Analysis] (first 200 mismatches)")
        print(f"    At tile boundary (pos 0 or 15): {tile_boundary_count}")
        print(f"    At tile interior              : {tile_interior_count}")

        oc_tile_boundary = 0
        oc_tile_interior = 0
        for cidx in channel_idx[:200]:
            pos_in_tile = cidx % 16
            if pos_in_tile == 0 or pos_in_tile == 15:
                oc_tile_boundary += 1
            else:
                oc_tile_interior += 1

        print(f"\n  [16-OC Tile Boundary Analysis] (first 200 mismatches)")
        print(f"    At OC tile boundary (pos 0 or 15): {oc_tile_boundary}")
        print(f"    At OC tile interior               : {oc_tile_interior}")

        intermediates = self._compute_intermediate_ref()
        acc_with_bias = intermediates['acc_with_bias']
        offset_comp = intermediates['offset_comp']
        filter_sum = intermediates['filter_sum']

        print(f"\n  [Detailed Samples] (first 20 mismatches)")
        print(f"    {'idx':>6} | {'(y,x,oc)':>12} | {'ref':>5} | {'opt':>5} | {'diff':>5} | "
              f"{'acc+bias':>12} | {'oc_comp':>10} | {'filt_sum':>10} | {'pixel_tile':>5} | {'oc_tile':>5}")
        print(f"    {'-'*110}")

        for i in range(min(20, len(mismatch_flat_idx))):
            flat_i = mismatch_flat_idx[i]
            p = flat_i // out_ch
            oc = flat_i % out_ch
            y = p // out_w
            x = p % out_w

            ref_val = int(ref_output[flat_i])
            opt_val = int(opt_output[flat_i])
            diff_val = opt_val - ref_val
            acc_val = int(acc_with_bias[y, x, oc])
            oc_comp = int(offset_comp[oc])
            f_sum = int(filter_sum[oc])
            p_tile = p % 16
            oc_tile = oc % 16

            print(f"    {flat_i:6d} | ({y:3d},{x:3d},{oc:3d}) | {ref_val:5d} | {opt_val:5d} | {diff_val:+5d} | "
                  f"{acc_val:12d} | {oc_comp:10d} | {f_sum:10d} | {p_tile:5d} | {oc_tile:5d}")

        all_off_by_one = np.all(np.abs(flat_diff) <= 1)
        if all_off_by_one:
            print(f"\n  [HINT] All diffs are +/-1 -> likely a rounding difference, not a logic bug.")

        for oc_start in range(0, out_ch, 16):
            oc_end = min(oc_start + 16, out_ch)
            tile_mismatches = mismatch_mask[:, :, oc_start:oc_end].sum()
            tile_total = out_h * out_w * (oc_end - oc_start)
            if tile_mismatches == tile_total and tile_total > 0:
                print(f"\n  [ALERT] OC tile [{oc_start}:{oc_end}] is 100% wrong! "
                      f"Likely offset_comp or quantization param misalignment.")

        opt_unique = np.unique(opt_output)
        if len(opt_unique) == 1:
            print(f"\n  [ALERT] Optimized output is constant: {opt_unique[0]}. "
                  f"MXU might not be writing results.")
        elif len(opt_unique) <= 3:
            print(f"\n  [ALERT] Optimized output has only {len(opt_unique)} unique values: {opt_unique}. "
                  f"Possible clamp/quantization issue.")

        # === 核心诊断：stuck pixel 检测 ===
        self._diagnose_stuck_pixel(opt_output, ref_output)

        # === 额外诊断：检查 opt 输出是否每个 pixel 都相同 ===
        print(f"\n  [Per-Pixel Constancy Check]", flush=True)
        opt_3d_local = opt_output.reshape(out_h, out_w, out_ch)
        pixel0_output = opt_3d_local[0, 0, :]
        all_same_as_p0 = True
        first_diff_pixel = None
        for y in range(out_h):
            for x in range(out_w):
                if not np.array_equal(opt_3d_local[y, x, :], pixel0_output):
                    all_same_as_p0 = False
                    if first_diff_pixel is None:
                        first_diff_pixel = (y, x)
                    break
            if not all_same_as_p0:
                break

        if all_same_as_p0:
            print(f"    ALL {out_h*out_w} pixels produce identical output vector!", flush=True)
            print(f"    pixel(0,0) output: {pixel0_output.tolist()}", flush=True)
            print(f"    >>> This CONFIRMS the activation data is not changing between pixels.", flush=True)
        else:
            # Count how many pixels match pixel 0
            match_p0 = sum(1 for y in range(out_h) for x in range(out_w)
                           if np.array_equal(opt_3d_local[y, x, :], pixel0_output))
            print(f"    Pixels identical to pixel(0,0): {match_p0}/{out_h*out_w}", flush=True)
            if first_diff_pixel:
                fy, fx = first_diff_pixel
                print(f"    First different pixel: ({fy},{fx})", flush=True)
                print(f"      pixel(0,0): {pixel0_output.tolist()}", flush=True)
                print(f"      pixel({fy},{fx}): {opt_3d_local[fy, fx, :].tolist()}", flush=True)

        # === 诊断：检查 16-pixel tile 边界行为 ===
        print(f"\n  [16-Pixel Tile Boundary Behavior]", flush=True)
        num_pixels = out_h * out_w
        opt_flat_pixels = opt_output.reshape(num_pixels, out_ch)
        num_tiles = (num_pixels + 15) // 16

        for tile_idx in range(min(num_tiles, 4)):
            p_start = tile_idx * 16
            p_end = min(p_start + 16, num_pixels)
            tile_data = opt_flat_pixels[p_start:p_end]

            # Check if all pixels in this tile are the same
            tile_all_same = all(np.array_equal(tile_data[j], tile_data[0])
                                for j in range(1, len(tile_data)))
            unique_in_tile = len(set(tuple(row) for row in tile_data))

            print(f"    Tile {tile_idx} [pixel {p_start}-{p_end-1}]: "
                  f"unique_vectors={unique_in_tile}/{p_end-p_start} "
                  f"{'ALL SAME' if tile_all_same else 'varied'}", flush=True)

            if tile_all_same:
                # Which source pixel does this tile's constant output correspond to?
                print(f"      Constant value: {tile_data[0].tolist()}", flush=True)

        print(f"{'='*80}\n", flush=True)

    async def test(self, ref_target, opt_target, run_ref=False, fixed_ref_cycles=None, check_python=False):
        print(f"\n[Start] {self.mode_str} {self.shape_str}", flush=True)

        opt_output, opt_cycles = await self.run('run_optimized', tolerate(opt_target))
        print(f"  -> Optimized Done: {opt_cycles} cycles", flush=True)

        ref_cycles = 0
        status = "SKIP"
        speedup = 0.0
        mismatch = False
        ref_output = None

        if fixed_ref_cycles is not None:
            ref_cycles = fixed_ref_cycles
            status = "FIXED"
            if opt_cycles > 0:
                speedup = ref_cycles / opt_cycles

            if check_python:
                ref_output = self.compute_python_ref()
                if not (opt_output == ref_output).all():
                    mismatch = True
                    status = "FAIL (Py)"
                else:
                    status = "PASS (Py)"

        elif run_ref:
            print("  -> Running C++ Reference...", flush=True)
            ref_output, ref_cycles = await self.run('run_ref', tolerate(ref_target, 2.0))
            print(f"  -> Reference Done: {ref_cycles} cycles", flush=True)

            if check_python:
                py_ref = self.compute_python_ref()
                if not (py_ref == ref_output).all():
                    py_diff = np.where(py_ref != ref_output)[0]
                    print(f"  [WARNING] Python Ref disagrees with C++ Ref at {len(py_diff)} positions!")
                    print(f"    First 5: idx={py_diff[:5]}, py={py_ref[py_diff[:5]]}, cpp={ref_output[py_diff[:5]]}")

            if not (opt_output == ref_output).all():
                mismatch = True
                status = "FAIL"
            else:
                status = "PASS"

            if opt_cycles > 0:
                speedup = ref_cycles / opt_cycles

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
            self._analyze_mismatch(opt_output, ref_output)
            assert False, f"Output mismatch detected in {self.mode_str} {self.shape_str}!"

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
async def test_mxu_debug_simple(dut):
    """【DEBUG】全 1 输入 × 全 1 权重，手算可验证"""
    # 最小 1x1 conv: 16 pixels, 16 output channels, 16 input channels
    t = ConvTest(in_ch=16, out_ch=16, h=4, w=4, kernel_size=1)
    await t.load_and_populate_input(dut)

    # 覆盖为固定已知数据
    # input: 全 1 (int8), shape [1, 4, 4, 16]
    fixed_input = np.ones(t.in_shape, dtype=np.int8)
    # filter: 全 1 (int8), shape [16, 1, 1, 16]
    fixed_filter = np.ones(t.f_shape, dtype=np.int8)
    # bias: 全 0
    fixed_bias = np.zeros(t.out_shape[3], dtype=np.int32)

    t.input_vals = fixed_input
    t.filter_vals = fixed_filter
    t.bias_vals = fixed_bias

    # 写入硬件
    await t.fixture.write('input_data', fixed_input.flatten())
    await t.fixture.write('filter_data', fixed_filter.flatten())
    await t.fixture.write('bias_data', fixed_bias)

    # 手算期望值:
    #   acc[pixel, oc] = sum_ic( (input[ic] + input_offset) * filter[oc, ic] ) + bias[oc]
    #                  = sum_ic( (1 + 128) * 1 ) + 0
    #                  = 16 * 129 = 2064
    #
    #   quantize: sat_rounding_doubling_high_mul(2064, 1215836872)
    #     = (2064 * 1215836872 + (1<<30)) >> 31
    #     ≈ 1168 (大约)
    #   >> (-output_shift) = >> 7 => 1168 >> 7 = 9 (带 rounding)
    #   + output_offset(-128) => 9 - 128 = -119
    #   clamp[-128, 127] => -119
    #
    # 所以期望输出: 全部 -119 (或附近，取决于精确 rounding)
    print("\n[DEBUG] Expected per-pixel accumulator = 16 * (1+128) * 1 = 2064", flush=True)

    py_ref = t.compute_python_ref()
    print(f"[DEBUG] Python ref output (first 16): {py_ref[:16].tolist()}", flush=True)
    print(f"[DEBUG] Python ref unique values: {np.unique(py_ref).tolist()}", flush=True)

    # 跑 optimized (MXU)
    opt_output, opt_cycles = await t.run('run_optimized', 100_000)
    print(f"[DEBUG] MXU output (first 16):    {opt_output[:16].tolist()}", flush=True)
    print(f"[DEBUG] MXU unique values:        {np.unique(opt_output).tolist()}", flush=True)
    print(f"[DEBUG] MXU cycles: {opt_cycles}", flush=True)

    # 逐 pixel 对比
    out_ch = int(t.out_shape[3])
    num_pixels = int(np.prod(t.out_shape[1:3]))
    mismatch_count = 0
    for p in range(num_pixels):
        s = p * out_ch
        e = s + out_ch
        if not np.array_equal(opt_output[s:e], py_ref[s:e]):
            if mismatch_count < 5:
                y, x = p // int(t.out_shape[2]), p % int(t.out_shape[2])
                print(f"[DEBUG] pixel({y},{x}): mxu={opt_output[s:e].tolist()} "
                      f"ref={py_ref[s:e].tolist()}", flush=True)
            mismatch_count += 1

    print(f"[DEBUG] Total pixel mismatches: {mismatch_count}/{num_pixels}", flush=True)

    if mismatch_count > 0:
        # 也跑一下 reference C++ 确认
        ref_output, ref_cycles = await t.run('run_ref', 500_000)
        print(f"[DEBUG] C++ ref (first 16):       {ref_output[:16].tolist()}", flush=True)
        assert False, f"{mismatch_count} pixels mismatch!"
        
@cocotb.test()
async def z_final_report(dut):
    ConvTest.print_final_summary()