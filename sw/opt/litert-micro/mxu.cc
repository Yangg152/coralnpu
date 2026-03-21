#include "sw/opt/litert-micro/mxu.h"
#include "sw/opt/litert-micro/conv.h"

#include <riscv_vector.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/rvv_opt.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/conv.h"

namespace coralnpu_v2::opt::litert_micro {

using tflite::ConvParams;
using tflite::RuntimeShape;

// =============================================================================
// MXU instruction primitives
// =============================================================================

static inline __attribute__((always_inline))
void mxu_mcfg(uint32_t config_val) {
    register uint32_t a0_val asm("a0") = config_val;
    asm volatile(".word 0x02051057" :: "r"(a0_val) : "memory");
}

static inline __attribute__((always_inline))
void mxu_mload_a_from_mem(const int8_t* src, bool is_last) {
    if (is_last) {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x09001057"
            :: "r"(src), "r"(16) : "memory", "v16");
    } else {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x0B001057"
            :: "r"(src), "r"(16) : "memory", "v16");
    }
}

static inline __attribute__((always_inline))
void mxu_mload_w_from_mem(const int8_t* src, bool is_last) {
    if (is_last) {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x05001057"
            :: "r"(src), "r"(16) : "memory", "v16");
    } else {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x07001057"
            :: "r"(src), "r"(16) : "memory", "v16");
    }
}

static inline __attribute__((always_inline))
void mxu_mzero() { asm volatile(".word 0x0E001057" ::: "memory"); }

static inline __attribute__((always_inline))
void mxu_mma() { asm volatile(".word 0x12001057" ::: "memory"); }

static inline __attribute__((always_inline))
void mxu_mstore_to_mem(int32_t* dst) {
    asm volatile(
        ".word 0x16001857\n\t"
        "vsetvli zero, %1, e32, m1, ta, ma\n\t"
        "vse32.v v16, (%0)"
        :: "r"(dst), "r"(4) : "memory", "v16");
}

static inline __attribute__((always_inline))
void mxu_mfence() { asm volatile(".word 0x1A001057" ::: "memory"); }

// =============================================================================
// Internal implementation
// =============================================================================
namespace {

struct AlignedFree {
    void operator()(void* ptr) const { std::free(ptr); }
};

template <typename T>
using aligned_array = std::unique_ptr<T[], AlignedFree>;

template <typename T>
aligned_array<T> make_aligned_array(size_t alignment, size_t nmemb) {
    size_t raw = sizeof(T) * nmemb;
    size_t aligned = ((raw + alignment - 1) / alignment) * alignment;
    void* ptr = aligned_alloc(alignment, aligned);
    return aligned_array<T>(reinterpret_cast<T*>(ptr));
}

inline __attribute__((always_inline)) vint8m1_t QuantizeResult_m4(
    vint32m4_t acc, vint32m4_t v_mult, vuint32m4_t v_lshift, vuint32m4_t v_rshift,
    int32_t output_offset, int32_t output_min, int32_t output_max, size_t vl) {
    constexpr uint32_t vxrm = 0;
    acc = __riscv_vsll_vv_i32m4(acc, v_lshift, vl);
    acc = __riscv_vsmul_vv_i32m4(acc, v_mult, vxrm, vl);
    acc = __riscv_vssra_vv_i32m4(acc, v_rshift, vxrm, vl);
    acc = __riscv_vadd_vx_i32m4(acc, output_offset, vl);
    vint16m2_t acc_16 = __riscv_vnclip_wx_i16m2(acc, 0, vxrm, vl);
    vint8m1_t acc_8 = __riscv_vnclip_wx_i8m1(acc_16, 0, vxrm, vl);
    acc_8 = __riscv_vmax_vx_i8m1(acc_8, (int8_t)output_min, vl);
    acc_8 = __riscv_vmin_vx_i8m1(acc_8, (int8_t)output_max, vl);
    return acc_8;
}

void Conv1x1PerChannel_MXU(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data)
{
    const int input_depth  = input_shape.Dims(3);
    const int output_depth = output_shape.Dims(3);
    const int num_pixels   = input_shape.Dims(0) * input_shape.Dims(1)
                           * input_shape.Dims(2);

    const int32_t input_offset  = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_min    = params.quantized_activation_min;
    const int32_t output_max    = params.quantized_activation_max;

    const int Tk = input_depth;
    const int num_chunks   = Tk / 16;
    const int num_oc_tiles = (output_depth + 15) / 16;

    auto lshift_data = make_aligned_array<uint8_t>(16, output_depth);
    auto rshift_data = make_aligned_array<uint8_t>(16, output_depth);
    if (!lshift_data || !rshift_data) return;
    PrepareShiftParams(lshift_data.get(), rshift_data.get(),
                       output_shift, output_depth);

    auto combined_bias = make_aligned_array<int32_t>(16, output_depth);
    if (!combined_bias) return;
    for (int oc = 0; oc < output_depth; oc++) {
        const int8_t* fp = filter_data + oc * input_depth;
        int32_t fsum = 0;
        int remaining = input_depth;
        const int8_t* p = fp;
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e8m4(remaining);
            vint8m4_t v = __riscv_vle8_v_i8m4(p, vl);
            vint16m8_t v16 = __riscv_vsext_vf2_i16m8(v, vl);
            // widening reduction or tree reduce
            vint32m1_t vsum = __riscv_vmv_s_x_i32m1(0, 1);
            vsum = __riscv_vwredsum_vs_i16m8_i32m1(v16, vsum, vl);
            fsum += __riscv_vmv_x_s_i32m1_i32(vsum);
            p += vl;
            remaining -= vl;
        }
        combined_bias[oc] = (bias_data ? bias_data[oc] : 0) + input_offset * fsum;
    }

    // Repack weights: [oc_tile][k][16] with dim-16 = oc within tile
    auto wt_all = make_aligned_array<int8_t>(16, num_oc_tiles * Tk * 16);
    if (!wt_all) return;
    for (int ot = 0; ot < num_oc_tiles; ot++) {
        const int oc_base = ot * 16;
        const int oc_tile = std::min(16, output_depth - oc_base);
        if (oc_tile == 16) {
            for (int k = 0; k < Tk; k++) {
                int8_t* dst = wt_all.get() + (ot * Tk + k) * 16;
                const int8_t* src = filter_data + oc_base * input_depth + k;
                vint8m1_t v = __riscv_vlse8_v_i8m1(src, input_depth, 16);
                __riscv_vse8_v_i8m1(dst, v, oc_tile);
            }
        } else {
            for (int k = 0; k < Tk; k++) {
                int8_t* dst = wt_all.get() + (ot * Tk + k) * 16;
                std::memset(dst, 0, 16);  
                const int8_t* src = filter_data + oc_base * input_depth + k;
                vint8m1_t v = __riscv_vlse8_v_i8m1(src, input_depth, oc_tile);
                __riscv_vse8_v_i8m1(dst, v, oc_tile);  
            }
        }
    }

    int32_t acc_buf[16 * 16] __attribute__((aligned(16)));
    int8_t zeros[16] __attribute__((aligned(16))) = {0};

    mxu_mcfg((uint32_t)(Tk & 0xFF) | 0x100);

    for (int p_base = 0; p_base < num_pixels; p_base += 16) {
        const int p_tile = std::min(16, num_pixels - p_base);

        // Load A once — abuf persists across W reloads and MMA calls
        const int total_a_beats = 16 * num_chunks;
        int beat = 0;
        for (int row = 0; row < 16; row++) {
            for (int chunk = 0; chunk < num_chunks; chunk++) {
                const int8_t* src = (row < p_tile)
                    ? input_data + (p_base + row) * input_depth + chunk * 16
                    : zeros;
                mxu_mload_a_from_mem(src, (beat == total_a_beats - 1));
                beat++;
            }
        }
        // mxu_mfence();

        // Sweep all oc_tiles — only W changes
        for (int ot = 0; ot < num_oc_tiles; ot++) {
            const int oc_base = ot * 16;
            const int oc_tile = std::min(16, output_depth - oc_base);

            const int8_t* wt_base = wt_all.get() + ot * Tk * 16;
            for (int k = 0; k < Tk; k++)
                mxu_mload_w_from_mem(wt_base + k * 16, (k == Tk - 1));
            // mxu_mfence();

            mxu_mzero();
            mxu_mma();
            // mxu_mfence();

            for (int i = 0; i < 64; i++)
                mxu_mstore_to_mem(&acc_buf[i * 4]);
            // mxu_mfence();

            int oc_off = 0;
            size_t oc_rem = oc_tile;
            while (oc_rem > 0) {
                const size_t vl = __riscv_vsetvl_e32m4(oc_rem);
                vint32m4_t bias_vec = __riscv_vle32_v_i32m4(
                    combined_bias.get() + oc_base + oc_off, vl);
                vint32m4_t v_mult = __riscv_vle32_v_i32m4(
                    output_multiplier + oc_base + oc_off, vl);
                vuint32m4_t v_lshift = __riscv_vzext_vf4_u32m4(
                    __riscv_vle8_v_u8m1(
                        lshift_data.get() + oc_base + oc_off, vl), vl);
                vuint32m4_t v_rshift = __riscv_vzext_vf4_u32m4(
                    __riscv_vle8_v_u8m1(
                        rshift_data.get() + oc_base + oc_off, vl), vl);

                for (int p = 0; p < p_tile; p++) {
                    vint32m4_t acc = __riscv_vle32_v_i32m4(
                        &acc_buf[p * 16 + oc_off], vl);
                    acc = __riscv_vadd_vv_i32m4(acc, bias_vec, vl);
                    vint8m1_t out8 = QuantizeResult_m4(
                        acc, v_mult, v_lshift, v_rshift,
                        output_offset, output_min, output_max, vl);
                    __riscv_vse8_v_i8m1(
                        output_data + (p_base + p) * output_depth
                            + oc_base + oc_off,
                        out8, vl);
                }
                oc_off += vl;
                oc_rem -= vl;
            }
        }
    }
}

static void ConvFirstLayer_MXU(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data)
{
    const int input_height  = input_shape.Dims(1);
    const int input_width   = input_shape.Dims(2);
    const int input_depth   = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width  = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width  = output_shape.Dims(2);
    const int output_depth  = output_shape.Dims(3);

    const int stride_h   = params.stride_height;
    const int stride_w   = params.stride_width;
    const int pad_h      = params.padding_values.height;
    const int pad_w      = params.padding_values.width;

    const int32_t input_offset  = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_min    = params.quantized_activation_min;
    const int32_t output_max    = params.quantized_activation_max;

    const int K_raw = filter_height * filter_width * input_depth;
    const int K_padded = ((K_raw + 15) / 16) * 16;
    const int num_k_chunks = K_padded / 16;
    const int input_row_stride = input_width * input_depth;
    const int tap_row_bytes = filter_width * input_depth;

    const int8_t pad_value = static_cast<int8_t>(-input_offset);

    // =========================================================
    // Shift params
    // =========================================================
    auto lshift_data = make_aligned_array<uint8_t>(16, output_depth);
    auto rshift_data = make_aligned_array<uint8_t>(16, output_depth);
    if (!lshift_data || !rshift_data) return;
    PrepareShiftParams(lshift_data.get(), rshift_data.get(),
                       output_shift, output_depth);

    // =========================================================
    // Repack weights + combined bias
    // =========================================================
    int8_t wt_buf[256 * 16] __attribute__((aligned(16)));
    std::memset(wt_buf, 0, K_padded * 16);
    int32_t combined_bias_buf[16] __attribute__((aligned(16)));
    std::memset(combined_bias_buf, 0, sizeof(combined_bias_buf));

    for (int oc = 0; oc < output_depth; oc++) {
        const int8_t* fp = filter_data + oc * K_raw;
        int32_t fsum = 0;
        for (int k = 0; k < K_raw; k++) {
            wt_buf[k * 16 + oc] = fp[k];
            fsum += (int32_t)fp[k];
        }
        combined_bias_buf[oc] = (bias_data ? bias_data[oc] : 0)
                                + input_offset * fsum;
    }

    // =========================================================
    // im2col buffer — two tiles for double buffering
    // =========================================================
    // Layout: im2col[tile][row][K_padded], tile=0 or 1
    int8_t im2col[2][16 * 256] __attribute__((aligned(16)));
    int32_t acc_buf[16 * 16] __attribute__((aligned(16)));

    // Pre-zero the K_padded tail for both buffers
    for (int t = 0; t < 2; t++) {
        if (K_padded > K_raw) {
            for (int r = 0; r < 16; r++)
                std::memset(im2col[t] + r * K_padded + K_raw, 0,
                            K_padded - K_raw);
        }
    }

    // =========================================================
    // Configure MXU and load W once
    // =========================================================
    uint8_t tk_cfg = (K_padded == 256) ? 0 : (uint8_t)K_padded;
    mxu_mcfg((uint32_t)tk_cfg | 0x100);

    for (int k = 0; k < K_padded; k++)
        mxu_mload_w_from_mem(wt_buf + k * 16, (k == K_padded - 1));

    // =========================================================
    // Preload quantization vectors
    // =========================================================
    const size_t oc_vl = __riscv_vsetvl_e32m4(output_depth);
    vint32m4_t v_bias = __riscv_vle32_v_i32m4(combined_bias_buf, oc_vl);
    vint32m4_t v_mult = __riscv_vle32_v_i32m4(output_multiplier, oc_vl);
    vuint32m4_t v_ls = __riscv_vzext_vf4_u32m4(
        __riscv_vle8_v_u8m1(lshift_data.get(), oc_vl), oc_vl);
    vuint32m4_t v_rs = __riscv_vzext_vf4_u32m4(
        __riscv_vle8_v_u8m1(rshift_data.get(), oc_vl), oc_vl);

    // =========================================================
    // Safe region boundaries
    // =========================================================
    const int oy_safe_start = (pad_h + stride_h - 1) / stride_h;
    const int oy_safe_end   = (input_height + pad_h - filter_height + 1) / stride_h;
    const int ox_safe_start = (pad_w + stride_w - 1) / stride_w;
    const int ox_safe_end   = (input_width + pad_w - filter_width + 1) / stride_w;

    // =========================================================
    // Collect all output pixel coordinates into a flat array
    // so we can process them in tiles without complex state
    // =========================================================
    const int total_pixels = output_height * output_width;

    // =========================================================
    // Optimized im2col: specialize for common small depths
    // For depth=3, tap_row_bytes=9: use 32-bit + 16-bit + 8-bit stores
    // =========================================================

    // Helper: fast im2col for safe pixels, no branches in inner loop
    auto fill_im2col_fast = [&](int8_t* col, int iy_base, int ix_base) 
        __attribute__((always_inline)) {
        const int8_t* base = input_data + iy_base * input_row_stride
                             + ix_base * input_depth;
        // Unroll for small filter sizes — compiler should handle this
        for (int fh = 0; fh < filter_height; fh++) {
            const int8_t* src = base + fh * input_row_stride;
            int8_t* dst = col + fh * tap_row_bytes;
            // For tap_row_bytes <= 16, a single memcpy is fine
            // Compiler will inline this for small constant sizes
            __builtin_memcpy(dst, src, tap_row_bytes);
        }
    };

    auto fill_im2col_safe = [&](int8_t* col, int oy, int ox)
        __attribute__((always_inline)) {
        // Use 32-bit fill for speed when possible
        // pad_value repeated 4 times
        const uint32_t pad4 = (uint32_t)(uint8_t)pad_value * 0x01010101u;
        int32_t* col32 = reinterpret_cast<int32_t*>(col);
        // Fill K_raw bytes with pad_value using 32-bit stores
        const int words = K_raw / 4;
        for (int w = 0; w < words; w++)
            col32[w] = (int32_t)pad4;
        for (int b = words * 4; b < K_raw; b++)
            col[b] = pad_value;

        const int iy_base = oy * stride_h - pad_h;
        const int ix_base = ox * stride_w - pad_w;

        for (int fh = 0; fh < filter_height; fh++) {
            const int iy = iy_base + fh;
            if ((unsigned)iy >= (unsigned)input_height) continue;

            const int8_t* in_row = input_data + iy * input_row_stride;
            int8_t* col_row = col + fh * tap_row_bytes;

            // Compute valid fw range to avoid per-element branch
            const int fw_start = std::max(0, -ix_base);
            const int fw_end   = std::min(filter_width, input_width - ix_base);
            if (fw_start < fw_end) {
                __builtin_memcpy(
                    col_row + fw_start * input_depth,
                    in_row + (ix_base + fw_start) * input_depth,
                    (fw_end - fw_start) * input_depth);
            }
        }
    };

    // =========================================================
    // Process tiles with double-buffered im2col
    // Fill tile N+1 while MXU computes tile N
    // =========================================================
    int cur_buf = 0;
    int p_count = 0;
    int p_base = 0;
    int pixel_idx = 0;  // linear index into output

    // Fill first tile
    auto fill_next_pixels = [&](int buf, int& count) {
        count = 0;
        while (count < 16 && pixel_idx < total_pixels) {
            const int oy = pixel_idx / output_width;
            const int ox = pixel_idx % output_width;
            pixel_idx++;

            int8_t* col = im2col[buf] + count * K_padded;

            const int iy_base = oy * stride_h - pad_h;
            const int ix_base = ox * stride_w - pad_w;
            const bool y_safe = (oy >= oy_safe_start && oy < oy_safe_end);
            const bool x_safe = (ox >= ox_safe_start && ox < ox_safe_end);

            if (y_safe && x_safe) {
                fill_im2col_fast(col, iy_base, ix_base);
            } else {
                fill_im2col_safe(col, oy, ox);
            }
            count++;
        }
        // Zero unused rows
        for (int r = count; r < 16; r++)
            std::memset(im2col[buf] + r * K_padded, 0, K_padded);
    };

    auto run_mxu_and_quantize = [&](int buf, int count, int base) {
        int8_t* im = im2col[buf];

        // Load A
        for (int row = 0; row < 16; row++)
            for (int chunk = 0; chunk < num_k_chunks; chunk++)
                mxu_mload_a_from_mem(
                    im + row * K_padded + chunk * 16,
                    (row == 15 && chunk == num_k_chunks - 1));

        mxu_mzero();
        mxu_mma();

        for (int i = 0; i < 64; i++)
            mxu_mstore_to_mem(&acc_buf[i * 4]);

        for (int p = 0; p < count; p++) {
            vint32m4_t acc = __riscv_vle32_v_i32m4(&acc_buf[p * 16], oc_vl);
            acc = __riscv_vadd_vv_i32m4(acc, v_bias, oc_vl);
            vint8m1_t out8 = QuantizeResult_m4(
                acc, v_mult, v_ls, v_rs,
                output_offset, output_min, output_max, oc_vl);
            __riscv_vse8_v_i8m1(
                output_data + (base + p) * output_depth, out8, oc_vl);
        }
    };

    // Pipeline: fill tile 0, then alternate fill/compute
    fill_next_pixels(0, p_count);

    while (p_count > 0) {
        int this_count = p_count;
        int this_base = p_base;
        int this_buf = cur_buf;

        p_base += this_count;
        cur_buf ^= 1;

        // Fill next tile into other buffer WHILE we could overlap
        // (In practice on single-issue in-order core, no true overlap,
        //  but double buffering avoids re-zeroing the same buffer)
        int next_count = 0;
        fill_next_pixels(cur_buf, next_count);

        // Run MXU on current tile
        run_mxu_and_quantize(this_buf, this_count, this_base);

        p_count = next_count;
    }
}

} // anonymous namespace

// =============================================================================
// Public interface
// =============================================================================
void MxuConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data)
{
    const int filter_height = filter_shape.Dims(1);
    const int filter_width  = filter_shape.Dims(2);
    const int input_depth   = input_shape.Dims(3);
    const int output_depth  = output_shape.Dims(3);

    const int K_raw = filter_height * filter_width * input_depth;
    const int K_padded = ((K_raw + 15) / 16) * 16;

    const bool is_1x1     = (filter_height == 1 && filter_width == 1);
    const bool is_stride1 = (params.stride_width == 1 && params.stride_height == 1);
    const bool depth_16   = (input_depth > 0) && (input_depth % 16 == 0);

    const bool mxu_feasible = (K_padded <= 256);
    const int k_util_pct = (K_raw * 100) / K_padded;
    const int n_util_pct = (std::min(output_depth, 16) * 100) / 16;
    const bool mxu_efficient = (k_util_pct > 50) && (n_util_pct >= 50);

    if (is_1x1 && is_stride1 && depth_16) {
        Conv1x1PerChannel_MXU(
            params, output_multiplier, output_shift,
            input_shape, input_data, filter_shape, filter_data,
            bias_shape, bias_data, output_shape, output_data);
    } else if (mxu_feasible && mxu_efficient) {
        ConvFirstLayer_MXU(
            params, output_multiplier, output_shift,
            input_shape, input_data, filter_shape, filter_data,
            bias_shape, bias_data, output_shape, output_data);
    } else {
        coralnpu_v2::opt::litert_micro::ConvPerChannel(
            params, output_multiplier, output_shift,
            input_shape, input_data, filter_shape, filter_data,
            bias_shape, bias_data, output_shape, output_data);
    }
}

TfLiteStatus MxuConvEval(TfLiteContext* context, TfLiteNode* node) {
    const auto& data =
        *(static_cast<const tflite::OpDataConv*>(node->user_data));
    const auto* params =
        reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

    const TfLiteEvalTensor* input =
        tflite::micro::GetEvalInput(context, node, tflite::kConvInputTensor);
    const TfLiteEvalTensor* filter =
        tflite::micro::GetEvalInput(context, node, tflite::kConvWeightsTensor);
    const TfLiteEvalTensor* bias =
        (tflite::micro::GetTensorData<int32_t>(
             tflite::micro::GetEvalInput(
                 context, node, tflite::kConvBiasTensor)) != nullptr)
        ? tflite::micro::GetEvalInput(context, node, tflite::kConvBiasTensor)
        : nullptr;
    TfLiteEvalTensor* output =
        tflite::micro::GetEvalOutput(context, node, tflite::kConvOutputTensor);

    ConvParams op_params;
    op_params.padding_type           = tflite::micro::RuntimePaddingType(
                                           params->padding);
    op_params.padding_values.width   = data.padding.width;
    op_params.padding_values.height  = data.padding.height;
    op_params.stride_width           = params->stride_width;
    op_params.stride_height          = params->stride_height;
    op_params.dilation_width_factor  = params->dilation_width_factor;
    op_params.dilation_height_factor = params->dilation_height_factor;
    op_params.input_offset           = -data.input_zero_point;
    op_params.weights_offset         = -data.filter_zero_point;
    op_params.output_offset          = data.output_zero_point;
    op_params.quantized_activation_min = data.output_activation_min;
    op_params.quantized_activation_max = data.output_activation_max;

    MxuConvPerChannel(
        op_params, data.per_channel_output_multiplier,
        data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        bias ? tflite::micro::GetTensorData<int32_t>(bias) : nullptr,
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));

    return kTfLiteOk;
}

TFLMRegistration Register_MXU_CONV_2D() {
    auto registration = tflite::Register_CONV_2D();
    registration.invoke = MxuConvEval;
    return registration;
}

} // namespace coralnpu_v2::opt::litert_micro