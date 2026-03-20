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

// -----------------------------------------------------------------------------
// 1x1 pointwise convolution via MXU
// A-matrix = pixels (loaded once per pixel tile, stays in abuf)
// W-matrix = filter weights (reloaded per oc_tile)
// Works for both large and small spatial sizes
// -----------------------------------------------------------------------------
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
        int32_t fsum = 0;
        const int8_t* fp = filter_data + oc * input_depth;
        for (int ic = 0; ic < input_depth; ic++) fsum += fp[ic];
        combined_bias[oc] = (bias_data ? bias_data[oc] : 0)
                          + input_offset * fsum;
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
                for (int j = 0; j < 16; j++)
                    dst[j] = filter_data[(oc_base + j) * input_depth + k];
            }
        } else {
            for (int k = 0; k < Tk; k++) {
                int8_t* dst = wt_all.get() + (ot * Tk + k) * 16;
                std::memset(dst, 0, 16);
                for (int j = 0; j < oc_tile; j++)
                    dst[j] = filter_data[(oc_base + j) * input_depth + k];
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

    const bool is_1x1    = (filter_height == 1 && filter_width == 1);
    const bool is_stride1 = (params.stride_width == 1 && params.stride_height == 1);
    const bool depth_16  = (input_depth > 0) && (input_depth % 16 == 0);

    if (is_1x1 && is_stride1 && depth_16) {
        Conv1x1PerChannel_MXU(
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