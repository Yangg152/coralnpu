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
//
// Key insight: avoid vmv1r.v + hardcoded register approach.
// Instead, use vle8/vse8 through memory to guarantee data reaches v16,
// which is the vs2 register encoded in the MXU instructions.
// =============================================================================

static inline __attribute__((always_inline))
void mxu_mcfg(uint32_t config_val) {
    register uint32_t a0_val asm("a0") = config_val;
    asm volatile(
        ".word 0x02051057"
        :: "r"(a0_val)
        : "memory"
    );
}

// MLOAD_W: load 16 bytes from memory into v16, then issue MXU_MLOAD_W
// Encoding: funct6=000001, vm=0/1, vs2=v16(10000), vs1=0, funct3=001, vd=0
//   not-last (vm=1): 000001_1_10000_00000_001_00000_1010111 = 0x07001057
//   is-last  (vm=0): 000001_0_10000_00000_001_00000_1010111 = 0x05001057
static void mxu_mload_w_from_mem(const int8_t* src, bool is_last) {
    if (is_last) {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x05001057"
            :: "r"(src), "r"(16)
            : "memory", "v16"
        );
    } else {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x07001057"
            :: "r"(src), "r"(16)
            : "memory", "v16"
        );
    }
}

// MLOAD_A: same pattern
//   not-last (vm=1): 000010_1_10000_00000_001_00000_1010111 = 0x0B001057
//   is-last  (vm=0): 000010_0_10000_00000_001_00000_1010111 = 0x09001057
static void mxu_mload_a_from_mem(const int8_t* src, bool is_last) {
    if (is_last) {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x09001057"
            :: "r"(src), "r"(16)
            : "memory", "v16"
        );
    } else {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x0B001057"
            :: "r"(src), "r"(16)
            : "memory", "v16"
        );
    }
}

// For padded weight loading (valid_oc < 16), go through stack buffer
static void mxu_mload_w_vec(vint8m1_t v_weight, bool is_last) {
    int8_t buf[16] __attribute__((aligned(16)));
    __riscv_vse8_v_i8m1(buf, v_weight, 16);
    mxu_mload_w_from_mem(buf, is_last);
}

static inline __attribute__((always_inline))
void mxu_mzero() {
    asm volatile(".word 0x0E001057" ::: "memory");
}

static inline __attribute__((always_inline))
void mxu_mma() {
    asm volatile(".word 0x12001057" ::: "memory");
}

// MSTORE: funct6=111111, vm=1, vs2=0, vs1=0, funct3=011(OPIVI), vd=10000(v16)
// Result appears in v16 after the instruction
static inline __attribute__((always_inline))
void mxu_mstore_to_mem(int32_t* dst) {
    asm volatile(
        ".word 0xFE001857\n\t"
        "vsetvli zero, %1, e32, m1, ta, ma\n\t"
        "vse32.v v16, (%0)"
        :: "r"(dst), "r"(4)
        : "memory", "v16"
    );
}

static inline __attribute__((always_inline))
void mxu_mfence() {
    asm volatile(".word 0x1A001057" ::: "memory");
}


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
  size_t raw_size = sizeof(T) * nmemb;
  size_t aligned_size = ((raw_size + alignment - 1) / alignment) * alignment;
  void* ptr = aligned_alloc(alignment, aligned_size);
  return aligned_array<T>(reinterpret_cast<T*>(ptr));
}

// Quantize 4 int32 accumulators to int8, write to out_ptr[col_base..col_base+3]
static inline __attribute__((always_inline))
void quantize_and_store_4(
    const int32_t* acc_4,
    int            col_base,
    int            valid_oc,
    const int32_t* offset_comp,
    const int32_t* bias,
    const uint8_t* lshift,
    const int32_t* multiplier,
    const uint8_t* rshift,
    int32_t        output_offset,
    int32_t        output_min,
    int32_t        output_max,
    int8_t*        out_ptr)
{
    int count = std::min(4, valid_oc - col_base);
    if (count <= 0) return;

    size_t vl = __riscv_vsetvl_e32m1(count);

    vint32m1_t acc = __riscv_vle32_v_i32m1(acc_4, vl);

    vint32m1_t v_oc = __riscv_vle32_v_i32m1(offset_comp + col_base, vl);
    vint32m1_t v_bi = __riscv_vle32_v_i32m1(bias + col_base, vl);
    acc = __riscv_vadd_vv_i32m1(acc, v_oc, vl);
    acc = __riscv_vadd_vv_i32m1(acc, v_bi, vl);

    vuint8mf4_t lsh8 = __riscv_vle8_v_u8mf4(lshift + col_base, vl);
    vuint8mf4_t rsh8 = __riscv_vle8_v_u8mf4(rshift + col_base, vl);
    vuint32m1_t v_lsh = __riscv_vzext_vf4_u32m1(lsh8, vl);
    vuint32m1_t v_rsh = __riscv_vzext_vf4_u32m1(rsh8, vl);
    vint32m1_t  v_mul = __riscv_vle32_v_i32m1(multiplier + col_base, vl);

    constexpr uint32_t vxrm = 0;
    acc = __riscv_vsll_vv_i32m1(acc, v_lsh, vl);
    acc = __riscv_vsmul_vv_i32m1(acc, v_mul, vxrm, vl);
    acc = __riscv_vssra_vv_i32m1(acc, v_rsh, vxrm, vl);

    acc = __riscv_vadd_vx_i32m1(acc, output_offset, vl);

    vint16mf2_t acc16 = __riscv_vnclip_wx_i16mf2(acc, 0, vxrm, vl);
    vint8mf4_t  acc8  = __riscv_vnclip_wx_i8mf4(acc16, 0, vxrm, vl);
    acc8 = __riscv_vmax_vx_i8mf4(acc8, (int8_t)output_min, vl);
    acc8 = __riscv_vmin_vx_i8mf4(acc8, (int8_t)output_max, vl);

    __riscv_vse8_v_i8mf4(out_ptr + col_base, acc8, vl);
}

// 1x1 convolution MXU accelerated path
void Conv1x1PerChannel_MXU_Optimized(
    const ConvParams&   params,
    const int32_t*      output_multiplier,
    const int32_t*      output_shift,
    const RuntimeShape& input_shape,
    const int8_t* __restrict__ input_data,
    const RuntimeShape& filter_shape,
    const int8_t* __restrict__ filter_data,
    const RuntimeShape& bias_shape,
    const int32_t* __restrict__ bias_data,
    const RuntimeShape& output_shape,
    int8_t* __restrict__ output_data)
{
    const int input_depth  = input_shape.Dims(3);
    const int output_depth = output_shape.Dims(3);
    const int batches      = input_shape.Dims(0);
    const int num_pixels   = batches * input_shape.Dims(1) * input_shape.Dims(2);
    const int num_chunks   = input_depth / 16;

    const int32_t input_offset  = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_min    = params.quantized_activation_min;
    const int32_t output_max    = params.quantized_activation_max;

    auto lshift_data = make_aligned_array<uint8_t>(16, output_depth);
    auto rshift_data = make_aligned_array<uint8_t>(16, output_depth);
    if (!lshift_data || !rshift_data) return;
    PrepareShiftParams(lshift_data.get(), rshift_data.get(),
                       output_shift, output_depth);

    // Prepare a 16-byte aligned scratch buffer for strided weight gather
    int8_t w_scratch[16] __attribute__((aligned(16)));

    mxu_mcfg(static_cast<uint32_t>(input_depth) | (1u << 8));

    for (int out_c = 0; out_c < output_depth; out_c += 16) {
        const int valid_oc = std::min(16, output_depth - out_c);

        // ============================================================
        // Stage 1: Load weights
        // ============================================================
        for (int k = 0; k < input_depth; ++k) {
            // Gather weight[out_c+0..out_c+15][k] with stride=input_depth
            // into a contiguous 16-byte buffer, then load from memory
            std::memset(w_scratch, 0, 16);
            for (int n = 0; n < valid_oc; ++n) {
                w_scratch[n] = filter_data[(out_c + n) * input_depth + k];
            }
            mxu_mload_w_from_mem(w_scratch, /*is_last=*/(k == input_depth - 1));
        }
        mxu_mfence();

        // ============================================================
        // Precompute per-channel offset compensation and bias
        // ============================================================
        int32_t offset_comp[16] __attribute__((aligned(16)));
        int32_t bias_buf[16]    __attribute__((aligned(16)));
        std::memset(offset_comp, 0, sizeof(offset_comp));
        std::memset(bias_buf, 0, sizeof(bias_buf));

        for (int k = 0; k < input_depth; ++k) {
            for (int n = 0; n < valid_oc; ++n) {
                offset_comp[n] += static_cast<int32_t>(
                    filter_data[(out_c + n) * input_depth + k]);
            }
        }
        for (int n = 0; n < valid_oc; ++n) {
            offset_comp[n] *= input_offset;
            bias_buf[n] = bias_data ? bias_data[out_c + n] : 0;
        }

        // ============================================================
        // Stage 2: Process pixel tiles (16 pixels at a time)
        // ============================================================
        for (int p = 0; p < num_pixels; p += 16) {
            const int valid_p = std::min(16, num_pixels - p);

            mxu_mzero();

            // Load activations: row-major, each row has num_chunks beats
            const int total_a_beats = 16 * num_chunks;
            int beat = 0;
            for (int row = 0; row < 16; ++row) {
                for (int chunk = 0; chunk < num_chunks; ++chunk) {
                    const int k_base = chunk * 16;
                    bool is_last_beat = (beat == total_a_beats - 1);

                    if (row < valid_p) {
                        const int8_t* act_ptr =
                            input_data + (p + row) * input_depth + k_base;
                        mxu_mload_a_from_mem(act_ptr, is_last_beat);
                    } else {
                        // Zero-padded row: use a zero buffer
                        static const int8_t zeros[16]
                            __attribute__((aligned(16))) = {0};
                        mxu_mload_a_from_mem(zeros, is_last_beat);
                    }
                    ++beat;
                }
            }

            mxu_mma();
            mxu_mfence();

            // --------------------------------------------------------
            // Read out 16x16 accumulator and quantize
            // --------------------------------------------------------
            for (int row = 0; row < 16; ++row) {
                int8_t* out_ptr = output_data +
                    (p + row) * output_depth + out_c;

                for (int col_chunk = 0; col_chunk < 4; ++col_chunk) {
                    int32_t acc_4[4] __attribute__((aligned(16)));
                    mxu_mstore_to_mem(acc_4);

                    int col_base = col_chunk * 4;
                    if (col_base >= valid_oc || row >= valid_p) {
                        continue;
                    }

                    quantize_and_store_4(
                        acc_4, col_base, valid_oc,
                        offset_comp, bias_buf,
                        lshift_data.get() + out_c,
                        output_multiplier + out_c,
                        rshift_data.get() + out_c,
                        output_offset, output_min, output_max,
                        out_ptr);
                }
            }
        }
    }
}

} // anonymous namespace

// =============================================================================
// Public interface
// =============================================================================

void MxuConvPerChannel(
    const tflite::ConvParams&   params,
    const int32_t*              output_multiplier,
    const int32_t*              output_shift,
    const tflite::RuntimeShape& input_shape,
    const int8_t*               input_data,
    const tflite::RuntimeShape& filter_shape,
    const int8_t*               filter_data,
    const tflite::RuntimeShape& bias_shape,
    const int32_t*              bias_data,
    const tflite::RuntimeShape& output_shape,
    int8_t*                     output_data)
{
    const bool is_1x1     = (filter_shape.Dims(1) == 1 &&
                              filter_shape.Dims(2) == 1);
    const bool is_stride1 = (params.stride_width  == 1 &&
                              params.stride_height == 1);
    const int  input_depth = input_shape.Dims(3);
    const bool depth_ok   = (input_depth > 0) && (input_depth % 16 == 0);

    if (is_1x1 && is_stride1 && depth_ok) {
        Conv1x1PerChannel_MXU_Optimized(
            params, output_multiplier, output_shift,
            input_shape,  input_data,
            filter_shape, filter_data,
            bias_shape,   bias_data,
            output_shape, output_data);
    } else {
        coralnpu_v2::opt::litert_micro::ConvPerChannel(
            params, output_multiplier, output_shift,
            input_shape,  input_data,
            filter_shape, filter_data,
            bias_shape,   bias_data,
            output_shape, output_data);
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

    tflite::ConvParams op_params;
    op_params.padding_type =
        tflite::micro::RuntimePaddingType(params->padding);
    op_params.padding_values.width     = data.padding.width;
    op_params.padding_values.height    = data.padding.height;
    op_params.stride_width             = params->stride_width;
    op_params.stride_height            = params->stride_height;
    op_params.dilation_width_factor    = params->dilation_width_factor;
    op_params.dilation_height_factor   = params->dilation_height_factor;
    op_params.input_offset             = -data.input_zero_point;
    op_params.weights_offset           = -data.filter_zero_point;
    op_params.output_offset            = data.output_zero_point;
    op_params.quantized_activation_min = data.output_activation_min;
    op_params.quantized_activation_max = data.output_activation_max;

    MxuConvPerChannel(
        op_params,
        data.per_channel_output_multiplier,
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

}  // namespace coralnpu_v2::opt::litert_micros