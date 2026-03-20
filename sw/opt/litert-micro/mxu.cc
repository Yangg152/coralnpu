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
    asm volatile(
        ".word 0x02051057"
        :: "r"(a0_val)
        : "memory"
    );
}

static void mxu_mload_a_from_mem(const int8_t* src, bool is_last) {
    if (is_last) {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            "vadd.vi v16, v16, 0\n\t"
            ".word 0x09001057"
            :: "r"(src), "r"(16)
            : "memory", "v16"
        );
    } else {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            "vadd.vi v16, v16, 0\n\t"
            ".word 0x0B001057"
            :: "r"(src), "r"(16)
            : "memory", "v16"
        );
    }
}

static void mxu_mload_w_from_mem(const int8_t* src, bool is_last) {
    if (is_last) {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            "vadd.vi v16, v16, 0\n\t"
            ".word 0x05001057"
            :: "r"(src), "r"(16)
            : "memory", "v16"
        );
    } else {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            "vadd.vi v16, v16, 0\n\t"
            ".word 0x07001057"
            :: "r"(src), "r"(16)
            : "memory", "v16"
        );
    }
}

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
    const int Tk           = input_depth;
    const int num_chunks   = Tk / 16;

    // Step 1: mcfg — 直接用 li 把值放到 a0
    {
        uint32_t cfg = (uint32_t)(Tk & 0xFF) | 0x100;
        asm volatile(
            "mv a0, %0\n\t"
            ".word 0x02051057"
            :: "r"(cfg)
            : "a0", "memory"
        );
    }

    // Step 2: mload_w — Tk 次
    {
        int8_t ones[16] __attribute__((aligned(16)));
        for (int i = 0; i < 16; i++) ones[i] = 1;
        for (int k = 0; k < Tk; k++) {
            mxu_mload_w_from_mem(ones, (k == Tk - 1));
        }
    }
    mxu_mfence();

    // Step 3: mzero
    mxu_mzero();

    // Step 4: mload_a — 16 rows × num_chunks
    {
        int8_t ones[16] __attribute__((aligned(16)));
        for (int i = 0; i < 16; i++) ones[i] = 1;
        int total_beats = 16 * num_chunks;
        for (int beat = 0; beat < total_beats; beat++) {
            mxu_mload_a_from_mem(ones, (beat == total_beats - 1));
        }
    }

    // Step 5: mma + fence
    mxu_mma();
    mxu_mfence();

    // Step 6: mstore — 64 次
    int32_t acc_buf[256] __attribute__((aligned(16)));
    std::memset(acc_buf, 0xAB, sizeof(acc_buf));
    for (int i = 0; i < 64; i++) {
        mxu_mstore_to_mem(&acc_buf[i * 4]);
    }
    mxu_mfence();

    // Step 7: 直接把 raw int32 截断输出，不 clamp
    // 这样如果 acc=16，output=16；如果 acc=0xABABABAB，output=0xAB=-85
    int total = num_pixels * output_depth;
    for (int i = 0; i < total; i++) {
        output_data[i] = static_cast<int8_t>(acc_buf[i % 256] & 0xFF);
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

}  // namespace coralnpu_v2::opt::litert_micro