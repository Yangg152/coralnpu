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

static inline __attribute__((always_inline))
void mxu_mload_a_from_mem(const int8_t* src, bool is_last) {
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

static inline __attribute__((always_inline))
void mxu_mload_w_from_mem(const int8_t* src, bool is_last) {
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
        ".word 0x16001857\n\t"
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

// -----------------------------------------------------------------------------
// MXU 1x1 convolution with correct quantization math
//
// TFLite quantized conv computes:
//   acc[p][oc] = sum_ic( (input[p][ic] + input_offset) * filter[oc][ic] ) + bias[oc]
//
// We decompose this as:
//   acc[p][oc] = sum_ic( input[p][ic] * filter[oc][ic] )          ... (A) MXU does this
//              + input_offset * sum_ic( filter[oc][ic] )           ... (B) precomputed
//              + bias[oc]                                          ... (C) from bias_data
//
// MXU computes (A) using raw int8 values (no offset applied to activations).
// Term (B) + (C) is a per-channel constant added after MXU readout.
//
// This avoids the int8 overflow problem of adding input_offset to activations.
// -----------------------------------------------------------------------------
void Conv1x1PerChannel_MXU(
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

    const int32_t input_offset  = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int8_t  output_min    = (int8_t)params.quantized_activation_min;
    const int8_t  output_max    = (int8_t)params.quantized_activation_max;

    const int Tk = input_depth;
    const int num_chunks = Tk / 16;

    // Prepare shift params for PostprocessAcc
    auto lshift_data = make_aligned_array<uint8_t>(16, output_depth);
    auto rshift_data = make_aligned_array<uint8_t>(16, output_depth);
    if (!lshift_data || !rshift_data) return;
    PrepareShiftParams(lshift_data.get(), rshift_data.get(),
                       output_shift, output_depth);

    // Precompute per-channel bias correction:
    //   combined_bias[oc] = bias[oc] + input_offset * sum_ic(filter[oc][ic])
    auto combined_bias = make_aligned_array<int32_t>(16, output_depth);
    if (!combined_bias) return;
    for (int oc = 0; oc < output_depth; oc++) {
        int32_t filter_sum = 0;
        const int8_t* f_ptr = filter_data + oc * input_depth;
        for (int ic = 0; ic < input_depth; ic++) {
            filter_sum += f_ptr[ic];
        }
        int32_t bias_val = bias_data ? bias_data[oc] : 0;
        combined_bias[oc] = bias_val + input_offset * filter_sum;
    }

    // Weight tile buffer: Tk rows × 16 cols
    auto wt_tile = make_aligned_array<int8_t>(16, Tk * 16);
    if (!wt_tile) return;

    // Accumulator readout buffer: 16 rows × 16 cols int32
    int32_t acc_buf[16 * 16] __attribute__((aligned(16)));

    // Temporary output buffer for PostprocessAcc when output_depth > 16
    // PostprocessAcc writes with stride = out_d, but we need stride = output_depth
    auto tmp_out = make_aligned_array<int8_t>(16, 16 * 16);
    if (!tmp_out) return;

    // Configure MXU
    {
        uint32_t cfg = (uint32_t)(Tk & 0xFF) | 0x100;
        mxu_mcfg(cfg);
    }

    // Tile over output channels in groups of 16
    for (int oc_base = 0; oc_base < output_depth; oc_base += 16) {
        const int oc_tile = std::min(16, output_depth - oc_base);

        // ---- Pack weight tile ----
        // MXU weight row k = { filter[oc_base+j][k] } for j=0..15
        for (int k = 0; k < Tk; k++) {
            int8_t* dst_row = wt_tile.get() + k * 16;
            for (int j = 0; j < 16; j++) {
                if (j < oc_tile) {
                    dst_row[j] = filter_data[(oc_base + j) * input_depth + k];
                } else {
                    dst_row[j] = 0;
                }
            }
        }

        // ---- Load weights into MXU ----
        for (int k = 0; k < Tk; k++) {
            mxu_mload_w_from_mem(wt_tile.get() + k * 16, (k == Tk - 1));
        }
        mxu_mfence();

        // Tile over pixels in groups of 16
        for (int p_base = 0; p_base < num_pixels; p_base += 16) {
            const int p_tile = std::min(16, num_pixels - p_base);

            // ---- Zero accumulators ----
            mxu_mzero();

            // ---- Load activations (raw, no offset) ----
            // Beat order: row0-chunk0, row1-chunk0, ..., row15-chunk0,
            //             row0-chunk1, ..., row15-chunk(num_chunks-1)
            {
                int total_beats = 16 * num_chunks;
                int beat = 0;
                for (int chunk = 0; chunk < num_chunks; chunk++) {
                    for (int row = 0; row < 16; row++) {
                        if (row < p_tile) {
                            const int8_t* src =
                                input_data + (p_base + row) * input_depth
                                + chunk * 16;
                            mxu_mload_a_from_mem(src, (beat == total_beats - 1));
                        } else {
                            // Zero-pad for unused rows
                            int8_t zeros[16] __attribute__((aligned(16))) = {0};
                            mxu_mload_a_from_mem(zeros, (beat == total_beats - 1));
                        }
                        beat++;
                    }
                }
            }

            // ---- Compute ----
            mxu_mma();
            mxu_mfence();

            // ---- Read out accumulators ----
            // 64 MSTORE calls, each returns 4 int32
            // Layout: call i → acc_buf[i*4 .. i*4+3]
            // Row r, col group g (g=0..3): call index = r*4 + g
            // acc_buf[r*16 + g*4 + 0..3]
            for (int i = 0; i < 64; i++) {
                mxu_mstore_to_mem(&acc_buf[i * 4]);
            }
            mxu_mfence();

            // ---- Postprocess ----
            // acc_buf[row * 16 + col] contains raw MXU result (term A)
            // We need to add combined_bias (terms B + C) before requantize
            //
            // PostprocessAcc expects accs WITHOUT bias (it adds bias internally),
            // so we pass combined_bias as the bias pointer.
            //
            // When output_depth == 16, we can write directly.
            // Otherwise, write to tmp_out then scatter.
            if (output_depth == 16) {
                // Direct write: PostprocessAcc stride matches output stride
                PostprocessAcc(
                    acc_buf,
                    combined_bias.get() + oc_base,
                    lshift_data.get() + oc_base,
                    output_multiplier + oc_base,
                    rshift_data.get() + oc_base,
                    output_offset,
                    output_min,
                    output_max,
                    output_data + p_base * output_depth + oc_base,
                    p_tile,
                    oc_tile);
            } else {
                // Write to temp buffer, then scatter to output
                PostprocessAcc(
                    acc_buf,
                    combined_bias.get() + oc_base,
                    lshift_data.get() + oc_base,
                    output_multiplier + oc_base,
                    rshift_data.get() + oc_base,
                    output_offset,
                    output_min,
                    output_max,
                    tmp_out.get(),
                    p_tile,
                    oc_tile);

                // Scatter: tmp_out[p * oc_tile + j] → output[p_base+p][oc_base+j]
                for (int p = 0; p < p_tile; p++) {
                    std::memcpy(
                        output_data + (p_base + p) * output_depth + oc_base,
                        tmp_out.get() + p * oc_tile,
                        oc_tile);
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
        Conv1x1PerChannel_MXU(
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