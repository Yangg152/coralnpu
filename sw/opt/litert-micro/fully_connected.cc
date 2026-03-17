// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sw/opt/litert-micro/fully_connected.h"

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/rvv_opt.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace coralnpu_v2::opt::litert_micro {

using tflite::FullyConnectedParams;
using tflite::NumInputs;
using tflite::OpDataFullyConnected;
using tflite::RuntimeShape;
using tflite::kFullyConnectedBiasTensor;
using tflite::kFullyConnectedInputTensor;
using tflite::kFullyConnectedWeightsTensor;
using tflite::kFullyConnectedOutputTensor;
using tflite::micro::GetEvalInput;
using tflite::micro::GetEvalOutput;
using tflite::micro::GetOptionalTensorData;
using tflite::micro::GetTensorData;
using tflite::micro::GetTensorShape;

namespace {

// TODO(davidgao): move away and share these with other kernels
struct AlignedFree {
  void operator()(void* ptr) const { std::free(ptr); }
};

template <typename T>
using aligned_array = std::unique_ptr<T[], AlignedFree>;

template <typename T>
aligned_array<T> make_aligned_array(size_t alignment, size_t nmemb) {
  return aligned_array<T>(
      reinterpret_cast<T*>(aligned_alloc(alignment, sizeof(T) * nmemb)));
}

// ---------------------------------------------------------------------------
// Core RVV-vectorised kernel
// ---------------------------------------------------------------------------

inline int32_t DotProductWithOffset(const int8_t* __restrict__ filter_row,
                                    const int8_t* __restrict__ input_row,
                                    int depth, int32_t input_offset) {
  int32_t total = 0;
  int d = 0;
  while (d < depth) {
    const size_t rem = static_cast<size_t>(depth - d);
    const size_t vl  = __riscv_vsetvl_e16m4(rem);

    const vint8m2_t  f_i8  = __riscv_vle8_v_i8m2(filter_row + d, vl);
    const vint8m2_t  x_i8  = __riscv_vle8_v_i8m2(input_row  + d, vl);
    const vint16m4_t f_i16 = __riscv_vsext_vf2_i16m4(f_i8, vl);
          vint16m4_t x_i16 = __riscv_vsext_vf2_i16m4(x_i8, vl);

    x_i16 = __riscv_vadd_vx_i16m4(x_i16, static_cast<int16_t>(input_offset),
                                   vl);

    vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, 1);
    const vint32m8_t prod = __riscv_vwmul_vv_i32m8(f_i16, x_i16, vl);
    const vint32m1_t red  = __riscv_vredsum_vs_i32m8_i32m1(prod, zero, vl);
    total += __riscv_vmv_x_s_i32m1_i32(red);

    d += static_cast<int>(vl);
  }
  return total;
}

void AccumulateAllBatches(const int8_t* __restrict__ filter_row,
                          const int8_t* __restrict__ input_data,
                          int batches, int accum_depth, int output_depth,
                          int32_t input_offset, int32_t* __restrict__ accs) {
  for (int b = 0; b < batches; ++b) {
    const int8_t* input_row = input_data + b * accum_depth;
    accs[b * output_depth] = DotProductWithOffset(filter_row, input_row,
                                                  accum_depth, input_offset);
  }
}

void FullyConnectedPerChannelRVV(const FullyConnectedParams& params,
                                 const int32_t* output_multiplier,
                                 const uint8_t* shift_left,
                                 const uint8_t* shift_right,
                                 const RuntimeShape& input_shape,
                                 const int8_t* input_data,
                                 const RuntimeShape& filter_shape,
                                 const int8_t* filter_data,
                                 const RuntimeShape& bias_shape,
                                 const int32_t* bias_data,
                                 const RuntimeShape& output_shape,
                                 int8_t* output_data,
                                 int32_t* accs_buf) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int8_t  output_activation_min =
      static_cast<int8_t>(params.quantized_activation_min);
  const int8_t  output_activation_max =
      static_cast<int8_t>(params.quantized_activation_max);

  // fix: declare filter_dim_count here where it is actually used
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches      = tflite::FlatSizeSkipDim(output_shape,
                                                    output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  const int accum_depth  = filter_shape.Dims(filter_dim_count - 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  for (int out_c = 0; out_c < output_depth; ++out_c) {
    const int8_t* filter_row = filter_data + out_c * accum_depth;
    AccumulateAllBatches(filter_row, input_data, batches, accum_depth,
                         output_depth, input_offset,
                         accs_buf + out_c);
  }

  for (int b = 0; b < batches; ++b) {
    PostprocessAcc(
        accs_buf + b * output_depth,
        bias_data,
        shift_left,
        output_multiplier,
        shift_right,
        output_offset,
        output_activation_min,
        output_activation_max,
        output_data + b * output_depth,
        /*out_w=*/1,
        /*out_d=*/output_depth);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void FullyConnectedPerChannel(const FullyConnectedParams& params,
                              const int32_t* output_multiplier,
                              const int32_t* output_shift,
                              const RuntimeShape& input_shape,
                              const int8_t* input_data,
                              const RuntimeShape& filter_shape,
                              const int8_t* filter_data,
                              const RuntimeShape& bias_shape,
                              const int32_t* bias_data,
                              const RuntimeShape& output_shape,
                              int8_t* output_data,
                              int32_t* accs_buf) {
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  const int accum_depth  = filter_shape.Dims(filter_dim_count - 1);

  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  // ---------------------------------------------------------------------------
  // Small-shape fast path: skip all RVV setup overhead (aligned_alloc x4,
  // Memcpy, PrepareShiftParams) and fall back directly to the reference scalar
  // implementation.
  //
  // Threshold: accum_depth * output_depth < 512.
  // This covers the observed negative-speedup cases:
  //   (1, 4)->4   :   16 ops  (0.44x without this guard)
  //   (1,13)->7   :   91 ops  (0.90x without this guard)
  //   (1,16)->8   :  128 ops  (1.04x — marginal, scalar is safer)
  //   (1,33)->16  :  528 ops  — above threshold, stays on RVV path
  // Adjust the threshold up/down based on profiling if needed.
  // ---------------------------------------------------------------------------
  if (accum_depth * output_depth < 128) {
    tflite::reference_integer_ops::FullyConnectedPerChannel(
        params, output_multiplier, reinterpret_cast<const int*>(output_shift),
        input_shape,  input_data,
        filter_shape, filter_data,
        bias_shape,   bias_data,
        output_shape, output_data);
    return;
  }
  // fix: removed unused filter_dim_count; output_depth derived from output_shape
  // Copy filter and bias into DTCM-aligned buffers (mirrors depthwise_conv).
  auto filter_copy = make_aligned_array<int8_t>(16, filter_shape.FlatSize());
  // TODO(davidgao): if allocation fails, don't copy, use orig
  TFLITE_DCHECK_NE(filter_copy, nullptr);
  Memcpy(filter_copy.get(), filter_data,
         sizeof(int8_t) * filter_shape.FlatSize());

  aligned_array<int32_t> bias_copy;
  if (bias_data) {
    bias_copy = make_aligned_array<int32_t>(16, output_depth);
    // TODO(davidgao): if allocation fails, don't copy, use orig
    TFLITE_DCHECK_NE(bias_copy, nullptr);
    Memcpy(bias_copy.get(), bias_data, sizeof(int32_t) * output_depth);
  }

  auto shift_left  = make_aligned_array<uint8_t>(16, output_depth);
  auto shift_right = make_aligned_array<uint8_t>(16, output_depth);
  TFLITE_DCHECK_NE(shift_left,  nullptr);
  TFLITE_DCHECK_NE(shift_right, nullptr);
  PrepareShiftParams(shift_left.get(), shift_right.get(),
                     output_shift, output_depth);

  FullyConnectedPerChannelRVV(
      params, output_multiplier, shift_left.get(), shift_right.get(),
      input_shape,  input_data,
      filter_shape, filter_copy.get(),
      bias_shape,   bias_copy.get(),
      output_shape, output_data,
      accs_buf);
}

// ---------------------------------------------------------------------------
// TFLite Micro kernel registration
// ---------------------------------------------------------------------------

TfLiteStatus FullyConnectedEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data    != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  // fix: removed unused 'params'; FullyConnectedParamsQuantized only needs data
  const auto& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));

  TfLiteEvalTensor* output =
      GetEvalOutput(context, node, kFullyConnectedOutputTensor);
  const TfLiteEvalTensor* input =
      GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? GetEvalInput(context, node, kFullyConnectedBiasTensor)
          : nullptr;

  // Allocate accs buffer for this eval call.
  const int output_dim_count = GetTensorShape(output).DimensionsCount();
  const int batches      = tflite::FlatSizeSkipDim(GetTensorShape(output),
                                                    output_dim_count - 1);
  const int output_depth = GetTensorShape(output).Dims(output_dim_count - 1);
  auto accs_buf_arr = make_aligned_array<int32_t>(16, batches * output_depth);
  TFLITE_DCHECK_NE(accs_buf_arr, nullptr);
  int32_t* accs_buf = accs_buf_arr.get();

  switch (input->type) {
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt8: {
          FullyConnectedPerChannel(
              tflite::FullyConnectedParamsQuantized(data),
              data.per_channel_output_multiplier,
              data.per_channel_output_shift,
              GetTensorShape(input),  GetTensorData<int8_t>(input),
              GetTensorShape(filter), GetTensorData<int8_t>(filter),
              GetTensorShape(bias),   GetOptionalTensorData<int32_t>(bias),
              GetTensorShape(output), GetTensorData<int8_t>(output),
              accs_buf);
          break;
        }
        default:
          MicroPrintf(
              "Filter type %s (%d) for input type %s not supported.",
              TfLiteTypeGetName(filter->type), filter->type,
              TfLiteTypeGetName(input->type));
          return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TFLMRegistration Register_FULLY_CONNECTED() {
  auto registration = tflite::Register_FULLY_CONNECTED();
  registration.invoke = FullyConnectedEval;
  return registration;
}

}  // namespace coralnpu_v2::opt::litert_micro
