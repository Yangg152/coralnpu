// Copyright 2025 Google LLC
#include "sw/opt/litert-micro/softmax.h"

#include <algorithm>
#include <limits>
#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

#include "fixedpoint/fixedpoint.h"

namespace tflite {
extern TFLMRegistration Register_SOFTMAX();
}

namespace coralnpu_v2::opt::litert_micro {

using tflite::SoftmaxParams;
using tflite::RuntimeShape;
using tflite::micro::GetEvalInput;
using tflite::micro::GetEvalOutput;
using tflite::micro::GetTensorData;
using tflite::micro::GetTensorShape;

using tflite::MultiplyByQuantizedMultiplierGreaterThanOne;
using tflite::GetReciprocal;

static const int kScaledDiffIntegerBits = 5;
static const int kAccumulationIntegerBits = 12;
using FixedPointScaledDiff = gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
using FixedPointAccum = gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;
using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;

void SoftmaxQuantized(
    const SoftmaxParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& output_shape, int8_t* output_data) {

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size = MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth = MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  const int32_t input_beta_multiplier = params.input_multiplier;
  const int32_t input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;

  int total_elements = outer_size * depth;

  // =========================================================
  // 小尺寸张量优化路径：免去建立全局 LUT 的过量开销
  // =========================================================
  if (total_elements < 256) {
    for (int i = 0; i < outer_size; ++i) {
      const int8_t* in_ptr = input_data + i * depth;
      int8_t* out_ptr = output_data + i * depth;

      // Find Max (Vectorized)
      int8_t max_in_row = std::numeric_limits<int8_t>::min();
      size_t n = depth;
      const int8_t* ptr = in_ptr;
      vint8m1_t v_max_val = __riscv_vmv_v_x_i8m1(max_in_row, 1);
      while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m8(n);
        vint8m8_t v_data = __riscv_vle8_v_i8m8(ptr, vl);
        v_max_val = __riscv_vredmax_vs_i8m8_i8m1(v_data, v_max_val, vl);
        ptr += vl;
        n -= vl;
      }
      max_in_row = __riscv_vmv_x_s_i8m1_i8(v_max_val);

      FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
      for (int c = 0; c < depth; ++c) {
        int32_t input_diff = static_cast<int32_t>(in_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32_t input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 = FixedPointScaledDiff::FromRaw(input_diff_rescaled);
          sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                          gemmlowp::exp_on_negative_values(scaled_diff_f8));
        }
      }

      int num_bits_over_unit;
      FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
          sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));
      const int exponent = num_bits_over_unit + 31 - 8;

      for (int c = 0; c < depth; ++c) {
        int32_t input_diff = static_cast<int32_t>(in_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32_t input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 = FixedPointScaledDiff::FromRaw(input_diff_rescaled);

          FixedPoint0 exp_in_0 = gemmlowp::exp_on_negative_values(scaled_diff_f8);
          int32_t unsat_output = gemmlowp::RoundingDivideByPOT((shifted_scale * exp_in_0).raw(), exponent);
          const int32_t shifted_output = unsat_output + static_cast<int32_t>(-128);
          out_ptr[c] = static_cast<int8_t>(std::max(std::min(shifted_output, static_cast<int32_t>(127)), static_cast<int32_t>(-128)));
        } else {
          out_ptr[c] = -128;
        }
      }
    }
    return;
  }

  // =========================================================
  // 大尺寸张量优化路径：全局 LUT 建立与 RVV 向量寻址写入
  // =========================================================
  int32_t exp_lut[256];
  int32_t exp_sum_lut[256];
  for (int i = 0; i < 256; ++i) {
    int32_t input_diff = -i;
    if (input_diff >= diff_min) {
      const int32_t input_diff_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_diff, input_beta_multiplier, input_beta_left_shift);
      const FixedPointScaledDiff scaled_diff_f8 =
          FixedPointScaledDiff::FromRaw(input_diff_rescaled);
      FixedPoint0 exp_val = gemmlowp::exp_on_negative_values(scaled_diff_f8);
      exp_lut[i] = exp_val.raw();
      exp_sum_lut[i] = gemmlowp::Rescale<kAccumulationIntegerBits>(exp_val).raw();
    } else {
      exp_lut[i] = 0;
      exp_sum_lut[i] = 0;
    }
  }

  for (int i = 0; i < outer_size; ++i) {
    const int8_t* in_ptr = input_data + i * depth;
    int8_t* out_ptr = output_data + i * depth;

    // 1. Find Max (Vectorized e8m8)
    int8_t max_in_row = std::numeric_limits<int8_t>::min();
    size_t n = depth;
    const int8_t* ptr = in_ptr;
    vint8m1_t v_max_val = __riscv_vmv_v_x_i8m1(max_in_row, 1);
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m8(n); 
      vint8m8_t v_data = __riscv_vle8_v_i8m8(ptr, vl);
      v_max_val = __riscv_vredmax_vs_i8m8_i8m1(v_data, v_max_val, vl);
      ptr += vl;
      n -= vl;
    }
    max_in_row = __riscv_vmv_x_s_i8m1_i8(v_max_val);

    // 2. Compute Exp Sum (Scalar lookup is extremely fast here, bypasses vluxei8 byte offset danger for 32bit)
    int32_t sum_raw = 0;
    for (int c = 0; c < depth; ++c) {
      uint8_t diff = static_cast<uint8_t>(max_in_row) - static_cast<uint8_t>(in_ptr[c]);
      sum_raw += exp_sum_lut[diff];
    }
    FixedPointAccum sum_of_exps = FixedPointAccum::FromRaw(sum_raw);

    // 3. Reciprocal computation
    int num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));
    const int exponent = num_bits_over_unit + 31 - 8; 

    // 4. Compute exact Row LUT
    int8_t row_out_lut[256];
    for (int diff = 0; diff < 256; ++diff) {
      if (-diff >= diff_min) {
        FixedPoint0 exp_in_0 = FixedPoint0::FromRaw(exp_lut[diff]);
        int32_t unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(), exponent);
        int32_t shifted_output = unsat_output + static_cast<int32_t>(-128);
        row_out_lut[diff] = static_cast<int8_t>(std::max(
            std::min(shifted_output, static_cast<int32_t>(127)),
            static_cast<int32_t>(-128)));
      } else {
        row_out_lut[diff] = -128;
      }
    }

    // 5. Apply Row LUT output mapping via RVV vluxei8
    // Safe to use vluxei8 here because row_out_lut is int8_t (byte offset = element offset)
    n = depth;
    ptr = in_ptr;
    int8_t* out = out_ptr;
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m8(n);
      vint8m8_t v_data = __riscv_vle8_v_i8m8(ptr, vl);
      
      // Compute diff = max_in_row - input (vectorized)
      vuint8m8_t v_diff = __riscv_vrsub_vx_u8m8(
          __riscv_vreinterpret_v_i8m8_u8m8(v_data), static_cast<uint8_t>(max_in_row), vl);
      
      // Look up indexed outputs and store
      vint8m8_t v_out = __riscv_vluxei8_v_i8m8(row_out_lut, v_diff, vl);
      __riscv_vse8_v_i8m8(out, v_out, vl);
      
      ptr += vl;
      out += vl;
      n -= vl;
    }
  }
}

// =========================================================
// TFLite Interface
// =========================================================

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const SoftmaxParams* params = static_cast<const SoftmaxParams*>(node->builtin_data);

  const TfLiteEvalTensor* input = GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = GetEvalOutput(context, node, 0);

  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
      SoftmaxQuantized(
          *params,
          GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output));
      return kTfLiteOk;
  }

  tflite::reference_ops::Softmax(
      *params,
      GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(output), GetTensorData<float>(output));

  return kTfLiteOk;
}

TFLMRegistration Register_SOFTMAX() {
  TFLMRegistration r = tflite::Register_SOFTMAX();
  r.invoke = SoftmaxEval;
  return r;
}

}  // namespace coralnpu_v2::opt::litert_micro
