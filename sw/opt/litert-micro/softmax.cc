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

// 引入 gemmlowp 的定点数工具，用于处理 Exp 的高精度部分
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

// 【修复点 1】引入缺失的 TFLite 数学函数
using tflite::MultiplyByQuantizedMultiplierGreaterThanOne;
using tflite::GetReciprocal;

// TFLite Softmax 需要的定点数类型定义
static const int kScaledDiffIntegerBits = 5;
static const int kAccumulationIntegerBits = 12;
using FixedPointScaledDiff = gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
using FixedPointAccum = gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;
using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;

// =========================================================
// 核心 RVV 实现
// =========================================================

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

  for (int i = 0; i < outer_size; ++i) {
    const int8_t* in_ptr = input_data + i * depth;
    int8_t* out_ptr = output_data + i * depth;

    // ---------------------------------------------------------
    // 1. Find Max (Vectorized)
    // ---------------------------------------------------------
    int8_t max_in_row = std::numeric_limits<int8_t>::min();
    size_t n = depth;
    const int8_t* ptr = in_ptr;
    
    // 初始化 max 向量
    vint8m1_t v_max_val = __riscv_vmv_v_x_i8m1(max_in_row, 1);

    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m2(n);
      vint8m2_t v_data = __riscv_vle8_v_i8m2(ptr, vl);
      // REDMAX: 规约求最大值
      v_max_val = __riscv_vredmax_vs_i8m2_i8m1(v_data, v_max_val, vl);
      ptr += vl;
      n -= vl;
    }
    max_in_row = __riscv_vmv_x_s_i8m1_i8(v_max_val);

    // ---------------------------------------------------------
    // 2. Compute Exp and Sum (Hybrid)
    // ---------------------------------------------------------
    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    
    for (int c = 0; c < depth; ++c) {
      int32_t input_diff = static_cast<int32_t>(in_ptr[c]) - max_in_row;
      if (input_diff >= diff_min) {
        // 使用 tflite::MultiplyByQuantizedMultiplierGreaterThanOne
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        
        // 【修复点 2】明确使用 gemmlowp::exp_on_negative_values
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        gemmlowp::exp_on_negative_values(scaled_diff_f8));
      }
    }

    // ---------------------------------------------------------
    // 3. Compute Reciprocal (Scalar)
    // ---------------------------------------------------------
    int num_bits_over_unit;
    // 使用 tflite::GetReciprocal
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));
    const int exponent = num_bits_over_unit + 31 - 8; // sizeof(int8_t)*8 = 8

    // ---------------------------------------------------------
    // 4. Output Scaling (Vectorized Attempt)
    // ---------------------------------------------------------
    for (int c = 0; c < depth; ++c) {
      int32_t input_diff = static_cast<int32_t>(in_ptr[c]) - max_in_row;
      
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);

        FixedPoint0 exp_in_0 = gemmlowp::exp_on_negative_values(scaled_diff_f8);
        
        // Mul + RoundingDivideByPOT
        int32_t unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(), exponent);

        // Add offset (min int8) and clamp
        const int32_t shifted_output =
            unsat_output + static_cast<int32_t>(std::numeric_limits<int8_t>::min());

        out_ptr[c] = static_cast<int8_t>(std::max(
            std::min(shifted_output,
                     static_cast<int32_t>(std::numeric_limits<int8_t>::max())),
            static_cast<int32_t>(std::numeric_limits<int8_t>::min())));
      } else {
        out_ptr[c] = std::numeric_limits<int8_t>::min();
      }
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

  // Fallback for float or int16
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
