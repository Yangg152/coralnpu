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
using FixedPointAccum      = gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;
using FixedPoint0          = gemmlowp::FixedPoint<int32_t, 0>;

void SoftmaxQuantized(
    const SoftmaxParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& output_shape, int8_t* output_data) {

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size   = MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth        = MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  const int32_t input_beta_multiplier = params.input_multiplier;
  const int32_t input_beta_left_shift = params.input_left_shift;
  const int     diff_min              = params.diff_min;

  // 提前计算 LUT 有效范围：-diff >= diff_min => diff <= -diff_min
  const int valid_diff_max = std::min(255, static_cast<int>(-diff_min));

  // =========================================================
  // 小尺寸张量退回路径：纯标量实现
  // 避免极短向量的 RVV 启动开销与建立行级 LUT 的过量开销
  // =========================================================
  if (depth < 128) {
    for (int i = 0; i < outer_size; ++i) {
      const int8_t* in_ptr  = input_data  + i * depth;
      int8_t*       out_ptr = output_data + i * depth;

      // 1. 纯标量 Find Max
      int8_t max_in_row = std::numeric_limits<int8_t>::min();
      for (int c = 0; c < depth; ++c) {
        max_in_row = std::max(max_in_row, in_ptr[c]);
      }

      // 2. 纯标量 Exp Sum
      FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
      for (int c = 0; c < depth; ++c) {
        const int32_t diff = static_cast<int32_t>(in_ptr[c]) - static_cast<int32_t>(max_in_row);
        if (diff >= diff_min) {
          const int32_t input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(
              diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);
          sum_of_exps = sum_of_exps +
              gemmlowp::Rescale<kAccumulationIntegerBits>(
                  gemmlowp::exp_on_negative_values(scaled_diff_f8));
        }
      }

      // 3. 计算倒数尺度
      int       num_bits_over_unit;
      FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
          sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));
      const int exponent = num_bits_over_unit + 31 - 8;

      // 4. 纯标量归一化输出
      for (int c = 0; c < depth; ++c) {
        const int32_t diff = static_cast<int32_t>(in_ptr[c]) - static_cast<int32_t>(max_in_row);
        if (diff >= diff_min) {
          const int32_t input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(
              diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);
          FixedPoint0 exp_in_0     = gemmlowp::exp_on_negative_values(scaled_diff_f8);
          int32_t     unsat_output = gemmlowp::RoundingDivideByPOT(
              (shifted_scale * exp_in_0).raw(), exponent);
          const int32_t shifted_output = unsat_output + static_cast<int32_t>(-128);
          out_ptr[c] = static_cast<int8_t>(
              std::max(std::min(shifted_output, static_cast<int32_t>(127)),
                       static_cast<int32_t>(-128)));
        } else {
          out_ptr[c] = -128;
        }
      }
    }
    return;
  }

  // =========================================================
  // 大尺寸张量优化路径：全局 LUT + 行级 LUT + RVV 向量查表访存
  // =========================================================

  // 建立全局 Exp LUT（仅计算有效范围，其余填 0 / -128）
  int32_t exp_lut[256];
  int32_t exp_sum_lut[256];

  for (int diff = 0; diff <= valid_diff_max; ++diff) {
    const int32_t input_diff            = -diff;
    const int32_t input_diff_rescaled   = MultiplyByQuantizedMultiplierGreaterThanOne(
        input_diff, input_beta_multiplier, input_beta_left_shift);
    const FixedPointScaledDiff scaled_diff_f8 =
        FixedPointScaledDiff::FromRaw(input_diff_rescaled);
    FixedPoint0 exp_val = gemmlowp::exp_on_negative_values(scaled_diff_f8);
    exp_lut[diff]     = exp_val.raw();
    exp_sum_lut[diff] = gemmlowp::Rescale<kAccumulationIntegerBits>(exp_val).raw();
  }
  for (int diff = valid_diff_max + 1; diff < 256; ++diff) {
    exp_lut[diff]     = 0;
    exp_sum_lut[diff] = 0;
  }

  for (int i = 0; i < outer_size; ++i) {
    const int8_t* in_ptr  = input_data  + i * depth;
    int8_t*       out_ptr = output_data + i * depth;

    // ----------------------------------------------------------
    // 步骤 1：Find Max（向量化 e8m8）
    // ----------------------------------------------------------
    int8_t   max_in_row = std::numeric_limits<int8_t>::min();
    size_t   n          = static_cast<size_t>(depth);
    const int8_t* ptr   = in_ptr;
    vint8m1_t v_max_val = __riscv_vmv_v_x_i8m1(max_in_row, 1);
    while (n > 0) {
      size_t    vl     = __riscv_vsetvl_e8m8(n);
      vint8m8_t v_data = __riscv_vle8_v_i8m8(ptr, vl);
      v_max_val = __riscv_vredmax_vs_i8m8_i8m1(v_data, v_max_val, vl);
      ptr += vl;
      n   -= vl;
    }
    max_in_row = __riscv_vmv_x_s_i8m1_i8(v_max_val);

    // ----------------------------------------------------------
    // 步骤 2：向量化累加 exp_sum（gather + vredsum）
    // exp_sum_lut 是 int32（4字节），byte offset = diff * 4
    // ----------------------------------------------------------
    int32_t sum_raw = 0;
    n   = static_cast<size_t>(depth);
    ptr = in_ptr;
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m2(n);

      // 计算 diff = max - input 
      // 巧用 uint8 模 256 算术特性：已知 max >= input，重解释为 uint8 做向量减法，
      // 溢出回绕的结果天然即是 [0, 255] 的正差值，省去升位开销
      vuint8m2_t v_diff8 = __riscv_vrsub_vx_u8m2(
          __riscv_vreinterpret_v_i8m2_u8m2(__riscv_vle8_v_i8m2(ptr, vl)),
          static_cast<uint8_t>(max_in_row), vl);

      // 扩展为 32bit byte offset（*4 因为 int32 元素占据 4 字节）
      vuint32m8_t v_offset = __riscv_vwmulu_vx_u32m8(
          __riscv_vzext_vf2_u16m4(v_diff8, vl), 4u, vl);

      // Gather exp_sum_lut
      vint32m8_t v_exp = __riscv_vluxei32_v_i32m8(exp_sum_lut, v_offset, vl);

      // 水平求和
      vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t v_sum  = __riscv_vredsum_vs_i32m8_i32m1(v_exp, v_zero, vl);
      sum_raw += __riscv_vmv_x_s_i32m1_i32(v_sum);

      ptr += vl;
      n   -= vl;
    }
    FixedPointAccum sum_of_exps = FixedPointAccum::FromRaw(sum_raw);

    // ----------------------------------------------------------
    // 步骤 3：计算倒数尺度
    // ----------------------------------------------------------
    int       num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));
    const int exponent = num_bits_over_unit + 31 - 8;

    // ----------------------------------------------------------
    // 步骤 4：构建行级输出 LUT（仅计算有效范围）
    // ----------------------------------------------------------
    int8_t row_out_lut[256];

    for (int diff = 0; diff <= valid_diff_max; ++diff) {
      FixedPoint0 exp_in_0     = FixedPoint0::FromRaw(exp_lut[diff]);
      int32_t     unsat_output = gemmlowp::RoundingDivideByPOT(
          (shifted_scale * exp_in_0).raw(), exponent);
      int32_t shifted_output   = unsat_output + static_cast<int32_t>(-128);
      row_out_lut[diff] = static_cast<int8_t>(
          std::max(std::min(shifted_output, static_cast<int32_t>(127)),
                   static_cast<int32_t>(-128)));
    }
    // diff > valid_diff_max 区域一律输出 -128
    for (int diff = valid_diff_max + 1; diff < 256; ++diff) {
      row_out_lut[diff] = -128;
    }

    // ----------------------------------------------------------
    // 步骤 5：向量化 gather 输出（vluxei8）
    // row_out_lut 是 int8（1字节），byte offset = diff，直接用
    // ----------------------------------------------------------
    n           = static_cast<size_t>(depth);
    ptr         = in_ptr;
    int8_t* out = out_ptr;
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m8(n);

      vint8m8_t  v_data = __riscv_vle8_v_i8m8(ptr, vl);
      
      // 巧用 uint8 模 256 算术特性：同步骤 2，直接得到正确的查表索引
      vuint8m8_t v_diff = __riscv_vrsub_vx_u8m8(
          __riscv_vreinterpret_v_i8m8_u8m8(v_data),
          static_cast<uint8_t>(max_in_row), vl);

      vint8m8_t v_out = __riscv_vluxei8_v_i8m8(row_out_lut, v_diff, vl);
      __riscv_vse8_v_i8m8(out, v_out, vl);

      ptr += vl;
      out += vl;
      n   -= vl;
    }
  }
}

// =========================================================
// TFLite Interface
// =========================================================

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const SoftmaxParams* params =
      static_cast<const SoftmaxParams*>(node->builtin_data);

  const TfLiteEvalTensor* input  = GetEvalInput(context, node, 0);
  TfLiteEvalTensor*       output = GetEvalOutput(context, node, 0);

  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
    SoftmaxQuantized(
        *params,
        GetTensorShape(input),  GetTensorData<int8_t>(input),
        GetTensorShape(output), GetTensorData<int8_t>(output));
    return kTfLiteOk;
  }

  tflite::reference_ops::Softmax(
      *params,
      GetTensorShape(input),  GetTensorData<float>(input),
      GetTensorShape(output), GetTensorData<float>(output));

  return kTfLiteOk;
}

TFLMRegistration Register_SOFTMAX() {
  TFLMRegistration r = tflite::Register_SOFTMAX();
  r.invoke = SoftmaxEval;
  return r;
}

}  // namespace coralnpu_v2::opt::litert_micro
