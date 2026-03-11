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

#include "sw/opt/litert-micro/mean.h"

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/reduce.h"

namespace coralnpu_v2::opt::litert_micro {

using tflite::OpDataReduce;
using tflite::RuntimeShape;

// -------------------------------------------------------------------------
// 辅助函数
// -------------------------------------------------------------------------

static inline int CountLeadingZeros32(uint32_t x) {
  return x == 0 ? 32 : __builtin_clz(x);
}

struct EffectiveScale {
  int32_t multiplier;
  int32_t shift;
};

// 根据规约元素的数量 N，调整乘数和位移，实现 sum / N 的效果。
// PrepareMeanOrSumHelper 存储的 shift 约定：负数表示右移（SmallerThanOneExp 约定）。
static EffectiveScale CalculateEffectiveScale(int32_t num_elements,
                                              int32_t base_multiplier,
                                              int32_t base_shift) {
  if (num_elements <= 0) return {0, 0};
  if (num_elements == 1) return {base_multiplier, base_shift};

  int division_shift = 31 - CountLeadingZeros32(static_cast<uint32_t>(num_elements));
  division_shift = std::min(division_shift, 32);
  division_shift = std::min(division_shift, static_cast<int>(31 + base_shift));

  int32_t effective_multiplier = static_cast<int32_t>(
      (static_cast<int64_t>(base_multiplier) << division_shift) / num_elements);
  int32_t effective_shift = base_shift - division_shift;

  return {effective_multiplier, effective_shift};
}

// 通用的 RVV 后处理与存储。
// shift 约定与 TFLite SmallerThanOneExp 一致：shift > 0 左移，shift < 0 右移。
static inline void PostProcessAndStore(
    vint32m8_t v_acc, int8_t* dst, size_t vl,
    const EffectiveScale& scale, int32_t offset_correction, int32_t output_zp) {

  constexpr uint32_t vxrm = 0;  // round-to-nearest-up

  // 步骤 1：应用 input_zero_point 修正
  v_acc = __riscv_vadd_vx_i32m8(v_acc, offset_correction, vl);

  // 步骤 2：乘法与移位（实现 / N 和 Requantize）
  if (scale.multiplier != 0) {
    v_acc = __riscv_vsmul_vx_i32m8(v_acc, scale.multiplier, vxrm, vl);
    // shift > 0 左移，shift < 0 右移（SmallerThanOneExp 约定）
    if (scale.shift > 0) {
      v_acc = __riscv_vsll_vx_i32m8(v_acc, static_cast<size_t>(scale.shift), vl);
    } else if (scale.shift < 0) {
      v_acc = __riscv_vssra_vx_i32m8(v_acc, static_cast<size_t>(-scale.shift), vxrm, vl);
    }
  }

  // 步骤 3：加 output_zp 并 clip
  v_acc = __riscv_vadd_vx_i32m8(v_acc, output_zp, vl);
  v_acc = __riscv_vmax_vx_i32m8(v_acc, -128, vl);
  v_acc = __riscv_vmin_vx_i32m8(v_acc, 127, vl);

  // 步骤 4：窄化并存储
  vint16m4_t acc_16 = __riscv_vnclip_wx_i16m4(v_acc, 0, vxrm, vl);
  vint8m2_t  acc_8  = __riscv_vnclip_wx_i8m2(acc_16, 0, vxrm, vl);
  __riscv_vse8_v_i8m2(dst, acc_8, vl);
}

// =========================================================
// 1. Global Pooling（Axis = {1, 2}，即 H 和 W 同时规约）
//
// 输入布局：[N, H, W, C]
// 输出布局：[N, C]（展平为 [N, 1, 1, C]）
// =========================================================
void MeanGlobalPoolingQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width,
    int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift) {

  const int32_t num_elements = input_height * input_width;
  // 修复：用 int64 防止 num_elements * input_zero_point 溢出
  const int32_t offset_correction = static_cast<int32_t>(
      -static_cast<int64_t>(num_elements) * input_zero_point);
  const auto scale = CalculateEffectiveScale(num_elements, multiplier, shift);
  const int32_t stride_c = num_channels;

  for (int b = 0; b < num_batches; ++b) {
    size_t c_idx    = 0;
    size_t c_remain = static_cast<size_t>(num_channels);

    while (c_remain > 0) {
      // 修复：使用 e32m8，与累加器 vint32m8_t 类型一致
      const size_t vl = __riscv_vsetvl_e32m8(c_remain);
      vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vl);

      const int8_t* src_ptr =
          input_data + (b * num_elements * stride_c) + c_idx;

      for (int i = 0; i < num_elements; ++i) {
        vint8m2_t  v_val    = __riscv_vle8_v_i8m2(src_ptr, vl);
        vint16m4_t v_val_16 = __riscv_vsext_vf2_i16m4(v_val, vl);
        v_acc = __riscv_vwadd_wv_i32m8(v_acc, v_val_16, vl);
        src_ptr += stride_c;
      }

      PostProcessAndStore(v_acc,
                          output_data + (b * stride_c) + c_idx,
                          vl, scale, offset_correction, output_zero_point);

      c_idx    += vl;
      c_remain -= vl;
    }
  }
}

// =========================================================
// 2. Reduce Height（Axis = 1）
//
// 输入布局：[N, H, W, C]
// 输出布局：[N, W, C]（展平为 [N, 1, W, C]）
// =========================================================
void MeanReduceHeightQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width,
    int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift) {

  const int32_t num_elements = input_height;
  const int32_t offset_correction = static_cast<int32_t>(
      -static_cast<int64_t>(num_elements) * input_zero_point);
  const auto scale = CalculateEffectiveScale(num_elements, multiplier, shift);

  const int32_t stride_w  = num_channels;
  const int32_t stride_h  = input_width * num_channels;
  const int32_t batch_step = input_height * stride_h;

  int8_t* out_ptr = output_data;

  for (int b = 0; b < num_batches; ++b) {
    for (int w = 0; w < input_width; ++w) {
      size_t c_idx    = 0;
      size_t c_remain = static_cast<size_t>(num_channels);

      while (c_remain > 0) {
        const size_t vl = __riscv_vsetvl_e32m8(c_remain);
        vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vl);

        const int8_t* src_ptr =
            input_data + (b * batch_step) + (w * stride_w) + c_idx;

        for (int h = 0; h < input_height; ++h) {
          vint8m2_t  v_val    = __riscv_vle8_v_i8m2(src_ptr, vl);
          vint16m4_t v_val_16 = __riscv_vsext_vf2_i16m4(v_val, vl);
          v_acc = __riscv_vwadd_wv_i32m8(v_acc, v_val_16, vl);
          src_ptr += stride_h;
        }

        PostProcessAndStore(v_acc, out_ptr, vl, scale,
                            offset_correction, output_zero_point);

        out_ptr  += vl;
        c_idx    += vl;
        c_remain -= vl;
      }
    }
  }
}

// =========================================================
// 3. Reduce Width（Axis = 2）
//
// 输入布局：[N, H, W, C]
// 输出布局：[N, H, C]（展平为 [N, H, 1, C]）
// =========================================================
void MeanReduceWidthQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width,
    int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift) {

  const int32_t num_elements = input_width;
  const int32_t offset_correction = static_cast<int32_t>(
      -static_cast<int64_t>(num_elements) * input_zero_point);
  const auto scale = CalculateEffectiveScale(num_elements, multiplier, shift);

  const int32_t stride_w  = num_channels;
  const int32_t stride_h  = input_width * num_channels;
  const int32_t batch_step = input_height * stride_h;

  int8_t* out_ptr = output_data;

  for (int b = 0; b < num_batches; ++b) {
    for (int h = 0; h < input_height; ++h) {
      size_t c_idx    = 0;
      size_t c_remain = static_cast<size_t>(num_channels);

      while (c_remain > 0) {
        const size_t vl = __riscv_vsetvl_e32m8(c_remain);
        vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vl);

        const int8_t* src_ptr =
            input_data + (b * batch_step) + (h * stride_h) + c_idx;

        for (int w = 0; w < input_width; ++w) {
          vint8m2_t  v_val    = __riscv_vle8_v_i8m2(src_ptr, vl);
          vint16m4_t v_val_16 = __riscv_vsext_vf2_i16m4(v_val, vl);
          v_acc = __riscv_vwadd_wv_i32m8(v_acc, v_val_16, vl);
          src_ptr += stride_w;
        }

        PostProcessAndStore(v_acc, out_ptr, vl, scale,
                            offset_correction, output_zero_point);

        out_ptr  += vl;
        c_idx    += vl;
        c_remain -= vl;
      }
    }
  }
}

// =========================================================
// TFLite Kernel Interface
// =========================================================

void* MeanInit(TfLiteContext* context, const char* buffer, size_t length) {
  return context->AllocatePersistentBuffer(context, sizeof(OpDataReduce));
}

TfLiteStatus MeanPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  return tflite::PrepareMeanOrSumHelper(context, node, op_data);
}

TfLiteStatus MeanEval(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* data = static_cast<OpDataReduce*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis_tensor =
      tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, 0);

  // 修复：安全解析 axis，支持 0-D scalar 和负数 axis
  int num_axis = 0;
  int axis[4]  = {};

  if (axis_tensor->type == kTfLiteInt32) {
    if (axis_tensor->dims->size == 0) {
      // 0-D scalar：只有一个 axis 值
      num_axis = 1;
      axis[0]  = tflite::micro::GetTensorData<int32_t>(axis_tensor)[0];
    } else {
      num_axis = axis_tensor->dims->data[0];
      const int32_t* axis_data =
          tflite::micro::GetTensorData<int32_t>(axis_tensor);
      for (int i = 0; i < num_axis && i < 4; ++i) axis[i] = axis_data[i];
    }
  } else {
    return tflite::EvalMeanHelper(context, node, data);
  }

  const auto& in_shape = tflite::micro::GetTensorShape(input);

  // 仅针对 4D int8 输入 NHWC 做 RVV 优化分发
  if (in_shape.DimensionsCount() == 4 &&
      input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
    const int32_t n = in_shape.Dims(0);
    const int32_t h = in_shape.Dims(1);
    const int32_t w = in_shape.Dims(2);
    const int32_t c = in_shape.Dims(3);

    bool has_axis_1 = false;
    bool has_axis_2 = false;
    for (int i = 0; i < num_axis; ++i) {
      // 修复：支持负数 axis（如 -3 => 1，-2 => 2）
      int a = axis[i];
      if (a < 0) a += 4;
      if (a == 1) has_axis_1 = true;
      if (a == 2) has_axis_2 = true;
    }

    // Case 1：Global Pooling（同时规约 H 和 W）
    if (has_axis_1 && has_axis_2) {
      MeanGlobalPoolingQuantizedRVV(
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorData<int8_t>(output),
          n, h, w, c,
          data->input_zp, data->output_zp,
          data->multiplier, data->shift);
      return kTfLiteOk;
    }

    // Case 2：仅规约 Height
    if (has_axis_1 && !has_axis_2) {
      MeanReduceHeightQuantizedRVV(
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorData<int8_t>(output),
          n, h, w, c,
          data->input_zp, data->output_zp,
          data->multiplier, data->shift);
      return kTfLiteOk;
    }

    // Case 3：仅规约 Width
    if (!has_axis_1 && has_axis_2) {
      MeanReduceWidthQuantizedRVV(
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorData<int8_t>(output),
          n, h, w, c,
          data->input_zp, data->output_zp,
          data->multiplier, data->shift);
      return kTfLiteOk;
    }
  }

  // Fallback：其他维度/类型走参考实现
  return tflite::EvalMeanHelper(context, node, data);
}

TFLMRegistration Register_MEAN() {
  return tflite::micro::RegisterOp(MeanInit, MeanPrepare, MeanEval);
}

}  // namespace coralnpu_v2::opt::litert_micro
