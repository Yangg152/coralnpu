// Copyright 2025 Google LLC
#include "sw/opt/litert-micro/mean.h"

#include <riscv_vector.h>
#include <algorithm>
#include <cstdint>
#include <cstring> // for memcpy if needed

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/reduce.h" 

namespace coralnpu_v2::opt::litert_micro {

using tflite::RuntimeShape;
using tflite::OpDataReduce;

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

// 根据规约元素的数量 N，调整乘数和位移，实现 sum / N 的效果
static EffectiveScale CalculateEffectiveScale(int32_t num_elements, int32_t base_multiplier, int32_t base_shift) {
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

// 通用的 RVV 累加与量化处理宏/内联函数
// 参数:
//   acc: 累加器向量
//   scale: 量化参数
//   offset_correction: input_zp * (-N)
//   output_zp: output_zero_point
//   vl: vector length
static inline void PostProcessAndStore(
    vint32m8_t v_acc, int8_t* dst, size_t vl,
    const EffectiveScale& scale, int32_t offset_correction, int32_t output_zp) {
    
    // 1. 应用 Input ZeroPoint 修正 ( v_acc += (-N * zp) )
    v_acc = __riscv_vadd_vx_i32m8(v_acc, offset_correction, vl);

    // 2. 乘法与移位 (实现 / N 和 Requantize)
    constexpr uint32_t vxrm = 0; 
    if (scale.multiplier != 0) {
        v_acc = __riscv_vsmul_vx_i32m8(v_acc, scale.multiplier, vxrm, vl);
        if (scale.shift > 0) {
             v_acc = __riscv_vsll_vx_i32m8(v_acc, scale.shift, vl);
        } else {
             v_acc = __riscv_vssra_vx_i32m8(v_acc, -scale.shift, vxrm, vl);
        }
    }

    // 3. 加 Output ZeroPoint 并 Clip
    v_acc = __riscv_vadd_vx_i32m8(v_acc, output_zp, vl);
    v_acc = __riscv_vmax_vx_i32m8(v_acc, -128, vl);
    v_acc = __riscv_vmin_vx_i32m8(v_acc, 127, vl);

    // 4. 窄化并存储
    vint16m4_t acc_16 = __riscv_vnclip_wx_i16m4(v_acc, 0, vxrm, vl);
    vint8m2_t acc_8 = __riscv_vnclip_wx_i8m2(acc_16, 0, vxrm, vl);
    __riscv_vse8_v_i8m2(dst, acc_8, vl);
}

// -------------------------------------------------------------------------
// 1. Global Pooling (H和W同时规约)
// -------------------------------------------------------------------------
void MeanGlobalPoolingQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width, int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift) {

  const int32_t num_elements = input_height * input_width;
  const int32_t offset_correction = -num_elements * input_zero_point;
  const auto scale = CalculateEffectiveScale(num_elements, multiplier, shift);
  const int32_t stride_c = num_channels; 

  for (int b = 0; b < num_batches; ++b) {
    size_t c_idx = 0;
    size_t c_remain = num_channels;
    // 每次处理 vl 个 Channel
    while (c_remain > 0) {
      size_t vl = __riscv_vsetvl_e8m2(c_remain);
      vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vl);

      const int8_t* src_ptr = input_data + (b * num_elements * stride_c) + c_idx;
      
      // 遍历所有空间像素
      for (int i = 0; i < num_elements; ++i) {
        vint8m2_t v_val = __riscv_vle8_v_i8m2(src_ptr, vl);
        vint16m4_t v_val_16 = __riscv_vsext_vf2_i16m4(v_val, vl);
        v_acc = __riscv_vwadd_wv_i32m8(v_acc, v_val_16, vl);
        src_ptr += stride_c;
      }
      
      PostProcessAndStore(v_acc, output_data + (b * stride_c) + c_idx, vl, scale, offset_correction, output_zero_point);

      c_idx += vl;
      c_remain -= vl;
    }
  }
}

// -------------------------------------------------------------------------
// 2. Reduce Height (Axis=1)
// -------------------------------------------------------------------------
void MeanReduceHeightQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width, int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift) {

  const int32_t num_elements = input_height;
  const int32_t offset_correction = -num_elements * input_zero_point;
  const auto scale = CalculateEffectiveScale(num_elements, multiplier, shift);
  
  // 内存布局: [N, H, W, C]
  // Reduce H -> Output [N, 1, W, C] (或 [N, W, C] 扁平化)
  // 遍历 N 和 W，对 H 进行规约
  
  const int32_t stride_w = num_channels;               // W 维度单步跨度 (1个像素)
  const int32_t stride_h = input_width * num_channels; // H 维度单步跨度 (1行像素)
  const int32_t batch_step = input_height * stride_h;
  
  int8_t* out_ptr = output_data;

  for (int b = 0; b < num_batches; ++b) {
    for (int w = 0; w < input_width; ++w) {
      // 此时针对特定的 (b, w)，规约一列 H
      size_t c_idx = 0;
      size_t c_remain = num_channels;
      
      while (c_remain > 0) {
        size_t vl = __riscv_vsetvl_e8m2(c_remain);
        vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vl);
        
        // 起始指针: Batch起始 + W偏移 + Channel偏移
        const int8_t* src_ptr = input_data + (b * batch_step) + (w * stride_w) + c_idx;

        for (int h = 0; h < input_height; ++h) {
          vint8m2_t v_val = __riscv_vle8_v_i8m2(src_ptr, vl);
          vint16m4_t v_val_16 = __riscv_vsext_vf2_i16m4(v_val, vl);
          v_acc = __riscv_vwadd_wv_i32m8(v_acc, v_val_16, vl);
          
          src_ptr += stride_h; // 跳到下一行
        }
        
        PostProcessAndStore(v_acc, out_ptr, vl, scale, offset_correction, output_zero_point);
        
        out_ptr += vl;
        c_idx += vl;
        c_remain -= vl;
      }
    }
  }
}

// -------------------------------------------------------------------------
// 3. Reduce Width (Axis=2)
// -------------------------------------------------------------------------
void MeanReduceWidthQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width, int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift) {

  const int32_t num_elements = input_width;
  const int32_t offset_correction = -num_elements * input_zero_point;
  const auto scale = CalculateEffectiveScale(num_elements, multiplier, shift);
  
  // 内存布局: [N, H, W, C]
  // Reduce W -> Output [N, H, 1, C]
  // 遍历 N 和 H，对 W 进行规约

  const int32_t stride_w = num_channels;               // W 维度单步跨度 (1个像素)
  const int32_t stride_h = input_width * num_channels; // H 维度单步跨度 (1行像素)
  const int32_t batch_step = input_height * stride_h;

  int8_t* out_ptr = output_data;

  for (int b = 0; b < num_batches; ++b) {
    for (int h = 0; h < input_height; ++h) {
      // 此时针对特定的 (b, h)，规约一行 W
      size_t c_idx = 0;
      size_t c_remain = num_channels;

      while (c_remain > 0) {
        size_t vl = __riscv_vsetvl_e8m2(c_remain);
        vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vl);
        
        // 起始指针: Batch起始 + H偏移 + Channel偏移 (W=0)
        const int8_t* src_ptr = input_data + (b * batch_step) + (h * stride_h) + c_idx;

        for (int w = 0; w < input_width; ++w) {
          vint8m2_t v_val = __riscv_vle8_v_i8m2(src_ptr, vl);
          vint16m4_t v_val_16 = __riscv_vsext_vf2_i16m4(v_val, vl);
          v_acc = __riscv_vwadd_wv_i32m8(v_acc, v_val_16, vl);
          
          src_ptr += stride_w; // 跳到下一个像素
        }

        PostProcessAndStore(v_acc, out_ptr, vl, scale, offset_correction, output_zero_point);
        
        out_ptr += vl;
        c_idx += vl;
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
  
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis_tensor = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  // 解析 Axis
  int num_axis = 0;
  // axis tensor 通常很小，可以放栈上
  int axis[4]; 
  
  if (axis_tensor->type == kTfLiteInt32) {
      num_axis = axis_tensor->dims->data[0];
      const int32_t* axis_data = tflite::micro::GetTensorData<int32_t>(axis_tensor);
      for(int i=0; i<num_axis && i<4; ++i) axis[i] = axis_data[i];
  } else {
      // 这里的 fallback 逻辑略去，通常 axis 是 int32
      return tflite::EvalMeanHelper(context, node, data);
  }

  const auto& in_shape = tflite::micro::GetTensorShape(input);
  
  // 仅针对 4D 输入 NHWC 做优化分发
  if (in_shape.DimensionsCount() == 4) {
      int32_t n = in_shape.Dims(0);
      int32_t h = in_shape.Dims(1);
      int32_t w = in_shape.Dims(2);
      int32_t c = in_shape.Dims(3);

      bool has_axis_1 = false;
      bool has_axis_2 = false;
      for(int i=0; i<num_axis; ++i) {
          if (axis[i] == 1) has_axis_1 = true;
          if (axis[i] == 2) has_axis_2 = true;
      }

      // Case 1: Global Pooling (H and W)
      if (has_axis_1 && has_axis_2) {
          MeanGlobalPoolingQuantizedRVV(
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorData<int8_t>(output),
              n, h, w, c,
              data->input_zp, data->output_zp, data->multiplier, data->shift);
          return kTfLiteOk;
      }
      
      // Case 2: Reduce Height Only
      if (has_axis_1 && !has_axis_2) {
          MeanReduceHeightQuantizedRVV(
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorData<int8_t>(output),
              n, h, w, c,
              data->input_zp, data->output_zp, data->multiplier, data->shift);
          return kTfLiteOk;
      }

      // Case 3: Reduce Width Only
      if (!has_axis_1 && has_axis_2) {
          MeanReduceWidthQuantizedRVV(
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorData<int8_t>(output),
              n, h, w, c,
              data->input_zp, data->output_zp, data->multiplier, data->shift);
          return kTfLiteOk;
      }
  }

  return tflite::EvalMeanHelper(context, node, data);
}

TFLMRegistration Register_MEAN() {
  return tflite::micro::RegisterOp(MeanInit, MeanPrepare, MeanEval);
}

}  // namespace coralnpu_v2::opt::litert_micro
