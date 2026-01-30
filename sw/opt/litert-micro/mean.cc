// Copyright 2025 Google LLC
#include "sw/opt/litert-micro/mean.h"

#include <riscv_vector.h>
#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/reduce.h" 

namespace coralnpu_v2::opt::litert_micro {

using tflite::RuntimeShape;
using tflite::OpDataReduce;

// =========================================================
// 1. RVV 优化核心函数
// =========================================================
// 注意：该函数必须定义在 coralnpu_v2::opt::litert_micro 命名空间下，
// 不能放在匿名 namespace { ... } 中，否则会与头文件声明冲突。
void MeanGlobalPoolingQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width, int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift) {

  const int32_t num_elements_in_axis = input_height * input_width;
  const int32_t offset_correction = -num_elements_in_axis * input_zero_point;
  
  for (int b = 0; b < num_batches; ++b) {
    // 按 Channel 分块处理
    size_t c_idx = 0;
    size_t c_remain = num_channels;

    while (c_remain > 0) {
      size_t vl = __riscv_vsetvl_e8m2(c_remain); 

      // 初始化累加器
      vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vl);

      const int8_t* src_ptr_base = input_data + (b * num_elements_in_axis * num_channels) + c_idx;
      
      // 遍历 H*W 个像素
      for (int i = 0; i < num_elements_in_axis; ++i) {
        // 加载当前像素的 C[c_idx ... c_idx+vl]
        const int8_t* current_pixel_ptr = src_ptr_base + (i * num_channels);
        
        vint8m2_t v_val = __riscv_vle8_v_i8m2(current_pixel_ptr, vl);
        vint16m4_t v_val_16 = __riscv_vsext_vf2_i16m4(v_val, vl);
        v_acc = __riscv_vwadd_wv_i32m8(v_acc, v_val_16, vl);
      }

      // 后处理
      v_acc = __riscv_vadd_vx_i32m8(v_acc, offset_correction, vl);

      constexpr uint32_t vxrm = 0; 
      if (multiplier != 0) {
        v_acc = __riscv_vsmul_vx_i32m8(v_acc, multiplier, vxrm, vl);
        if (shift > 0) {
             v_acc = __riscv_vsll_vx_i32m8(v_acc, shift, vl);
        } else {
             v_acc = __riscv_vssra_vx_i32m8(v_acc, -shift, vxrm, vl);
        }
      }

      v_acc = __riscv_vadd_vx_i32m8(v_acc, output_zero_point, vl);
      v_acc = __riscv_vmax_vx_i32m8(v_acc, -128, vl);
      v_acc = __riscv_vmin_vx_i32m8(v_acc, 127, vl);

      vint16m4_t acc_16 = __riscv_vnclip_wx_i16m4(v_acc, 0, vxrm, vl);
      vint8m2_t acc_8 = __riscv_vnclip_wx_i8m2(acc_16, 0, vxrm, vl);

      __riscv_vse8_v_i8m2(output_data + (b * num_channels) + c_idx, acc_8, vl);

      c_idx += vl;
      c_remain -= vl;
    }
  }
}

// =========================================================
// 2. TFLite Kernel Interface
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
  // 移除未使用的 axis 变量
  // const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  // 1. 检查是否满足 RVV 优化条件：Global Pooling
  bool is_global_pooling = false;
  
  const auto& in_shape = tflite::micro::GetTensorShape(input);
  const auto& out_shape = tflite::micro::GetTensorShape(output);
  
  // 检查: Rank=4, H,W > 1, Out H,W = 1, Channels 保持不变
  if (in_shape.DimensionsCount() == 4 && out_shape.DimensionsCount() == 4) {
      if (in_shape.Dims(1) > 1 && in_shape.Dims(2) > 1 && 
          out_shape.Dims(1) == 1 && out_shape.Dims(2) == 1 &&
          in_shape.Dims(3) == out_shape.Dims(3)) {
          is_global_pooling = true;
      }
  }

  if (is_global_pooling) {
    int num_batches = in_shape.Dims(0);
    int input_height = in_shape.Dims(1);
    int input_width = in_shape.Dims(2);
    int num_channels = in_shape.Dims(3);

    // 调用我们在上面定义的函数 (不再有歧义)
    MeanGlobalPoolingQuantizedRVV(
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorData<int8_t>(output),
        num_batches, input_height, input_width, num_channels,
        data->input_zp,
        data->output_zp,
        data->multiplier, 
        data->shift);
        
    return kTfLiteOk;
  }

  // 2. Fallback
  return tflite::EvalMeanHelper(context, node, data);
}

TFLMRegistration Register_MEAN() {
  return tflite::micro::RegisterOp(MeanInit, MeanPrepare, MeanEval);
}

}  // namespace coralnpu_v2::opt::litert_micro
