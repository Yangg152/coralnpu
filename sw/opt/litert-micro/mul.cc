// Copyright 2025 Google LLC
#include "sw/opt/litert-micro/mul.h"

#include <riscv_vector.h>
#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/mul.h" 

namespace coralnpu_v2::opt::litert_micro {

using tflite::ArithmeticParams;
using tflite::RuntimeShape;
using tflite::micro::GetEvalInput;
using tflite::micro::GetEvalOutput;
using tflite::micro::GetTensorData;
using tflite::micro::GetTensorShape;

// 现在可以直接使用 tflite 命名空间下的定义了
using tflite::OpDataMul;
using tflite::kMulInput1Tensor;
using tflite::kMulInput2Tensor;
using tflite::kMulOutputTensor;

namespace {

// =========================================================
// 1. RVV 优化核心函数 (保持不变)
// =========================================================

// 辅助：向量化量化后处理
inline void VectorQuantizedScaleAndPack(
    vint32m8_t acc, const ArithmeticParams& params, int8_t* out_ptr, size_t vl) {
  
  constexpr uint32_t vxrm = 0; // Round-to-nearest-up (RNU)
  
  if (params.output_multiplier != 0) {
      acc = __riscv_vsmul_vx_i32m8(acc, params.output_multiplier, vxrm, vl);
      if (params.output_shift > 0) {
           acc = __riscv_vsll_vx_i32m8(acc, params.output_shift, vl);
      } else {
           acc = __riscv_vssra_vx_i32m8(acc, -params.output_shift, vxrm, vl);
      }
  }

  acc = __riscv_vadd_vx_i32m8(acc, params.output_offset, vl);
  acc = __riscv_vmax_vx_i32m8(acc, params.quantized_activation_min, vl);
  acc = __riscv_vmin_vx_i32m8(acc, params.quantized_activation_max, vl);

  vint16m4_t acc_16 = __riscv_vnclip_wx_i16m4(acc, 0, vxrm, vl);
  vint8m2_t acc_8 = __riscv_vnclip_wx_i8m2(acc_16, 0, vxrm, vl);

  __riscv_vse8_v_i8m2(out_ptr, acc_8, vl);
}

} // namespace

// 场景 1: Element-wise
void MulQuantizedElementWise(
    const int8_t* input1_data, const int8_t* input2_data, int8_t* output_data,
    int flat_size, const ArithmeticParams& params) {

  int32_t offset1 = params.input1_offset;
  int32_t offset2 = params.input2_offset;
  
  size_t n = flat_size;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m2(n);

    vint8m2_t v1 = __riscv_vle8_v_i8m2(input1_data, vl);
    vint8m2_t v2 = __riscv_vle8_v_i8m2(input2_data, vl);

    vint16m4_t v1_16 = __riscv_vsext_vf2_i16m4(v1, vl);
    vint16m4_t v2_16 = __riscv_vsext_vf2_i16m4(v2, vl);

    v1_16 = __riscv_vadd_vx_i16m4(v1_16, offset1, vl);
    v2_16 = __riscv_vadd_vx_i16m4(v2_16, offset2, vl);

    vint32m8_t v_acc = __riscv_vwmul_vv_i32m8(v1_16, v2_16, vl);

    VectorQuantizedScaleAndPack(v_acc, params, output_data, vl);

    input1_data += vl;
    input2_data += vl;
    output_data += vl;
    n -= vl;
  }
}

// 场景 2: Broadcast Scalar
void MulQuantizedBroadcastScalar(
    const int8_t* input_vec_data, int8_t input_scalar_val, int8_t* output_data,
    int flat_size, const ArithmeticParams& params) {

  int32_t offset1 = params.input1_offset;
  int16_t scalar_term = (int16_t)((int32_t)input_scalar_val + params.input2_offset);

  size_t n = flat_size;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m2(n);

    vint8m2_t v1 = __riscv_vle8_v_i8m2(input_vec_data, vl);
    vint16m4_t v1_16 = __riscv_vsext_vf2_i16m4(v1, vl);
    v1_16 = __riscv_vadd_vx_i16m4(v1_16, offset1, vl);

    vint32m8_t v_acc = __riscv_vwmul_vx_i32m8(v1_16, scalar_term, vl);

    VectorQuantizedScaleAndPack(v_acc, params, output_data, vl);

    input_vec_data += vl;
    output_data += vl;
    n -= vl;
  }
}

// =========================================================
// 2. TFLite Kernel Interface
// =========================================================

TfLiteStatus MulEval(TfLiteContext* context, TfLiteNode* node) {
  // 1. 获取 user_data
  // 因为我们复用了官方的 Init/Prepare，所以这里的 user_data 就是官方定义的 OpDataMul
  const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);

  const TfLiteEvalTensor* input1 = GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 = GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output = GetEvalOutput(context, node, kMulOutputTensor);

  // 2. 转换参数格式：从 OpDataMul (存储层) -> ArithmeticParams (计算层)
  ArithmeticParams op_params;
  op_params.input1_offset = -data->input1_zero_point;
  op_params.input2_offset = -data->input2_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  
  // 浮点激活极值，用于 fallback 的 float 实现 (OpDataMul 成员名是 _f32)
  op_params.float_activation_min = data->output_activation_min_f32;
  op_params.float_activation_max = data->output_activation_max_f32;

  // 3. 执行优化逻辑
  if (input1->type == kTfLiteInt8 && input2->type == kTfLiteInt8) {
    const int flat_size = GetTensorShape(output).FlatSize();
    const int input1_size = GetTensorShape(input1).FlatSize();
    const int input2_size = GetTensorShape(input2).FlatSize();

    if (input1_size == flat_size && input2_size == flat_size) {
      MulQuantizedElementWise(
          GetTensorData<int8_t>(input1), GetTensorData<int8_t>(input2),
          GetTensorData<int8_t>(output), flat_size, op_params);
      return kTfLiteOk;
    } 
    else if (input2_size == 1) {
      MulQuantizedBroadcastScalar(
          GetTensorData<int8_t>(input1), *GetTensorData<int8_t>(input2),
          GetTensorData<int8_t>(output), flat_size, op_params);
      return kTfLiteOk;
    }
    else if (input1_size == 1) {
      ArithmeticParams swapped_params = op_params;
      swapped_params.input1_offset = op_params.input2_offset;
      swapped_params.input2_offset = op_params.input1_offset;
      
      MulQuantizedBroadcastScalar(
          GetTensorData<int8_t>(input2), *GetTensorData<int8_t>(input1),
          GetTensorData<int8_t>(output), flat_size, swapped_params);
      return kTfLiteOk;
    }
  }

  // Fallback to Reference
  tflite::reference_integer_ops::Mul(
      op_params, 
      GetTensorShape(input1), GetTensorData<int8_t>(input1), 
      GetTensorShape(input2), GetTensorData<int8_t>(input2), 
      GetTensorShape(output), GetTensorData<int8_t>(output));
      
  return kTfLiteOk;
}

TFLMRegistration Register_MUL() {
  // 使用官方的 Register_MUL 获取默认的 Init/Prepare
  // 这样我们不需要自己实现 Init/Prepare，OpDataMul 的内存分配和填充完全由 TFLite 处理
  TFLMRegistration r = tflite::Register_MUL();
  
  // 仅替换 Invoke 函数为我们的 RVV 优化版
  r.invoke = MulEval;
  
  return r;
}

}  // namespace coralnpu_v2::opt::litert_micro
