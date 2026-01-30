// Copyright 2025 Google LLC
#include "sw/opt/litert-micro/sub.h"

#include <algorithm>
#include <cstdint>
#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/sub.h" 

namespace tflite {
extern TFLMRegistration Register_SUB();
}

namespace coralnpu_v2::opt::litert_micro {

using tflite::ArithmeticParams;
using tflite::micro::GetEvalInput;
using tflite::micro::GetEvalOutput;
using tflite::micro::GetTensorData;
using tflite::micro::GetTensorShape;

// 1. 本地定义常量 (TFLite Micro 内部常习惯使用 0, 1, 2)
constexpr int kSubInput1Tensor = 0;
constexpr int kSubInput2Tensor = 1;
constexpr int kSubOutputTensor = 0;

// 2. 这里的结构体必须与 TFLite Micro 官方 sub.cc 中的 OpData 内存布局一致
// 通常情况下，官方实现只包含一个 ArithmeticParams
struct OpData {
  ArithmeticParams arithmetic_params;
};

namespace {

// =========================================================
// RVV 优化核心工具 (保持不变)
// =========================================================

// Scale -> Shift -> Offset -> Clamp -> Narrowing
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

// =========================================================
// Kernel 实现 (保持不变)
// =========================================================

// 场景 1: Element-wise (Vector - Vector)
void SubQuantizedElementWise(
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

    vint16m4_t v_res_16 = __riscv_vsub_vv_i16m4(v1_16, v2_16, vl);
    vint32m8_t v_res_32 = __riscv_vsext_vf2_i32m8(v_res_16, vl);

    VectorQuantizedScaleAndPack(v_res_32, params, output_data, vl);

    input1_data += vl;
    input2_data += vl;
    output_data += vl;
    n -= vl;
  }
}

// 场景 2: Input1(Vector) - Input2(Scalar)
void SubQuantizedBroadcastInput2Scalar(
    const int8_t* input1_data, int8_t input2_val, int8_t* output_data,
    int flat_size, const ArithmeticParams& params) {

  int32_t offset1 = params.input1_offset;
  int16_t scalar_term = (int16_t)((int32_t)input2_val + params.input2_offset);

  size_t n = flat_size;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m2(n);

    vint8m2_t v1 = __riscv_vle8_v_i8m2(input1_data, vl);
    vint16m4_t v1_16 = __riscv_vsext_vf2_i16m4(v1, vl);
    v1_16 = __riscv_vadd_vx_i16m4(v1_16, offset1, vl);

    vint16m4_t v_res_16 = __riscv_vsub_vx_i16m4(v1_16, scalar_term, vl);
    vint32m8_t v_res_32 = __riscv_vsext_vf2_i32m8(v_res_16, vl);

    VectorQuantizedScaleAndPack(v_res_32, params, output_data, vl);

    input1_data += vl;
    output_data += vl;
    n -= vl;
  }
}

// 场景 3: Input1(Scalar) - Input2(Vector)
void SubQuantizedBroadcastInput1Scalar(
    int8_t input1_val, const int8_t* input2_data, int8_t* output_data,
    int flat_size, const ArithmeticParams& params) {

  int16_t scalar_term = (int16_t)((int32_t)input1_val + params.input1_offset);
  int32_t offset2 = params.input2_offset;

  size_t n = flat_size;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m2(n);

    vint8m2_t v2 = __riscv_vle8_v_i8m2(input2_data, vl);
    vint16m4_t v2_16 = __riscv_vsext_vf2_i16m4(v2, vl);
    v2_16 = __riscv_vadd_vx_i16m4(v2_16, offset2, vl);

    // vrsub (Reverse Subtract): scalar - vector
    vint16m4_t v_res_16 = __riscv_vrsub_vx_i16m4(v2_16, scalar_term, vl);
    vint32m8_t v_res_32 = __riscv_vsext_vf2_i32m8(v_res_16, vl);

    VectorQuantizedScaleAndPack(v_res_32, params, output_data, vl);

    input2_data += vl;
    output_data += vl;
    n -= vl;
  }
}

// =========================================================
// TFLite Kernel Interface
// =========================================================

TfLiteStatus SubEval(TfLiteContext* context, TfLiteNode* node) {
  // 关键修改：直接将 user_data 转换为我们定义的 OpData
  // 只要 Register_SUB 使用的是默认的 Prepare，它就会生成这个结构
  const OpData* data = static_cast<const OpData*>(node->user_data);
  const ArithmeticParams& op_params = data->arithmetic_params;

  const TfLiteEvalTensor* input1 = GetEvalInput(context, node, kSubInput1Tensor);
  const TfLiteEvalTensor* input2 = GetEvalInput(context, node, kSubInput2Tensor);
  TfLiteEvalTensor* output = GetEvalOutput(context, node, kSubOutputTensor);

  if (input1->type == kTfLiteInt8 && input2->type == kTfLiteInt8) {
    const int flat_size = GetTensorShape(output).FlatSize();
    const int input1_size = GetTensorShape(input1).FlatSize();
    const int input2_size = GetTensorShape(input2).FlatSize();

    // 1. Element-wise
    if (input1_size == flat_size && input2_size == flat_size) {
      SubQuantizedElementWise(
          GetTensorData<int8_t>(input1), GetTensorData<int8_t>(input2),
          GetTensorData<int8_t>(output), flat_size, op_params);
      return kTfLiteOk;
    } 
    // 2. Broadcast: Vector - Scalar
    else if (input2_size == 1) {
      SubQuantizedBroadcastInput2Scalar(
          GetTensorData<int8_t>(input1), *GetTensorData<int8_t>(input2),
          GetTensorData<int8_t>(output), flat_size, op_params);
      return kTfLiteOk;
    }
    // 3. Broadcast: Scalar - Vector
    else if (input1_size == 1) {
      SubQuantizedBroadcastInput1Scalar(
          *GetTensorData<int8_t>(input1), GetTensorData<int8_t>(input2),
          GetTensorData<int8_t>(output), flat_size, op_params);
      return kTfLiteOk;
    }
  }

  // Fallback: 使用 reference_ops (不是 reference_integer_ops)
  tflite::reference_ops::Sub(
      op_params, 
      GetTensorShape(input1), GetTensorData<int8_t>(input1), 
      GetTensorShape(input2), GetTensorData<int8_t>(input2), 
      GetTensorShape(output), GetTensorData<int8_t>(output));
      
  return kTfLiteOk;
}

TFLMRegistration Register_SUB() {
  // 使用官方注册获取 Init 和 Prepare
  TFLMRegistration r = tflite::Register_SUB();
  // 替换 Eval 函数
  r.invoke = SubEval;
  return r;
}

}  // namespace coralnpu_v2::opt::litert_micro
