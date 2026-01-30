// Copyright 2025 Google LLC
#ifndef SW_OPT_LITERT_MICRO_SUB_H_
#define SW_OPT_LITERT_MICRO_SUB_H_

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace coralnpu_v2::opt::litert_micro {

// 暴露优化后的核心函数，用于单元测试
// 注意：减法不满足交换律，因此 Broadcast 需要区分两种情况

// 1. Element-wise: Input1(Vector) - Input2(Vector)
void SubQuantizedElementWise(
    const int8_t* input1_data, const int8_t* input2_data, int8_t* output_data,
    int flat_size, const tflite::ArithmeticParams& params);

// 2. Broadcast: Input1(Vector) - Input2(Scalar)
void SubQuantizedBroadcastInput2Scalar(
    const int8_t* input1_data, int8_t input2_val, int8_t* output_data,
    int flat_size, const tflite::ArithmeticParams& params);

// 3. Broadcast: Input1(Scalar) - Input2(Vector)
void SubQuantizedBroadcastInput1Scalar(
    int8_t input1_val, const int8_t* input2_data, int8_t* output_data,
    int flat_size, const tflite::ArithmeticParams& params);

// 注册函数
TFLMRegistration Register_SUB();

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_SUB_H_
