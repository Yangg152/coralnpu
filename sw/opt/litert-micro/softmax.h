// Copyright 2025 Google LLC
#ifndef SW_OPT_LITERT_MICRO_SOFTMAX_H_
#define SW_OPT_LITERT_MICRO_SOFTMAX_H_

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace coralnpu_v2::opt::litert_micro {

// 暴露优化后的核心函数
// 针对 TFLite Micro 标准的 Int8 Softmax
void SoftmaxQuantized(
    const tflite::SoftmaxParams& params,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data);

// 注册函数
TFLMRegistration Register_SOFTMAX();

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_SOFTMAX_H_
