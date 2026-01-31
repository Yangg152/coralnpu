// Copyright 2025 Google LLC
#ifndef SW_OPT_LITERT_MICRO_MEAN_H_
#define SW_OPT_LITERT_MICRO_MEAN_H_

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace coralnpu_v2::opt::litert_micro {

// 1. 全局池化 (Axis=[1,2])
void MeanGlobalPoolingQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width, int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift);

// 2. 仅规约高度 (Axis=[1]) -> Output [N, 1, W, C]
void MeanReduceHeightQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width, int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift);

// 3. 仅规约宽度 (Axis=[2]) -> Output [N, H, 1, C]
void MeanReduceWidthQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width, int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift);

TFLMRegistration Register_MEAN();

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_MEAN_H_
