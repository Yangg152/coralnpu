// sw/opt/litert-micro/conv.h
#ifndef CORALNPU_SW_OPT_LITERT_MICRO_CONV_H_
#define CORALNPU_SW_OPT_LITERT_MICRO_CONV_H_

#include "tensorflow/lite/micro/micro_common.h"
// 新增这一行：定义了 ConvParams 和 RuntimeShape
#include "tensorflow/lite/kernels/internal/types.h" 

namespace coralnpu_v2::opt::litert_micro {
void ConvPerChannel(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data);

TFLMRegistration Register_CONV_2D();
}  // namespace coralnpu_v2::opt::litert_micro

#endif  // CORALNPU_SW_OPT_LITERT_MICRO_CONV_H_
