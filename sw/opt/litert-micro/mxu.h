#ifndef CORALNPU_SW_OPT_LITERT_MICRO_MXU_H_
#define CORALNPU_SW_OPT_LITERT_MICRO_MXU_H_

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace coralnpu_v2::opt::litert_micro {

// 专门为 MXU 优化的 Conv2D / MatMul 算子入口
void MxuConvPerChannel(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data);

// 注册使用 MXU 硬件加速的 Conv2D 算子
TFLMRegistration Register_MXU_CONV_2D();

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // CORALNPU_SW_OPT_LITERT_MICRO_MXU_H_
