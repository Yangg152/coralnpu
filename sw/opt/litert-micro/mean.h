#ifndef SW_OPT_LITERT_MICRO_MEAN_H_
#define SW_OPT_LITERT_MICRO_MEAN_H_

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace coralnpu_v2::opt::litert_micro {

// 暴露核心 RVV 优化函数，方便单元测试
// 专门针对 Global Average Pooling (NHWC -> 1x1xC)
void MeanGlobalPoolingQuantizedRVV(
    const int8_t* input_data, int8_t* output_data,
    int32_t num_batches, int32_t input_height, int32_t input_width, int32_t num_channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift);

// 注册函数
TFLMRegistration Register_MEAN();

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_MEAN_H_
