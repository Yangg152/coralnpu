#ifndef SW_OPT_LITERT_MICRO_MUL_H_
#define SW_OPT_LITERT_MICRO_MUL_H_

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace coralnpu_v2::opt::litert_micro {

// 暴露优化后的核心函数，接受标准的 ArithmeticParams
// 这样在单元测试中可以直接调用，无需构造完整的 TfLiteNode
void MulQuantizedElementWise(
    const int8_t* input1_data, const int8_t* input2_data, int8_t* output_data,
    int flat_size, const tflite::ArithmeticParams& params);

void MulQuantizedBroadcastScalar(
    const int8_t* input_vec_data, int8_t input_scalar_val, int8_t* output_data,
    int flat_size, const tflite::ArithmeticParams& params);

// 注册函数
TFLMRegistration Register_MUL();

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_MUL_H_
