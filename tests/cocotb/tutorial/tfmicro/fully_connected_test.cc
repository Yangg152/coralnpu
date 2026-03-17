// tests/cocotb/tutorial/tfmicro/fc_test.cc

#include "sw/opt/litert-micro/fully_connected.h"

#include <cstdint>

#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"

namespace {
constexpr size_t kMaxTensorSize = 128 * 1024;
constexpr size_t kMaxChannels   = 1024;
}  // namespace

// FullyConnectedParams: FC 没有 padding / stride / dilation，
// 只需要量化零点和激活范围。
static tflite::FullyConnectedParams params = {
    .input_offset              = 128,   // uint8 -> int8: 0 -> -128
    .weights_offset            = 0,     // 对称量化权重，zero_point = 0
    .output_offset             = -128,
    .quantized_activation_min  = -128,
    .quantized_activation_max  = 127,
};

// RuntimeShape wrappers（由 prep() 填充）
static tflite::RuntimeShape input_shape_;
static tflite::RuntimeShape filter_shape_;
static tflite::RuntimeShape bias_shape_;
static tflite::RuntimeShape output_shape_;

// === 暴露给 Python 的形状变量 ===
// input_shape  [1, accum_depth]
// filter_shape [output_depth, accum_depth]
// output_shape [1, output_depth]
int32_t input_shape[2]  __attribute__((section(".data")));
int32_t filter_shape[2] __attribute__((section(".data")));
int32_t bias_shape[1]   __attribute__((section(".data")));
int32_t output_shape[2] __attribute__((section(".data")));

// 数据 Buffer
int8_t  input_data[kMaxTensorSize]  __attribute__((section(".data"), aligned(16)));
int8_t  output_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));
int8_t  filter_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));
int32_t bias_data[kMaxChannels]     __attribute__((section(".data"), aligned(16)));

// 量化参数（per-channel）
int32_t output_multiplier[kMaxChannels] __attribute__((section(".data"), aligned(16)));
int32_t output_shift[kMaxChannels] __attribute__((section(".data"), aligned(16)));

// 累加器 scratch buffer（供优化版使用）
int32_t accs_buf[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));

void prep() {
  input_shape_.ReplaceWith(2, input_shape);
  filter_shape_.ReplaceWith(2, filter_shape);
  bias_shape_.ReplaceWith(1, bias_shape);
  output_shape_.ReplaceWith(2, output_shape);

  const int output_depth = filter_shape[0];
  for (int i = 0; i < output_depth; ++i) {
    output_multiplier[i] = 1215836872;
    output_shift[i]      = -7;
  }
}

extern "C" {
__attribute__((used, retain)) void run_ref() {
  tflite::reference_integer_ops::FullyConnectedPerChannel(
      params, output_multiplier,
      reinterpret_cast<const int*>(output_shift),  // ← 只加这一个 cast
      input_shape_,  input_data,
      filter_shape_, filter_data,
      bias_shape_,   bias_data,
      output_shape_, output_data);
}


__attribute__((used, retain)) void run_optimized() {
  coralnpu_v2::opt::litert_micro::FullyConnectedPerChannel(
      params, output_multiplier, output_shift,
      input_shape_,  input_data,
      filter_shape_, filter_data,
      bias_shape_,   bias_data,
      output_shape_, output_data,
      accs_buf);
}
}  // extern "C"

void (*impl)() __attribute__((section(".data"))) = run_optimized;

int main(void) {
  prep();
  impl();
  return 0;
}
