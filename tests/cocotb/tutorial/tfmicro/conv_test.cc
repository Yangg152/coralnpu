// tests/cocotb/tutorial/tfmicro/conv_test.cc

#include "sw/opt/litert-micro/conv.h"

#include <cstdint>
#include <vector>

#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

namespace {
// 最大支持的 Tensor 大小，根据需要调整
constexpr size_t kMaxTensorSize = 128 * 1024; 
constexpr size_t kMaxChannels = 256;
}  // namespace

// 定义全局变量，位于固定内存段，供 Python 脚本读写
static tflite::ConvParams params = {
    .padding_type = tflite::PaddingType::kValid,
    .padding_values = {.width = 0, .height = 0},
    // stride 将由 Python 填充
    .stride_width = 1,
    .stride_height = 1,
    .dilation_width_factor = 1,
    .dilation_height_factor = 1,
    .input_offset = 128,  // 模拟 uint8->int8
    .weights_offset = 0,
    .output_offset = -128,
    .quantized_activation_min = -128,
    .quantized_activation_max = 127,
};

// Shape 对象
static tflite::RuntimeShape input_shape_;
static tflite::RuntimeShape filter_shape_;
static tflite::RuntimeShape bias_shape_;
static tflite::RuntimeShape output_shape_;

// 原始数组数据，Python 会写入这些内存
int32_t input_shape[4] __attribute__((section(".data")));
int32_t filter_shape[4] __attribute__((section(".data"))); // [Out, H, W, In]
int32_t bias_shape[1] __attribute__((section(".data")));
int32_t output_shape[4] __attribute__((section(".data")));

int stride_width __attribute__((section(".data"))) = 1;
int stride_height __attribute__((section(".data"))) = 1;

// 数据 Buffer
// 使用 extdata (DRAM) 防止 TCM 溢出
int8_t input_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));
int8_t output_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));
int8_t filter_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));
int32_t bias_data[kMaxChannels] __attribute__((section(".data"), aligned(16)));

// 量化参数 (简化版：假设所有通道相同，或者由 Python 填充)
// 这里简单初始化为固定值，实际测试中 Python 可以覆盖
int32_t output_multiplier[kMaxChannels] __attribute__((section(".data"), aligned(16)));
int32_t output_shift[kMaxChannels] __attribute__((section(".data"), aligned(16)));

void prep() {
  input_shape_.ReplaceWith(4, input_shape);
  filter_shape_.ReplaceWith(4, filter_shape);
  bias_shape_.ReplaceWith(1, bias_shape);
  output_shape_.ReplaceWith(4, output_shape);
  
  params.stride_width = stride_width;
  params.stride_height = stride_height;

  // 修改：int -> size_t 以匹配 kMaxChannels 类型
  for(size_t i=0; i<kMaxChannels; ++i) {
      output_multiplier[i] = 1215836872; 
      output_shift[i] = -7;              
  }
}
extern "C" {
__attribute__((used, retain)) void run_ref() {
  tflite::reference_integer_ops::ConvPerChannel(
      params, output_multiplier, output_shift, input_shape_, input_data,
      filter_shape_, filter_data, bias_shape_, bias_data, output_shape_,
      output_data);
}

__attribute__((used, retain)) void run_optimized() {
  // 调用我们在 conv.h 中暴露的 Wrapper
  coralnpu_v2::opt::litert_micro::ConvPerChannel(
      params, output_multiplier, output_shift, input_shape_, input_data,
      filter_shape_, filter_data, bias_shape_, bias_data, output_shape_,
      output_data);
}
}

void (*impl)() __attribute__((section(".data"))) = run_optimized;

int main(void) {
  prep();
  impl();
  return 0;
}
