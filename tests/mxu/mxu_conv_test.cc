// tests/cocotb/tutorial/tfmicro/conv_test.cc

// 1. 【修改】将原来的 conv.h 替换为刚刚写的 mxu.h
#include "sw/opt/litert-micro/mxu.h"
#include "sw/opt/litert-micro/conv.h" // 保留这个如果需要其它基础定义

#include <cstdint>
#include <vector>

#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

namespace {
// 适当增大 Buffer 以容纳 MobileNet 的层
constexpr size_t kMaxTensorSize = 128 * 1024; 
constexpr size_t kMaxChannels = 1024;
}  // namespace

// 默认参数 (保持不变)
static tflite::ConvParams params = {
    .padding_type = tflite::PaddingType::kSame, 
    .padding_values = {.width = 0, .height = 0},
    .stride_width = 1,
    .stride_height = 1,
    .dilation_width_factor = 1,
    .dilation_height_factor = 1,
    .input_offset = 128,  
    .weights_offset = 0,
    .output_offset = -128,
    .quantized_activation_min = -128,
    .quantized_activation_max = 127,
};

static tflite::RuntimeShape input_shape_;
static tflite::RuntimeShape filter_shape_;
static tflite::RuntimeShape bias_shape_;
static tflite::RuntimeShape output_shape_;

// === 暴露给 Python 的变量 (保持不变) ===
int32_t input_shape[4] __attribute__((section(".data")));
int32_t filter_shape[4] __attribute__((section(".data"))); 
int32_t bias_shape[1] __attribute__((section(".data")));
int32_t output_shape[4] __attribute__((section(".data")));

int32_t padding_width __attribute__((section(".data"))) = 0;
int32_t padding_height __attribute__((section(".data"))) = 0;

int stride_width __attribute__((section(".data"))) = 1;
int stride_height __attribute__((section(".data"))) = 1;

int8_t input_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));
int8_t output_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));
int8_t filter_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));
int32_t bias_data[kMaxChannels] __attribute__((section(".data"), aligned(16)));

int32_t output_multiplier[kMaxChannels] __attribute__((section(".data"), aligned(16)));
int32_t output_shift[kMaxChannels] __attribute__((section(".data"), aligned(16)));

void prep() {
  input_shape_.ReplaceWith(4, input_shape);
  filter_shape_.ReplaceWith(4, filter_shape);
  bias_shape_.ReplaceWith(1, bias_shape);
  output_shape_.ReplaceWith(4, output_shape);
  
  params.stride_width = stride_width;
  params.stride_height = stride_height;
  
  params.padding_values.width = padding_width;
  params.padding_values.height = padding_height;

  for(size_t i=0; i<kMaxChannels; ++i) {
      output_multiplier[i] = 1215836872; 
      output_shift[i] = -7;              
  }
}

extern "C" {
__attribute__((used, retain)) void run_ref() {
  coralnpu_v2::opt::litert_micro::ConvPerChannel(
      params, output_multiplier, output_shift, input_shape_, input_data,
      filter_shape_, filter_data, bias_shape_, bias_data, output_shape_,
      output_data);
}

// 2. 【核心修改】将原本的 RVV Conv 替换为你自定义的 MXU Conv
__attribute__((used, retain)) void run_optimized() {
  coralnpu_v2::opt::litert_micro::MxuConvPerChannel(
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
