// tests/cocotb/tutorial/tfmicro/mean_test.cc

#include "sw/opt/litert-micro/mean.h"

#include <cstdint>
#include <algorithm>
#include <vector>

#include "tensorflow/lite/kernels/internal/types.h"

namespace {
// 最大支持的 Tensor 大小
constexpr size_t kMaxTensorSize = 128 * 1024; 
constexpr int kMaxDims = 4;

// ============================================================================
// 1. 本地辅助函数：定点乘法
// ============================================================================
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
  using int64 = std::int64_t;
  int64 val = static_cast<int64>(x) * static_cast<int64>(quantized_multiplier);
  int right_shift = -shift;
  if (right_shift < 0) {
      return static_cast<int32_t>(val << (-right_shift));
  }
  int64 one_half = 1LL << (right_shift - 1);
  val += one_half;
  val >>= right_shift;
  return static_cast<int32_t>(val);
}

// ============================================================================
// 2. 本地参考实现：Global Average Pooling (Int8)
// ============================================================================
void ReferenceMeanGlobal(
    const tflite::MeanParams& params,
    int32_t multiplier,
    int32_t shift,
    const tflite::RuntimeShape& input_shape,
    const int8_t* input_data,
    int32_t input_zero_point,
    const tflite::RuntimeShape& output_shape,
    int8_t* output_data,
    int32_t output_zero_point) {

  const int batches = input_shape.Dims(0);
  const int height = input_shape.Dims(1);
  const int width = input_shape.Dims(2);
  const int channels = input_shape.Dims(3);
  
  const int num_elements_in_axis = height * width;
  const int32_t offset_correction = -num_elements_in_axis * input_zero_point;

  for (int b = 0; b < batches; ++b) {
    for (int c = 0; c < channels; ++c) {
      int32_t acc = 0;
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = ((b * height + h) * width + w) * channels + c;
          acc += input_data[idx];
        }
      }
      acc += offset_correction;
      acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
      acc += output_zero_point;
      acc = std::max(acc, (int32_t)-128);
      acc = std::min(acc, (int32_t)127);
      output_data[b * channels + c] = static_cast<int8_t>(acc);
    }
  }
}

}  // namespace

static tflite::MeanParams params;

// 用于 Python 握手的标志位
// volatile 防止编译器优化读取操作
volatile int32_t g_sync_flag __attribute__((section(".data"))) = 0;

// 全局变量定义（无需显式初始化，Python 会在握手阶段填充）
int32_t input_shape_dims[kMaxDims] __attribute__((section(".data")));
int32_t output_shape_dims[kMaxDims] __attribute__((section(".data")));

int32_t axis_data[2] __attribute__((section(".data"))); 
int32_t axis_count __attribute__((section(".data")));

int32_t input_zero_point __attribute__((section(".data")));
int32_t output_zero_point __attribute__((section(".data")));
int32_t multiplier __attribute__((section(".data")));
int32_t shift __attribute__((section(".data")));

int8_t input_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));
int8_t output_data[kMaxTensorSize] __attribute__((section(".data"), aligned(16)));

static tflite::RuntimeShape input_shape_;
static tflite::RuntimeShape output_shape_;

void prep() {
  input_shape_.ReplaceWith(4, input_shape_dims);
  output_shape_.ReplaceWith(4, output_shape_dims);

  params.axis_count = axis_count;
  for (int i = 0; i < axis_count; ++i) {
    params.axis[i] = axis_data[i];
  }
}

extern "C" {

__attribute__((used, retain)) void run_ref() {
  ReferenceMeanGlobal(
      params,
      multiplier,
      shift,
      input_shape_,
      input_data,
      input_zero_point,
      output_shape_,
      output_data,
      output_zero_point);
}

__attribute__((used, retain)) void run_optimized() {
  int32_t batches = input_shape_.Dims(0);
  int32_t height = input_shape_.Dims(1);
  int32_t width = input_shape_.Dims(2);
  int32_t channels = input_shape_.Dims(3);

  coralnpu_v2::opt::litert_micro::MeanGlobalPoolingQuantizedRVV(
      input_data,
      output_data,
      batches,
      height,
      width,
      channels,
      input_zero_point,
      output_zero_point,
      multiplier,
      shift);
}

} // extern "C"

// 函数指针，初始值会被 CRT 设置，但随后会被 Python 在握手时覆盖
void (*impl)() __attribute__((section(".data"))) = run_optimized;

int main(void) {
  // 1. 设置标志位，通知 Python：CRT 初始化已完成，内存安全
  g_sync_flag = 1;

  // 2. 等待 Python 写入数据
  // Python 写入完成后会将 g_sync_flag 设为 2
  while (g_sync_flag == 1) {
    // Spin wait
  }

  // 3. 执行准备工作（解析 Shape）和核心逻辑
  prep();
  impl();
  return 0;
}
