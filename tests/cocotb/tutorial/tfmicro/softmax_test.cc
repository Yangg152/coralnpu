// Copyright 2025 Google LLC
#include "sw/opt/litert-micro/softmax.h"

#include <cstdint>
#include <vector>

#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace {
// 最大 Tensor 大小
constexpr size_t kMaxTensorSize = 128 * 128 * 4; 
}  // namespace

// ==========================================
// 1. Python 接口变量 (位于 .data 段)
// ==========================================

// Shape
int32_t shape_dims[4] __attribute__((section(".data")));
int32_t dims_count __attribute__((section(".data"))) = 4;

// Softmax Params
int32_t input_multiplier __attribute__((section(".data"))) = 0;
int32_t input_left_shift __attribute__((section(".data"))) = 0;
int32_t diff_min         __attribute__((section(".data"))) = 0;
float   beta             __attribute__((section(".data"))) = 1.0f; // 只用于 Float 参考，Int8 用上面的

// Data Buffers
int8_t input_data[kMaxTensorSize] __attribute__((section(".extdata"), aligned(16)));
int8_t output_data[kMaxTensorSize] __attribute__((section(".extdata"), aligned(16)));

// ==========================================
// 2. 内部结构体
// ==========================================
static tflite::SoftmaxParams params;
static tflite::RuntimeShape shape_;

// ==========================================
// 3. 准备函数
// ==========================================
void prep() {
  shape_.ReplaceWith(dims_count, shape_dims);

  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  params.diff_min = diff_min;
  params.beta = beta;
  params.scale = 1.0f; // Scale 通常用于 Float
  params.zero_point = 0;
}

extern "C" {

// ---------------------------------------------------------
// Reference Implementation
// ---------------------------------------------------------
__attribute__((used, retain)) void run_ref_softmax() {
  // 使用 TFLite 官方 Reference 
  // 注意：Input/Output Shape 相同
  tflite::reference_ops::Softmax(
      params, 
      shape_, input_data,
      shape_, output_data);
}

// ---------------------------------------------------------
// Optimized Implementation
// ---------------------------------------------------------
__attribute__((used, retain)) void run_opt_softmax() {
  coralnpu_v2::opt::litert_micro::SoftmaxQuantized(
      params,
      shape_, input_data,
      shape_, output_data);
}

} // extern "C"

// 函数指针
void (*impl)() __attribute__((section(".data"))) = run_opt_softmax;

int main(void) {
  prep();
  impl();
  return 0;
}
