// Copyright 2025 Google LLC
#include <cstdint>
#include <vector>

// [关键] 引入优化算子的定义
#include "sw/opt/litert-micro/mean.h"

// 引入 Reference 实现
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace {
// 最大 Tensor 大小
constexpr size_t kMaxTensorSize = 128 * 128 * 4; 
// 最大维度数
constexpr int kMaxDims = 4;
}  // namespace

// ==========================================
// 1. Python 接口变量 (位于 .data 段)
// ==========================================

// Shape 数组
int32_t input_shape[kMaxDims] __attribute__((section(".data")));
int32_t output_shape[kMaxDims] __attribute__((section(".data")));
int32_t axis_data[kMaxDims]   __attribute__((section(".data"))); 

// 维度计数
int32_t input_dims_count __attribute__((section(".data"))) = 4;
int32_t output_dims_count __attribute__((section(".data"))) = 4;
int32_t axis_count        __attribute__((section(".data"))) = 0;
int32_t keep_dims         __attribute__((section(".data"))) = 0;

// 量化参数
int32_t input_zero_point  __attribute__((section(".data"))) = 0;
int32_t output_zero_point __attribute__((section(".data"))) = 0;
int32_t output_multiplier __attribute__((section(".data"))) = 0;
int32_t output_shift      __attribute__((section(".data"))) = 0;

// 数据 Buffer
int8_t input_data[kMaxTensorSize] __attribute__((section(".extdata"), aligned(16)));
int8_t output_data[kMaxTensorSize] __attribute__((section(".extdata"), aligned(16)));

// ==========================================
// 2. Reference 实现需要的临时 Buffer
// ==========================================
int32_t temp_sum_buffer[kMaxTensorSize] __attribute__((section(".extdata"), aligned(16)));
int32_t temp_index[kMaxDims] __attribute__((section(".extdata"), aligned(16)));
int32_t resolved_axis[kMaxDims] __attribute__((section(".extdata"), aligned(16)));

static tflite::RuntimeShape input_shape_;
static tflite::RuntimeShape output_shape_;

// ==========================================
// 3. 准备函数
// ==========================================
void prep() {
  input_shape_.ReplaceWith(input_dims_count, input_shape);
  output_shape_.ReplaceWith(output_dims_count, output_shape);
}

extern "C" {

// ---------------------------------------------------------
// MEAN (Reference) 实现
// ---------------------------------------------------------

__attribute__((used, retain)) void run_ref_mean() {
  // [修复] 使用 reinterpret_cast 强制转换指针类型，解决编译错误
  tflite::reference_ops::QuantizedMeanOrSum<int8_t, int32_t>(
      input_data,
      input_zero_point,
      reinterpret_cast<const int*>(input_shape_.DimsData()), 
      input_shape_.DimensionsCount(),
      output_data,
      output_multiplier,
      output_shift,
      output_zero_point,
      reinterpret_cast<const int*>(output_shape_.DimsData()),
      output_shape_.DimensionsCount(),
      reinterpret_cast<const int*>(axis_data),
      axis_count,
      (keep_dims != 0),
      reinterpret_cast<int*>(temp_index),
      reinterpret_cast<int*>(resolved_axis),
      temp_sum_buffer,
      false // compute_sum = false (Mean)
  );
}

// ---------------------------------------------------------
// MEAN (Optimized) 实现
// ---------------------------------------------------------

__attribute__((used, retain)) void run_opt_mean() {
  int32_t batches = input_shape_.Dims(0);
  int32_t height = input_shape_.Dims(1);
  int32_t width = input_shape_.Dims(2);
  int32_t channels = input_shape_.Dims(3);

  // [修复] 调用优化函数，并传入正确的全局变量
  coralnpu_v2::opt::litert_micro::MeanGlobalPoolingQuantizedRVV(
      input_data,
      output_data,
      batches,
      height,
      width,
      channels,
      input_zero_point,
      output_zero_point,
      output_multiplier, // 使用全局变量 output_multiplier
      output_shift       // 使用全局变量 output_shift
  );
}

} // extern "C"

// 函数指针 (默认指向 ref)
void (*impl)() __attribute__((section(".data"))) = run_ref_mean;

int main(void) {
  prep();
  impl();
  return 0;
}
