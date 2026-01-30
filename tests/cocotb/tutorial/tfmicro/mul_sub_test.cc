// Copyright 2025 Google LLC
#include "sw/opt/litert-micro/mul.h"
#include "sw/opt/litert-micro/sub.h"

#include <cstdint>
#include <vector>

#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h" 

namespace {
// 最大 Tensor 大小 (根据需要调整)
constexpr size_t kMaxTensorSize = 128 * 128 * 4; 
}  // namespace

// ==========================================
// 1. Python 接口变量 (位于 .data 段)
// ==========================================

// Shape 数组
int32_t input1_shape[4] __attribute__((section(".data")));
int32_t input2_shape[4] __attribute__((section(".data")));
int32_t output_shape[4] __attribute__((section(".data")));

// 量化参数
int32_t input1_zero_point __attribute__((section(".data"))) = 0;
int32_t input2_zero_point __attribute__((section(".data"))) = 0;
int32_t output_zero_point __attribute__((section(".data"))) = 0;
int32_t output_multiplier __attribute__((section(".data"))) = 0;
int32_t output_shift      __attribute__((section(".data"))) = 0;
int32_t output_min        __attribute__((section(".data"))) = -128;
int32_t output_max        __attribute__((section(".data"))) = 127;

// 数据 Buffer
// 注意：保持 int8_t 类型，与 sub.h 中的重载匹配
int8_t input1_data[kMaxTensorSize] __attribute__((section(".extdata"), aligned(16)));
int8_t input2_data[kMaxTensorSize] __attribute__((section(".extdata"), aligned(16)));
int8_t output_data[kMaxTensorSize] __attribute__((section(".extdata"), aligned(16)));

// ==========================================
// 2. 内部结构体
// ==========================================
static tflite::ArithmeticParams params;
static tflite::RuntimeShape input1_shape_;
static tflite::RuntimeShape input2_shape_;
static tflite::RuntimeShape output_shape_;

// ==========================================
// 3. 准备函数
// ==========================================
void prep() {
  // 转换 Shape
  input1_shape_.ReplaceWith(4, input1_shape);
  input2_shape_.ReplaceWith(4, input2_shape);
  output_shape_.ReplaceWith(4, output_shape);

  // 转换 Params
  params.input1_offset = -input1_zero_point;
  params.input2_offset = -input2_zero_point;
  params.output_offset = output_zero_point;
  params.output_multiplier = output_multiplier;
  params.output_shift = output_shift;
  params.quantized_activation_min = output_min;
  params.quantized_activation_max = output_max;
  
  params.float_activation_min = 0.0f;
  params.float_activation_max = 0.0f;
}

extern "C" {

// ---------------------------------------------------------
// MUL (乘法) 实现
// ---------------------------------------------------------

__attribute__((used, retain)) void run_ref_mul() {
  // 1. 简单的 Shape 对比
  bool need_broadcast = false;
  int dims = input1_shape_.DimensionsCount();
  
  if (input2_shape_.DimensionsCount() != dims) {
    need_broadcast = true;
  } else {
    for (int i = 0; i < dims; ++i) {
      if (input1_shape_.Dims(i) != input2_shape_.Dims(i)) {
        need_broadcast = true;
        break;
      }
    }
  }

  // 2. 根据判断结果分发
  if (need_broadcast) {
    // 广播模式：调用 integer_ops 下的 Broadcast 方法
    tflite::reference_integer_ops::BroadcastMul4DSlow(
        params, 
        input1_shape_, input1_data,
        input2_shape_, input2_data,
        output_shape_, output_data);
  } else {
    // 元素对元素模式
    tflite::reference_integer_ops::Mul(
        params, 
        input1_shape_, input1_data,
        input2_shape_, input2_data,
        output_shape_, output_data);
  }
}


__attribute__((used, retain)) void run_opt_mul() {
  int flat_size = output_shape_.FlatSize();
  int in1_size = input1_shape_.FlatSize();
  int in2_size = input2_shape_.FlatSize();

  if (in1_size == flat_size && in2_size == flat_size) {
    coralnpu_v2::opt::litert_micro::MulQuantizedElementWise(
        input1_data, input2_data, output_data, flat_size, params);
  } else if (in2_size == 1) {
    coralnpu_v2::opt::litert_micro::MulQuantizedBroadcastScalar(
        input1_data, input2_data[0], output_data, flat_size, params);
  } else if (in1_size == 1) {
    // 乘法交换律
    tflite::ArithmeticParams swapped_params = params;
    swapped_params.input1_offset = params.input2_offset;
    swapped_params.input2_offset = params.input1_offset;
    coralnpu_v2::opt::litert_micro::MulQuantizedBroadcastScalar(
        input2_data, input1_data[0], output_data, flat_size, swapped_params);
  }
}

// ---------------------------------------------------------
// SUB (减法) 实现
// ---------------------------------------------------------

__attribute__((used, retain)) void run_ref_sub() {
  // 1. 判断是否需要广播 (逻辑与 Mul 相同)
  bool need_broadcast = false;
  int dims = input1_shape_.DimensionsCount();
  
  if (input2_shape_.DimensionsCount() != dims) {
    need_broadcast = true;
  } else {
    for (int i = 0; i < dims; ++i) {
      if (input1_shape_.Dims(i) != input2_shape_.Dims(i)) {
        need_broadcast = true;
        break;
      }
    }
  }

  // 2. 手动分发
  if (need_broadcast) {
    // [关键点]：广播模式调用 BroadcastQuantSubSlow
    // 这个函数在 sub.h 第 280 行定义，支持 int8_t 量化广播
    tflite::reference_ops::BroadcastQuantSubSlow(
        params, 
        input1_shape_, input1_data,
        input2_shape_, input2_data,
        output_shape_, output_data);
  } else {
    // Element-wise 模式继续调用 Sub
    // 此时它会命中 sub.h 第 344 行的 int8_t 重载，效率较高
    tflite::reference_ops::Sub(
        params, 
        input1_shape_, input1_data,
        input2_shape_, input2_data,
        output_shape_, output_data);
  }
}


__attribute__((used, retain)) void run_opt_sub() {
  int flat_size = output_shape_.FlatSize();
  int in1_size = input1_shape_.FlatSize();
  int in2_size = input2_shape_.FlatSize();

  if (in1_size == flat_size && in2_size == flat_size) {
    // 1. Element-wise
    coralnpu_v2::opt::litert_micro::SubQuantizedElementWise(
        input1_data, input2_data, output_data, flat_size, params);
  } else if (in2_size == 1) {
    // 2. Broadcast: Vector - Scalar
    coralnpu_v2::opt::litert_micro::SubQuantizedBroadcastInput2Scalar(
        input1_data, input2_data[0], output_data, flat_size, params);
  } else if (in1_size == 1) {
    // 3. Broadcast: Scalar - Vector
    coralnpu_v2::opt::litert_micro::SubQuantizedBroadcastInput1Scalar(
        input1_data[0], input2_data, output_data, flat_size, params);
  }
}

} // extern "C"

// 函数指针 (默认指向 mul opt)
void (*impl)() __attribute__((section(".data"))) = run_opt_mul;

int main(void) {
  prep();
  impl();
  return 0;
}
