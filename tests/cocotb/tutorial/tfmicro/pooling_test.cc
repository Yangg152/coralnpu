// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>

// Optimized kernel under test
#include "sw/opt/litert-micro/pooling.h"
// TFLite reference implementations
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace {
// Maximum tensor size: 128×128×256 channels
constexpr size_t kMaxTensorSize = 128 * 128 * 256;
// Maximum spatial dimension (used for filter-row pointer arrays)
constexpr int kMaxDims = 4;
}  // namespace

// ============================================================
// Python-visible interface variables (placed in .data section)
// ============================================================

// Input / output tensor shapes: [N, H, W, C]
int32_t input_shape[kMaxDims]  __attribute__((section(".data")));
int32_t output_shape[kMaxDims] __attribute__((section(".data")));

// Pooling parameters
int32_t stride_height  __attribute__((section(".data"))) = 1;
int32_t stride_width   __attribute__((section(".data"))) = 1;
int32_t filter_height  __attribute__((section(".data"))) = 2;
int32_t filter_width   __attribute__((section(".data"))) = 2;
int32_t pad_height     __attribute__((section(".data"))) = 0;
int32_t pad_width      __attribute__((section(".data"))) = 0;

// Activation clipping (int8 range, stored as int32 for easy Python write)
int32_t activation_min __attribute__((section(".data"))) = -128;
int32_t activation_max __attribute__((section(".data"))) =  127;

// Data buffers
int8_t input_data[kMaxTensorSize]
    __attribute__((section(".extdata"), aligned(16)));
int8_t output_data[kMaxTensorSize]
    __attribute__((section(".extdata"), aligned(16)));

// ============================================================
// Static shape objects (rebuilt in prep())
// ============================================================
static tflite::RuntimeShape input_shape_;
static tflite::RuntimeShape output_shape_;

// ============================================================
// Prepare: reconstruct RuntimeShape objects from Python arrays
// ============================================================
void prep() {
  input_shape_.ReplaceWith(kMaxDims, input_shape);
  output_shape_.ReplaceWith(kMaxDims, output_shape);
}

// ============================================================
// Helper: fill a PoolParams from the global parameter variables
// ============================================================
static tflite::PoolParams MakePoolParams() {
  tflite::PoolParams p;
  p.stride_height = stride_height;
  p.stride_width  = stride_width;
  p.filter_height = filter_height;
  p.filter_width  = filter_width;
  p.padding_values.height = pad_height;
  p.padding_values.width  = pad_width;
  p.quantized_activation_min = static_cast<int8_t>(activation_min);
  p.quantized_activation_max = static_cast<int8_t>(activation_max);
  return p;
}

extern "C" {

// ------------------------------------------------------------
// AveragePool — Reference implementation
// ------------------------------------------------------------
__attribute__((used, retain)) void run_ref_avgpool() {
  const tflite::PoolParams params = MakePoolParams();
  tflite::reference_integer_ops::AveragePool(
      params, input_shape_, input_data, output_shape_, output_data);
}


// ------------------------------------------------------------
// AveragePool — RVV-optimized implementation
// ------------------------------------------------------------
__attribute__((used, retain)) void run_opt_avgpool() {
  const tflite::PoolParams params = MakePoolParams();
  coralnpu_v2::opt::litert_micro::AveragePoolInt8RVV(
      params,
      input_shape_,
      input_data,
      output_shape_,
      output_data);
}

// ------------------------------------------------------------
// MaxPool — Reference implementation
// ------------------------------------------------------------
__attribute__((used, retain)) void run_ref_maxpool() {
  const tflite::PoolParams params = MakePoolParams();
  tflite::reference_integer_ops::MaxPool(
      params, input_shape_, input_data, output_shape_, output_data);
}

// ------------------------------------------------------------
// MaxPool — RVV-optimized implementation
// ------------------------------------------------------------
__attribute__((used, retain)) void run_opt_maxpool() {
  const tflite::PoolParams params = MakePoolParams();
  coralnpu_v2::opt::litert_micro::MaxPoolInt8RVV(
      params,
      input_shape_,
      input_data,
      output_shape_,
      output_data);
}

}  // extern "C"

// Function pointer — Python switches this between ref / opt variants
void (*impl)() __attribute__((section(".data"))) = run_ref_avgpool;

int main(void) {
  prep();
  impl();
  return 0;
}
