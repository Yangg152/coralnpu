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

#ifndef SW_OPT_LITERT_MICRO_POOLING_H_
#define SW_OPT_LITERT_MICRO_POOLING_H_

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace coralnpu_v2::opt::litert_micro {

// -----------------------------------------------------------------------
// AveragePool — RVV-optimized int8 average pooling.
//
// Quantization contract (same as TFLite reference int8 AveragePool):
//   - input_zero_point is subtracted before accumulation so we work in
//     the "de-quantized integer" domain.
//   - The sum is divided by filter_count via integer rounding
//     (round-half-up: (sum + filter_count/2) / filter_count).
//   - The result is then clipped to [quantized_activation_min,
//     quantized_activation_max] and stored as int8.
//   - No per-layer multiplier/shift is needed when input_scale ==
//     output_scale (the common TFLite AveragePool case).
// -----------------------------------------------------------------------
void AveragePoolInt8RVV(const tflite::PoolParams& params,
                        const tflite::RuntimeShape& input_shape,
                        const int8_t* input_data,
                        const tflite::RuntimeShape& output_shape,
                        int8_t* output_data);

// -----------------------------------------------------------------------
// MaxPool — RVV-optimized int8 max pooling.
//
// Scans the filter window and keeps the element-wise maximum across
// channels, then clips to [quantized_activation_min,
// quantized_activation_max].
// -----------------------------------------------------------------------
void MaxPoolInt8RVV(const tflite::PoolParams& params,
                    const tflite::RuntimeShape& input_shape,
                    const int8_t* input_data,
                    const tflite::RuntimeShape& output_shape,
                    int8_t* output_data);

// TFLite Micro kernel registrations.
TFLMRegistration Register_AVERAGE_POOL_2D();
TFLMRegistration Register_MAX_POOL_2D();

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_POOLING_H_
