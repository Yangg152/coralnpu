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

#include "sw/opt/litert-micro/pooling.h"

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/builtin_ops.h"

namespace coralnpu_v2::opt::litert_micro {

using tflite::PoolParams;
using tflite::RuntimeShape;

// =========================================================================
// Internal helpers
// =========================================================================

struct FilterBounds {
  int start;
  int end;
};

static inline FilterBounds ComputeFilterBounds(int out_coord, int stride,
                                               int pad, int filter_size,
                                               int input_size) {
  const int in_origin = out_coord * stride - pad;
  return {std::max(0, -in_origin),
          std::min(filter_size, input_size - in_origin)};
}

// =========================================================================
// AveragePool — int8, NHWC layout
// =========================================================================

static void AveragePoolOutputPixelRVV(const int8_t* in_base,
                                      int8_t* out_pixel,
                                      int num_channels,
                                      int filter_count,
                                      int input_width, int num_input_channels,
                                      const int8_t* filter_row_ptrs[],
                                      int num_valid_rows,
                                      int filter_x_start, int filter_x_end,
                                      int8_t act_min, int8_t act_max) {
  (void)in_base;
  (void)input_width;
  (void)num_input_channels;

  const int32_t rounding = filter_count / 2;

  size_t c_remain = static_cast<size_t>(num_channels);
  size_t c_idx    = 0;

  while (c_remain > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(c_remain);

    vint32m8_t v_sum = __riscv_vmv_v_x_i32m8(0, vl);

    for (int fy = 0; fy < num_valid_rows; ++fy) {
      const int8_t* row_ptr = filter_row_ptrs[fy];
      for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
        const int8_t* px = row_ptr + fx * num_channels + c_idx;
        vint8m2_t  v8  = __riscv_vle8_v_i8m2(px, vl);
        vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
        v_sum = __riscv_vwadd_wv_i32m8(v_sum, v16, vl);
      }
    }

    v_sum = __riscv_vadd_vx_i32m8(v_sum, rounding, vl);
    v_sum = __riscv_vdiv_vx_i32m8(v_sum, filter_count, vl);
    v_sum = __riscv_vmax_vx_i32m8(v_sum, static_cast<int32_t>(act_min), vl);
    v_sum = __riscv_vmin_vx_i32m8(v_sum, static_cast<int32_t>(act_max), vl);

    constexpr uint32_t vxrm = 0;
    vint16m4_t v16 = __riscv_vnclip_wx_i16m4(v_sum, 0, vxrm, vl);
    vint8m2_t  v8  = __riscv_vnclip_wx_i8m2(v16, 0, vxrm, vl);
    __riscv_vse8_v_i8m2(out_pixel + c_idx, v8, vl);

    c_idx    += vl;
    c_remain -= vl;
  }
}

static void AveragePoolGeneralRVV(const PoolParams& params,
                                  const RuntimeShape& input_shape,
                                  const int8_t* input_data,
                                  const RuntimeShape& output_shape,
                                  int8_t* output_data,
                                  int out_y_start, int out_y_end,
                                  int out_x_start, int out_x_end) {
  const int batches      = input_shape.Dims(0);
  const int in_h         = input_shape.Dims(1);
  const int in_w         = input_shape.Dims(2);
  const int depth        = input_shape.Dims(3);
  const int out_w        = output_shape.Dims(2);

  const int stride_h    = params.stride_height;
  const int stride_w    = params.stride_width;
  const int filter_h    = params.filter_height;
  const int filter_w    = params.filter_width;
  const int pad_h       = params.padding_values.height;
  const int pad_w       = params.padding_values.width;
  const int8_t act_min  = static_cast<int8_t>(params.quantized_activation_min);
  const int8_t act_max  = static_cast<int8_t>(params.quantized_activation_max);

  const int8_t* row_ptrs[128];

  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* batch_in = input_data + batch * in_h * in_w * depth;

    for (int out_y = out_y_start; out_y < out_y_end; ++out_y) {
      const auto fy_bounds = ComputeFilterBounds(out_y, stride_h, pad_h,
                                                 filter_h, in_h);
      const int in_y_origin = out_y * stride_h - pad_h;

      for (int out_x = out_x_start; out_x < out_x_end; ++out_x) {
        const auto fx_bounds = ComputeFilterBounds(out_x, stride_w, pad_w,
                                                   filter_w, in_w);
        const int in_x_origin = out_x * stride_w - pad_w;
        const int filter_count = (fy_bounds.end - fy_bounds.start) *
                                 (fx_bounds.end - fx_bounds.start);
        if (filter_count == 0) continue;

        int num_rows = 0;
        for (int fy = fy_bounds.start; fy < fy_bounds.end; ++fy) {
          const int in_y = in_y_origin + fy;
          row_ptrs[num_rows++] = batch_in + (in_y * in_w + in_x_origin) * depth;
        }

        int8_t* out_pixel = output_data +
            (batch * output_shape.Dims(1) * out_w + out_y * out_w + out_x) *
            depth;

        AveragePoolOutputPixelRVV(nullptr, out_pixel, depth,
                                  filter_count, in_w, depth,
                                  row_ptrs, num_rows,
                                  fx_bounds.start, fx_bounds.end,
                                  act_min, act_max);
      }
    }
  }
}

static void AveragePoolCenterRVV(const PoolParams& params,
                                 const RuntimeShape& input_shape,
                                 const int8_t* input_data,
                                 const RuntimeShape& output_shape,
                                 int8_t* output_data,
                                 int out_y_start, int out_y_end,
                                 int out_x_start, int out_x_end) {
  const int batches     = input_shape.Dims(0);
  const int in_h        = input_shape.Dims(1);
  const int in_w        = input_shape.Dims(2);
  const int depth       = input_shape.Dims(3);
  const int out_h       = output_shape.Dims(1);
  const int out_w       = output_shape.Dims(2);

  const int stride_h    = params.stride_height;
  const int stride_w    = params.stride_width;
  const int filter_h    = params.filter_height;
  const int filter_w    = params.filter_width;
  const int pad_h       = params.padding_values.height;
  const int pad_w       = params.padding_values.width;
  const int8_t act_min  = static_cast<int8_t>(params.quantized_activation_min);
  const int8_t act_max  = static_cast<int8_t>(params.quantized_activation_max);
  const int filter_count = filter_h * filter_w;
  const int32_t rounding = filter_count / 2;

  (void)in_h; (void)out_h;

  const int8_t* row_ptrs[128];

  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* batch_in = input_data + batch * in_h * in_w * depth;

    for (int out_y = out_y_start; out_y < out_y_end; ++out_y) {
      const int in_y_origin = out_y * stride_h - pad_h;

      for (int fy = 0; fy < filter_h; ++fy) {
        row_ptrs[fy] = batch_in + ((in_y_origin + fy) * in_w) * depth;
      }

      for (int out_x = out_x_start; out_x < out_x_end; ++out_x) {
        const int in_x_origin = out_x * stride_w - pad_w;
        int8_t* out_pixel = output_data +
            (batch * output_shape.Dims(1) * out_w + out_y * out_w + out_x) *
            depth;

        const int8_t* shifted_row_ptrs[128];
        for (int fy = 0; fy < filter_h; ++fy) {
          shifted_row_ptrs[fy] = row_ptrs[fy] + in_x_origin * depth;
        }

        size_t c_remain = static_cast<size_t>(depth);
        size_t c_idx    = 0;

        while (c_remain > 0) {
          const size_t vl = __riscv_vsetvl_e32m8(c_remain);
          vint32m8_t v_sum = __riscv_vmv_v_x_i32m8(0, vl);

          for (int fy = 0; fy < filter_h; ++fy) {
            const int8_t* row = shifted_row_ptrs[fy];
            for (int fx = 0; fx < filter_w; ++fx) {
              const int8_t* px = row + fx * depth + c_idx;
              vint8m2_t  v8  = __riscv_vle8_v_i8m2(px, vl);
              vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
              v_sum = __riscv_vwadd_wv_i32m8(v_sum, v16, vl);
            }
          }

          v_sum = __riscv_vadd_vx_i32m8(v_sum, rounding, vl);
          v_sum = __riscv_vdiv_vx_i32m8(v_sum, filter_count, vl);
          v_sum = __riscv_vmax_vx_i32m8(v_sum,
                                        static_cast<int32_t>(act_min), vl);
          v_sum = __riscv_vmin_vx_i32m8(v_sum,
                                        static_cast<int32_t>(act_max), vl);

          constexpr uint32_t vxrm = 0;
          vint16m4_t v16 = __riscv_vnclip_wx_i16m4(v_sum, 0, vxrm, vl);
          vint8m2_t  v8  = __riscv_vnclip_wx_i8m2(v16, 0, vxrm, vl);
          __riscv_vse8_v_i8m2(out_pixel + c_idx, v8, vl);

          c_idx    += vl;
          c_remain -= vl;
        }
      }
    }
  }
}

void AveragePoolInt8RVV(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const int8_t* input_data,
                        const RuntimeShape& output_shape,
                        int8_t* output_data) {
  const int out_h    = output_shape.Dims(1);
  const int out_w    = output_shape.Dims(2);
  const int stride_h = params.stride_height;
  const int stride_w = params.stride_width;
  const int filter_h = params.filter_height;
  const int filter_w = params.filter_width;
  const int pad_h    = params.padding_values.height;
  const int pad_w    = params.padding_values.width;

  auto idiv_ceil = [](int x, int y) { return (x + y - 1) / y; };

  const int out_y_top    = idiv_ceil(pad_h, stride_h);
  const int out_y_bottom = idiv_ceil(out_h + pad_h - filter_h, stride_h);
  const int out_x_left   = idiv_ceil(pad_w, stride_w);
  const int out_x_right  = idiv_ceil(out_w + pad_w - filter_w, stride_w);

  const int cy_top    = std::max(0, std::min(out_y_top,    out_h));
  const int cy_bottom = std::max(cy_top, std::min(out_y_bottom, out_h));
  const int cx_left   = std::max(0, std::min(out_x_left,   out_w));
  const int cx_right  = std::max(cx_left, std::min(out_x_right,  out_w));

  if (cy_top > 0)
    AveragePoolGeneralRVV(params, input_shape, input_data,
                          output_shape, output_data,
                          0, cy_top, 0, out_w);

  if (cx_left > 0)
    AveragePoolGeneralRVV(params, input_shape, input_data,
                          output_shape, output_data,
                          cy_top, cy_bottom, 0, cx_left);

  if (cy_top < cy_bottom && cx_left < cx_right)
    AveragePoolCenterRVV(params, input_shape, input_data,
                         output_shape, output_data,
                         cy_top, cy_bottom, cx_left, cx_right);

  if (cx_right < out_w)
    AveragePoolGeneralRVV(params, input_shape, input_data,
                          output_shape, output_data,
                          cy_top, cy_bottom, cx_right, out_w);

  if (cy_bottom < out_h)
    AveragePoolGeneralRVV(params, input_shape, input_data,
                          output_shape, output_data,
                          cy_bottom, out_h, 0, out_w);
}

// =========================================================================
// MaxPool — int8, NHWC layout
// =========================================================================

static void MaxPoolGeneralRVV(const PoolParams& params,
                               const RuntimeShape& input_shape,
                               const int8_t* input_data,
                               const RuntimeShape& output_shape,
                               int8_t* output_data,
                               int out_y_start, int out_y_end,
                               int out_x_start, int out_x_end) {
  const int batches  = input_shape.Dims(0);
  const int in_h     = input_shape.Dims(1);
  const int in_w     = input_shape.Dims(2);
  const int depth    = input_shape.Dims(3);
  const int out_h    = output_shape.Dims(1);
  const int out_w    = output_shape.Dims(2);

  const int stride_h  = params.stride_height;
  const int stride_w  = params.stride_width;
  const int filter_h  = params.filter_height;
  const int filter_w  = params.filter_width;
  const int pad_h     = params.padding_values.height;
  const int pad_w     = params.padding_values.width;
  const int8_t act_min = static_cast<int8_t>(params.quantized_activation_min);
  const int8_t act_max = static_cast<int8_t>(params.quantized_activation_max);

  (void)out_h;

  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* batch_in = input_data + batch * in_h * in_w * depth;

    for (int out_y = out_y_start; out_y < out_y_end; ++out_y) {
      const auto fy_bounds = ComputeFilterBounds(out_y, stride_h, pad_h,
                                                 filter_h, in_h);
      const int in_y_origin = out_y * stride_h - pad_h;

      for (int out_x = out_x_start; out_x < out_x_end; ++out_x) {
        const auto fx_bounds = ComputeFilterBounds(out_x, stride_w, pad_w,
                                                   filter_w, in_w);
        const int in_x_origin = out_x * stride_w - pad_w;

        int8_t* out_pixel = output_data +
            (batch * output_shape.Dims(1) * out_w + out_y * out_w + out_x) *
            depth;

        size_t c_remain = static_cast<size_t>(depth);
        size_t c_idx    = 0;

        while (c_remain > 0) {
          const size_t vl = __riscv_vsetvl_e8m2(c_remain);
          vint8m2_t v_max = __riscv_vmv_v_x_i8m2(-128, vl);

          for (int fy = fy_bounds.start; fy < fy_bounds.end; ++fy) {
            const int in_y = in_y_origin + fy;
            for (int fx = fx_bounds.start; fx < fx_bounds.end; ++fx) {
              const int in_x = in_x_origin + fx;
              const int8_t* px = batch_in +
                  (in_y * in_w + in_x) * depth + c_idx;
              vint8m2_t v_val = __riscv_vle8_v_i8m2(px, vl);
              v_max = __riscv_vmax_vv_i8m2(v_max, v_val, vl);
            }
          }

          v_max = __riscv_vmax_vx_i8m2(v_max, act_min, vl);
          v_max = __riscv_vmin_vx_i8m2(v_max, act_max, vl);
          __riscv_vse8_v_i8m2(out_pixel + c_idx, v_max, vl);

          c_idx    += vl;
          c_remain -= vl;
        }
      }
    }
  }
}

static void MaxPoolCenterRVV(const PoolParams& params,
                              const RuntimeShape& input_shape,
                              const int8_t* input_data,
                              const RuntimeShape& output_shape,
                              int8_t* output_data,
                              int out_y_start, int out_y_end,
                              int out_x_start, int out_x_end) {
  const int batches  = input_shape.Dims(0);
  const int in_h     = input_shape.Dims(1);
  const int in_w     = input_shape.Dims(2);
  const int depth    = input_shape.Dims(3);
  const int out_h    = output_shape.Dims(1);
  const int out_w    = output_shape.Dims(2);

  const int stride_h  = params.stride_height;
  const int stride_w  = params.stride_width;
  const int filter_h  = params.filter_height;
  const int filter_w  = params.filter_width;
  const int pad_h     = params.padding_values.height;
  const int pad_w     = params.padding_values.width;
  const int8_t act_min = static_cast<int8_t>(params.quantized_activation_min);
  const int8_t act_max = static_cast<int8_t>(params.quantized_activation_max);

  (void)in_h; (void)out_h;

  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* batch_in = input_data + batch * in_h * in_w * depth;

    for (int out_y = out_y_start; out_y < out_y_end; ++out_y) {
      const int in_y_origin = out_y * stride_h - pad_h;

      for (int out_x = out_x_start; out_x < out_x_end; ++out_x) {
        const int in_x_origin = out_x * stride_w - pad_w;

        int8_t* out_pixel = output_data +
            (batch * output_shape.Dims(1) * out_w + out_y * out_w + out_x) *
            depth;

        size_t c_remain = static_cast<size_t>(depth);
        size_t c_idx    = 0;

        while (c_remain > 0) {
          const size_t vl = __riscv_vsetvl_e8m2(c_remain);
          vint8m2_t v_max = __riscv_vmv_v_x_i8m2(-128, vl);

          for (int fy = 0; fy < filter_h; ++fy) {
            const int in_y = in_y_origin + fy;
            const int8_t* row = batch_in + (in_y * in_w + in_x_origin) * depth;
            for (int fx = 0; fx < filter_w; ++fx) {
              vint8m2_t v_val = __riscv_vle8_v_i8m2(
                  row + fx * depth + c_idx, vl);
              v_max = __riscv_vmax_vv_i8m2(v_max, v_val, vl);
            }
          }

          v_max = __riscv_vmax_vx_i8m2(v_max, act_min, vl);
          v_max = __riscv_vmin_vx_i8m2(v_max, act_max, vl);
          __riscv_vse8_v_i8m2(out_pixel + c_idx, v_max, vl);

          c_idx    += vl;
          c_remain -= vl;
        }
      }
    }
  }
}

void MaxPoolInt8RVV(const PoolParams& params,
                    const RuntimeShape& input_shape,
                    const int8_t* input_data,
                    const RuntimeShape& output_shape,
                    int8_t* output_data) {
  const int out_h    = output_shape.Dims(1);
  const int out_w    = output_shape.Dims(2);
  const int stride_h = params.stride_height;
  const int stride_w = params.stride_width;
  const int filter_h = params.filter_height;
  const int filter_w = params.filter_width;
  const int pad_h    = params.padding_values.height;
  const int pad_w    = params.padding_values.width;

  auto idiv_ceil = [](int x, int y) { return (x + y - 1) / y; };

  const int out_y_top    = idiv_ceil(pad_h, stride_h);
  const int out_y_bottom = idiv_ceil(out_h + pad_h - filter_h, stride_h);
  const int out_x_left   = idiv_ceil(pad_w, stride_w);
  const int out_x_right  = idiv_ceil(out_w + pad_w - filter_w, stride_w);

  const int cy_top    = std::max(0, std::min(out_y_top,    out_h));
  const int cy_bottom = std::max(cy_top, std::min(out_y_bottom, out_h));
  const int cx_left   = std::max(0, std::min(out_x_left,   out_w));
  const int cx_right  = std::max(cx_left, std::min(out_x_right,  out_w));

  if (cy_top > 0)
    MaxPoolGeneralRVV(params, input_shape, input_data,
                      output_shape, output_data,
                      0, cy_top, 0, out_w);

  if (cx_left > 0)
    MaxPoolGeneralRVV(params, input_shape, input_data,
                      output_shape, output_data,
                      cy_top, cy_bottom, 0, cx_left);

  if (cy_top < cy_bottom && cx_left < cx_right)
    MaxPoolCenterRVV(params, input_shape, input_data,
                     output_shape, output_data,
                     cy_top, cy_bottom, cx_left, cx_right);

  if (cx_right < out_w)
    MaxPoolGeneralRVV(params, input_shape, input_data,
                      output_shape, output_data,
                      cy_top, cy_bottom, cx_right, out_w);

  if (cy_bottom < out_h)
    MaxPoolGeneralRVV(params, input_shape, input_data,
                      output_shape, output_data,
                      cy_bottom, out_h, 0, out_w);
}

// =========================================================================
// TFLite Micro Kernel Interface
// =========================================================================

namespace {

// ---------------------------------------------------------------------------
// Init: allocate persistent OpDataPooling storage.
// ---------------------------------------------------------------------------
void* PoolingInit(TfLiteContext* context, const char* buffer, size_t length) {
  (void)buffer;
  (void)length;
  return context->AllocatePersistentBuffer(context,
                                           sizeof(tflite::OpDataPooling));
}

// ---------------------------------------------------------------------------
// Prepare: inline all padding + activation range computation.
// Does NOT call tflite::CalculateOpDataPooling to avoid ODR symbol conflict
// with the official TFLM pooling.cc compilation unit.
// Uses MicroContext::AllocateTempInputTensor / AllocateTempOutputTensor as
// required by the TFLM Prepare contract (context->GetTensor is not valid here).
// ---------------------------------------------------------------------------
TfLiteStatus PoolingPrepare(TfLiteContext* context, TfLiteNode* node) {
  if (node->builtin_data == nullptr) {
    MicroPrintf("PoolingPrepare: builtin_data is null");
    return kTfLiteError;
  }
  if (node->user_data == nullptr) {
    MicroPrintf("PoolingPrepare: user_data is null (Init not called?)");
    return kTfLiteError;
  }

  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  auto* data   = static_cast<tflite::OpDataPooling*>(node->user_data);

  // AllocateTempInputTensor / AllocateTempOutputTensor are the correct TFLM
  // Prepare-phase accessors; context->GetTensor must not be used in Prepare.
  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, /*index=*/0);
  if (input == nullptr) {
    MicroPrintf("PoolingPrepare: AllocateTempInputTensor failed");
    return kTfLiteError;
  }
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, /*index=*/0);
  if (output == nullptr) {
    micro_context->DeallocateTempTfLiteTensor(input);
    MicroPrintf("PoolingPrepare: AllocateTempOutputTensor failed");
    return kTfLiteError;
  }

  // --- Padding (inlined from CalculateOpDataPooling) -----------------------
  int out_height_unused, out_width_unused;
  data->padding = tflite::ComputePaddingHeightWidth(
      params->stride_height,
      params->stride_width,
      /*dilation_rate_height=*/1,
      /*dilation_rate_width=*/1,
      /*in_height=*/input->dims->data[1],
      /*in_width=*/input->dims->data[2],
      params->filter_height,
      params->filter_width,
      params->padding,
      &out_height_unused,
      &out_width_unused);

  // --- Quantization consistency check + activation range -------------------
  TfLiteStatus status = kTfLiteOk;

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    const double scale_diff = static_cast<double>(
        std::abs(input->params.scale - output->params.scale));
    if (scale_diff > 1.0e-6) {
      MicroPrintf("PoolingPrepare: input/output scale mismatch (%f vs %f)",
                  static_cast<double>(input->params.scale),
                  static_cast<double>(output->params.scale));
      status = kTfLiteError;
    } else if (input->params.zero_point != output->params.zero_point) {
      MicroPrintf("PoolingPrepare: input/output zero_point mismatch (%d vs %d)",
                  input->params.zero_point, output->params.zero_point);
      status = kTfLiteError;
    } else {
      status = tflite::CalculateActivationRangeQuantized(
          context, params->activation, output,
          &data->activation_min, &data->activation_max);
    }
  } else if (input->type == kTfLiteFloat32) {
    tflite::CalculateActivationRange(params->activation,
                                     &data->activation_min_f32,
                                     &data->activation_max_f32);
  } else {
    MicroPrintf("PoolingPrepare: unsupported type %s (%d)",
                TfLiteTypeGetName(input->type), input->type);
    status = kTfLiteError;
  }

  // Always deallocate temp tensors, even on error.
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);

  return status;
}

// ---------------------------------------------------------------------------
// AveragePool Eval
// ---------------------------------------------------------------------------
TfLiteStatus AveragePoolEval(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    MicroPrintf("AveragePoolEval: op_data is null");
    return kTfLiteError;
  }
  const auto* op_data =
      reinterpret_cast<const tflite::OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, 0);

  const auto& params =
      *reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);

  tflite::PoolParams op_params;
  op_params.stride_height            = params.stride_height;
  op_params.stride_width             = params.stride_width;
  op_params.filter_height            = params.filter_height;
  op_params.filter_width             = params.filter_width;
  op_params.padding_values.height    = op_data->padding.height;
  op_params.padding_values.width     = op_data->padding.width;
  op_params.quantized_activation_min = op_data->activation_min;
  op_params.quantized_activation_max = op_data->activation_max;

  if (input->type == kTfLiteInt8) {
    AveragePoolInt8RVV(op_params,
                       tflite::micro::GetTensorShape(input),
                       tflite::micro::GetTensorData<int8_t>(input),
                       tflite::micro::GetTensorShape(output),
                       tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
  }

  MicroPrintf("AveragePoolEval: unsupported input type %d", input->type);
  return kTfLiteError;
}

// ---------------------------------------------------------------------------
// MaxPool Eval
// ---------------------------------------------------------------------------
TfLiteStatus MaxPoolEval(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    MicroPrintf("MaxPoolEval: op_data is null");
    return kTfLiteError;
  }
  const auto* op_data =
      reinterpret_cast<const tflite::OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, 0);

  const auto& params =
      *reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);

  tflite::PoolParams op_params;
  op_params.stride_height            = params.stride_height;
  op_params.stride_width             = params.stride_width;
  op_params.filter_height            = params.filter_height;
  op_params.filter_width             = params.filter_width;
  op_params.padding_values.height    = op_data->padding.height;
  op_params.padding_values.width     = op_data->padding.width;
  op_params.quantized_activation_min = op_data->activation_min;
  op_params.quantized_activation_max = op_data->activation_max;

  if (input->type == kTfLiteInt8) {
    MaxPoolInt8RVV(op_params,
                   tflite::micro::GetTensorShape(input),
                   tflite::micro::GetTensorData<int8_t>(input),
                   tflite::micro::GetTensorShape(output),
                   tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
  }

  MicroPrintf("MaxPoolEval: unsupported input type %d", input->type);
  return kTfLiteError;
}

}  // namespace

TFLMRegistration Register_AVERAGE_POOL_2D() {
  return tflite::micro::RegisterOp(PoolingInit, PoolingPrepare,
                                   AveragePoolEval);
}

TFLMRegistration Register_MAX_POOL_2D() {
  return tflite::micro::RegisterOp(PoolingInit, PoolingPrepare, MaxPoolEval);
}

}  // namespace coralnpu_v2::opt::litert_micro
