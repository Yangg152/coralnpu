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

#include "sw/opt/litert-micro/conv.h"

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/rvv_opt.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/conv.h" 

#ifdef USE_TFLM_COMPRESSION
#error "USE_TFLM_COMPRESSION is not supported"
#endif  // USE_TFLM_COMPRESSION

namespace coralnpu_v2::opt::litert_micro {

using tflite::ConvParams;
using tflite::kConvBiasTensor;
using tflite::kConvInputTensor;
using tflite::kConvOutputTensor;
using tflite::kConvWeightsTensor;
using tflite::OpDataConv;
using tflite::RuntimeShape;
using tflite::micro::GetEvalInput;
using tflite::micro::GetEvalOutput;
using tflite::micro::GetOptionalTensorData;
using tflite::micro::GetTensorData;
using tflite::micro::GetTensorShape;

namespace {

// =========================================================
// 1. 内存管理辅助工具
// =========================================================

struct AlignedFree {
  void operator()(void* ptr) const { std::free(ptr); }
};

template <typename T>
using aligned_array = std::unique_ptr<T[], AlignedFree>;

template <typename T>
aligned_array<T> make_aligned_array(size_t alignment, size_t nmemb) {
  void* ptr = aligned_alloc(alignment, sizeof(T) * nmemb);
  return aligned_array<T>(reinterpret_cast<T*>(ptr));
}

// =========================================================
// 2. 量化核心辅助函数 (内联)
// =========================================================
// 将 int32 累加器转换为 int8 输出
inline __attribute__((always_inline)) vint8m1_t QuantizeResult_m4(
    vint32m4_t acc, 
    vint32m4_t v_mult, 
    vuint32m4_t v_lshift, 
    vuint32m4_t v_rshift,
    int32_t output_offset,
    int32_t output_min,
    int32_t output_max,
    size_t vl) {
    
    constexpr uint32_t vxrm = 0; // Round-to-nearest-up

    // Pipeline: Left Shift -> MultiplyHigh -> Right Shift -> Add Offset
    acc = __riscv_vsll_vv_i32m4(acc, v_lshift, vl);
    acc = __riscv_vsmul_vv_i32m4(acc, v_mult, vxrm, vl);
    acc = __riscv_vssra_vv_i32m4(acc, v_rshift, vxrm, vl);
    acc = __riscv_vadd_vx_i32m4(acc, output_offset, vl);
    
    // Narrowing: int32 -> int16 -> int8
    vint16m2_t acc_16 = __riscv_vnclip_wx_i16m2(acc, 0, vxrm, vl);
    vint8m1_t acc_8 = __riscv_vnclip_wx_i8m1(acc_16, 0, vxrm, vl);
    
    // Clamp
    acc_8 = __riscv_vmax_vx_i8m1(acc_8, (int8_t)output_min, vl);
    acc_8 = __riscv_vmin_vx_i8m1(acc_8, (int8_t)output_max, vl);
    
    return acc_8;
}

// =========================================================
// 3. 极致优化的 1x1 卷积 (Weight Stationary)
// =========================================================
void Conv1x1PerChannelRVV_Optimized(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

  const int input_depth = input_shape.Dims(3);
  const int output_depth = output_shape.Dims(3);
  const int batches = input_shape.Dims(0);
  const int num_pixels = batches * input_shape.Dims(1) * input_shape.Dims(2);

  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_min = params.quantized_activation_min;
  const int32_t output_max = params.quantized_activation_max;

  // 预处理 Shift 参数
  auto lshift_data = make_aligned_array<uint8_t>(16, output_depth);
  auto rshift_data = make_aligned_array<uint8_t>(16, output_depth);
  if (!lshift_data || !rshift_data) return;
  PrepareShiftParams(lshift_data.get(), rshift_data.get(), output_shift, output_depth);

  // --------------------------------------------------------
  // Loop 1: Output Channel Blocking (Weight Stationary)
  // --------------------------------------------------------
  int out_c = 0;
  size_t out_c_rem = output_depth;

  while (out_c_rem > 0) {
    // 使用 m4 以平衡寄存器压力 (32 regs: 4 acc + 1 weight + 3 quant params + temp = ~10-12 regs used)
    const size_t vl = __riscv_vsetvl_e32m4(out_c_rem);

    // 1. 预加载 Quantization 参数
    vint32m4_t v_mult = __riscv_vle32_v_i32m4(output_multiplier + out_c, vl);
    vuint8m1_t v_lshift_8 = __riscv_vle8_v_u8m1(lshift_data.get() + out_c, vl);
    vuint8m1_t v_rshift_8 = __riscv_vle8_v_u8m1(rshift_data.get() + out_c, vl);
    vuint32m4_t v_lshift = __riscv_vzext_vf4_u32m4(v_lshift_8, vl);
    vuint32m4_t v_rshift = __riscv_vzext_vf4_u32m4(v_rshift_8, vl);

    // 2. 预加载 Bias
    vint32m4_t bias_vec;
    if (bias_data) {
      bias_vec = __riscv_vle32_v_i32m4(bias_data + out_c, vl);
    } else {
      bias_vec = __riscv_vmv_v_x_i32m4(0, vl);
    }

    // 权重基地址
    const int8_t* w_ptr_base = filter_data + out_c * input_depth;
    const ptrdiff_t w_stride = input_depth * sizeof(int8_t);

    // ------------------------------------------------------
    // Loop 2: Pixels (Unrolled x4)
    // ------------------------------------------------------
    int p = 0;
    const int p_loop_end = num_pixels - 4;

    for (; p <= p_loop_end; p += 4) {
      // 初始化 4 个 Accumulators
      vint32m4_t acc0 = bias_vec;
      vint32m4_t acc1 = bias_vec;
      vint32m4_t acc2 = bias_vec;
      vint32m4_t acc3 = bias_vec;

      const int8_t* in_ptr0 = input_data + (p + 0) * input_depth;
      const int8_t* in_ptr1 = input_data + (p + 1) * input_depth;
      const int8_t* in_ptr2 = input_data + (p + 2) * input_depth;
      const int8_t* in_ptr3 = input_data + (p + 3) * input_depth;
      const int8_t* w_ptr = w_ptr_base;

      // Inner Loop: Input Channels
      for (int ic = 0; ic < input_depth; ++ic) {
        // [关键优化] 权重加载：对于 4 个像素，只加载 1 次！
        // TFLite Weight: [Out, 1, 1, In] -> Stride access required across Out
        vint8m1_t w_8 = __riscv_vlse8_v_i8m1(w_ptr + ic, w_stride, vl);
        vint16m2_t w_16 = __riscv_vsext_vf2_i16m2(w_8, vl);

        // 加载 4 个像素的输入 (Scalar Load)
        int16_t in0 = (int16_t)in_ptr0[ic] + input_offset;
        int16_t in1 = (int16_t)in_ptr1[ic] + input_offset;
        int16_t in2 = (int16_t)in_ptr2[ic] + input_offset;
        int16_t in3 = (int16_t)in_ptr3[ic] + input_offset;

        // MAC: Vector += Scalar * Vector
        acc0 = __riscv_vwmacc_vx_i32m4(acc0, in0, w_16, vl);
        acc1 = __riscv_vwmacc_vx_i32m4(acc1, in1, w_16, vl);
        acc2 = __riscv_vwmacc_vx_i32m4(acc2, in2, w_16, vl);
        acc3 = __riscv_vwmacc_vx_i32m4(acc3, in3, w_16, vl);
      }

      // Quantize & Store
      vint8m1_t out0 = QuantizeResult_m4(acc0, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl);
      vint8m1_t out1 = QuantizeResult_m4(acc1, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl);
      vint8m1_t out2 = QuantizeResult_m4(acc2, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl);
      vint8m1_t out3 = QuantizeResult_m4(acc3, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl);

      __riscv_vse8_v_i8m1(output_data + (p + 0) * output_depth + out_c, out0, vl);
      __riscv_vse8_v_i8m1(output_data + (p + 1) * output_depth + out_c, out1, vl);
      __riscv_vse8_v_i8m1(output_data + (p + 2) * output_depth + out_c, out2, vl);
      __riscv_vse8_v_i8m1(output_data + (p + 3) * output_depth + out_c, out3, vl);
    }

    // 处理剩余像素 (Cleanup)
    for (; p < num_pixels; ++p) {
      vint32m4_t acc = bias_vec;
      const int8_t* in_ptr = input_data + p * input_depth;
      const int8_t* w_ptr = w_ptr_base;

      for (int ic = 0; ic < input_depth; ++ic) {
        vint8m1_t w_8 = __riscv_vlse8_v_i8m1(w_ptr + ic, w_stride, vl);
        int16_t in_val = (int16_t)in_ptr[ic] + input_offset;
        acc = __riscv_vwmacc_vx_i32m4(acc, in_val, __riscv_vsext_vf2_i16m2(w_8, vl), vl);
      }
      
      vint8m1_t out = QuantizeResult_m4(acc, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl);
      __riscv_vse8_v_i8m1(output_data + p * output_depth + out_c, out, vl);
    }

    out_c += vl;
    out_c_rem -= vl;
  }
}

// =========================================================
// 4. 改进的通用卷积 Kernel (2x Unroll + Pointer Optimization)
// =========================================================
void ConvGeneralPerChannelRVV_Optimized(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);
  const int batches = input_shape.Dims(0);
  
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int stride_h = params.stride_height;
  const int stride_w = params.stride_width;
  const int pad_h = params.padding_values.height;
  const int pad_w = params.padding_values.width;

  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_min = params.quantized_activation_min;
  const int32_t output_max = params.quantized_activation_max;

  auto lshift_data = make_aligned_array<uint8_t>(16, output_depth);
  auto rshift_data = make_aligned_array<uint8_t>(16, output_depth);
  if (!lshift_data || !rshift_data) return;
  PrepareShiftParams(lshift_data.get(), rshift_data.get(), output_shift, output_depth);

  // Filter stride for jumping to the next output channel
  const ptrdiff_t w_stride_oc = filter_height * filter_width * input_depth * sizeof(int8_t);

  // --------------------------------------------------------
  // Outer Loop: Output Channel Blocking
  // --------------------------------------------------------
  int out_c = 0;
  size_t out_c_rem = output_depth;

  while (out_c_rem > 0) {
    // 使用 m4，如果发现寄存器溢出严重，可尝试改为 m2
    const size_t vl = __riscv_vsetvl_e32m4(out_c_rem);

    // 1. Preload Quant Params & Bias
    vint32m4_t v_mult = __riscv_vle32_v_i32m4(output_multiplier + out_c, vl);
    vuint8m1_t v_lshift_8 = __riscv_vle8_v_u8m1(lshift_data.get() + out_c, vl);
    vuint8m1_t v_rshift_8 = __riscv_vle8_v_u8m1(rshift_data.get() + out_c, vl);
    vuint32m4_t v_lshift = __riscv_vzext_vf4_u32m4(v_lshift_8, vl);
    vuint32m4_t v_rshift = __riscv_vzext_vf4_u32m4(v_rshift_8, vl);

    vint32m4_t bias_vec;
    if (bias_data) bias_vec = __riscv_vle32_v_i32m4(bias_data + out_c, vl);
    else bias_vec = __riscv_vmv_v_x_i32m4(0, vl);

    // Current block weight base
    const int8_t* w_base_block = filter_data + out_c * (filter_height * filter_width * input_depth);

    // ------------------------------------------------------
    // Loop Pixels
    // ------------------------------------------------------
    for (int b = 0; b < batches; ++b) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        
        const int in_y_origin = out_y * stride_h - pad_h;
        
        // Output Row Base Pointer
        int8_t* out_row_ptr = output_data + ((b * output_height + out_y) * output_width) * output_depth + out_c;

        // [优化] 展开 Output Width 循环 (2x Unroll)
        // 这样可以复用加载到寄存器的权重，减少 50% 的权重加载指令
        int out_x = 0;
        for (; out_x <= output_width - 2; out_x += 2) {
            
            vint32m4_t acc0 = bias_vec;
            vint32m4_t acc1 = bias_vec;

            const int in_x_origin_0 = out_x * stride_w - pad_w;
            const int in_x_origin_1 = (out_x + 1) * stride_w - pad_w;

            // Kernel Loop
            for (int ky = 0; ky < filter_height; ++ky) {
                const int in_y = in_y_origin + ky;
                // 纵向边界检查 (对于两个像素都是一样的)
                if (in_y < 0 || in_y >= input_height) continue;

                // 预计算 Row Pointer
                const int8_t* in_row_ptr = input_data + (b * input_height + in_y) * input_width * input_depth;

                for (int kx = 0; kx < filter_width; ++kx) {
                    const int in_x_0 = in_x_origin_0 + kx;
                    const int in_x_1 = in_x_origin_1 + kx;

                    // 检查两个像素的横向边界
                    bool valid0 = (in_x_0 >= 0 && in_x_0 < input_width);
                    bool valid1 = (in_x_1 >= 0 && in_x_1 < input_width);

                    if (!valid0 && !valid1) continue;

                    // 指针计算优化
                    const int8_t* in_pixel_0 = in_row_ptr + in_x_0 * input_depth;
                    const int8_t* in_pixel_1 = in_row_ptr + in_x_1 * input_depth;
                    
                    // Weight offset for this kernel position
                    int w_offset_base = (ky * filter_width + kx) * input_depth;

                    // Input Channel Loop
                    for (int ic = 0; ic < input_depth; ++ic) {
                        // 1. 加载权重 (昂贵操作：跨步加载) - 现在两个像素共享这一次加载!
                        // 注意：这里仍然是瓶颈，但频率减半了
                        vint8m1_t w_8 = __riscv_vlse8_v_i8m1(w_base_block + w_offset_base + ic, w_stride_oc, vl);
                        vint16m2_t w_16 = __riscv_vsext_vf2_i16m2(w_8, vl);

                        // 2. 像素 0 累加
                        if (valid0) {
                            int16_t in_val0 = (int16_t)in_pixel_0[ic] + input_offset;
                            acc0 = __riscv_vwmacc_vx_i32m4(acc0, in_val0, w_16, vl);
                        }

                        // 3. 像素 1 累加
                        if (valid1) {
                            int16_t in_val1 = (int16_t)in_pixel_1[ic] + input_offset;
                            acc1 = __riscv_vwmacc_vx_i32m4(acc1, in_val1, w_16, vl);
                        }
                    } // end ic
                } // end kx
            } // end ky

            // Store Pixel 0
            vint8m1_t out0 = QuantizeResult_m4(acc0, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl);
            __riscv_vse8_v_i8m1(out_row_ptr + 0 * output_depth, out0, vl); // offset is relative to current out_c ptr

            // Store Pixel 1
            vint8m1_t out1 = QuantizeResult_m4(acc1, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl);
            __riscv_vse8_v_i8m1(out_row_ptr + 1 * output_depth, out1, vl);

            out_row_ptr += 2 * output_depth;
        }

        // 处理剩余的单个像素 (Remainder Loop)
        for (; out_x < output_width; ++out_x) {
             vint32m4_t acc = bias_vec;
             const int in_x_origin = out_x * stride_w - pad_w;

             for (int ky = 0; ky < filter_height; ++ky) {
                const int in_y = in_y_origin + ky;
                if (in_y < 0 || in_y >= input_height) continue;
                
                const int8_t* in_row_ptr = input_data + (b * input_height + in_y) * input_width * input_depth;

                for (int kx = 0; kx < filter_width; ++kx) {
                    const int in_x = in_x_origin + kx;
                    if (in_x < 0 || in_x >= input_width) continue;

                    const int8_t* in_pixel = in_row_ptr + in_x * input_depth;
                    int w_offset_base = (ky * filter_width + kx) * input_depth;

                    for (int ic = 0; ic < input_depth; ++ic) {
                        vint8m1_t w_8 = __riscv_vlse8_v_i8m1(w_base_block + w_offset_base + ic, w_stride_oc, vl);
                        int16_t in_val = (int16_t)in_pixel[ic] + input_offset;
                        acc = __riscv_vwmacc_vx_i32m4(acc, in_val, __riscv_vsext_vf2_i16m2(w_8, vl), vl);
                    }
                }
             }
             vint8m1_t out = QuantizeResult_m4(acc, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl);
             __riscv_vse8_v_i8m1(out_row_ptr, out, vl);
             out_row_ptr += output_depth;
        }
      }
    }

    out_c += vl;
    out_c_rem -= vl;
  }
}

}  // namespace

// =========================================================
// 5. 主逻辑 Dispatcher
// =========================================================

void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
    
  const int filter_h = filter_shape.Dims(1);
  const int filter_w = filter_shape.Dims(2);
  const int stride_h = params.stride_height;
  const int stride_w = params.stride_width;
  const int dilation_h = params.dilation_height_factor;
  const int dilation_w = params.dilation_width_factor;

  // Case 1: 1x1 Pointwise (极致优化)
  if (filter_h == 1 && filter_w == 1 && 
      stride_h == 1 && stride_w == 1 &&
      dilation_h == 1 && dilation_w == 1) {
     Conv1x1PerChannelRVV_Optimized(params, output_multiplier, output_shift, input_shape, 
         input_data, filter_shape, filter_data, bias_shape, bias_data, 
         output_shape, output_data);
  } 
  // Case 2: 通用情况 (包括 Layer 0)
  else if (dilation_h == 1 && dilation_w == 1) {
     ConvGeneralPerChannelRVV_Optimized(params, output_multiplier, output_shift, input_shape, 
         input_data, filter_shape, filter_data, bias_shape, bias_data, 
         output_shape, output_data);
  }
  // Case 3: Fallback
  else {
     tflite::reference_integer_ops::ConvPerChannel(
         params, output_multiplier, output_shift, input_shape, input_data,
         filter_shape, filter_data, bias_shape, bias_data, output_shape,
         output_data);
  }
}

TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
  // ... (保持原有的 Boilerplate 代码不变) ...
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));
  const auto* params =
      reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (tflite::micro::GetTensorData<int32_t>(
           GetEvalInput(context, node, kConvBiasTensor)) != nullptr)
          ? GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      GetEvalOutput(context, node, kConvOutputTensor);

  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
     MicroPrintf("Type %s (%d) not supported by optimized Conv.", TfLiteTypeGetName(input->type), input->type);
     return kTfLiteError;
  }

  tflite::ConvParams op_params;
  op_params.padding_type = tflite::PaddingType::kSame; 
  op_params.padding_values.width = data.padding.width;
  op_params.padding_values.height = data.padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  
  op_params.input_offset = -data.input_zero_point;
  op_params.weights_offset = -data.filter_zero_point;
  op_params.output_offset = data.output_zero_point;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;

  ConvPerChannel(
      op_params, data.per_channel_output_multiplier,
      data.per_channel_output_shift, GetTensorShape(input),
      GetTensorData<int8_t>(input),
      GetTensorShape(filter),
      GetTensorData<int8_t>(filter),
      GetTensorShape(bias),
      GetOptionalTensorData<int32_t>(bias),
      GetTensorShape(output),
      GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

TFLMRegistration Register_CONV_2D() {
  auto registration = tflite::Register_CONV_2D();
  registration.invoke = ConvEval;
  return registration;
}

}  // namespace coralnpu_v2::opt::litert_micro
