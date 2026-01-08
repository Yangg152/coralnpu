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

// 引用 TFLite 命名空间中的类型，避免重复定义结构体
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
// 1. 内存管理辅助工具 (与 depthwise_conv.cc 保持一致)
// =========================================================

struct AlignedFree {
  void operator()(void* ptr) const { std::free(ptr); }
};

template <typename T>
using aligned_array = std::unique_ptr<T[], AlignedFree>;

template <typename T>
aligned_array<T> make_aligned_array(size_t alignment, size_t nmemb) {
  // 注意：如果你的编译环境不支持 aligned_alloc (C11)，请替换为 memalign 或 posix_memalign
  void* ptr = aligned_alloc(alignment, sizeof(T) * nmemb);
  return aligned_array<T>(reinterpret_cast<T*>(ptr));
}

// =========================================================
// 2. RVV 优化的 1x1 卷积 Kernel
// =========================================================

// 针对 1x1 卷积的优化实现
// 策略：外层循环遍历像素点，内层循环向量化处理 Output Channels (C_out)。
// 权重访问：由于 TFLite 权重布局为 [C_out, C_in]，向量化 C_out 需要使用 stride load (跨度为 C_in)。
void Conv1x1PerChannelRVV(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

  const int batches = input_shape.Dims(0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = output_shape.Dims(3);
  
  const int num_pixels = batches * input_height * input_width;

  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // 使用 aligned_array 替代栈上数组，防止栈溢出
  auto lshift_data = make_aligned_array<uint8_t>(16, output_depth);
  auto rshift_data = make_aligned_array<uint8_t>(16, output_depth);
  
  if (!lshift_data || !rshift_data) {
    // 内存分配失败，直接返回（在实际应用中可能需要错误处理机制）
    return;
  }

  // 预计算 Shift 参数 (假设 accumulator_util.h 中有此函数)
  PrepareShiftParams(lshift_data.get(), rshift_data.get(), output_shift, output_depth);

  constexpr uint32_t vxrm = 0; // Rounding mode: round-to-nearest-up

  for (int p = 0; p < num_pixels; ++p) {
    const int8_t* in_ptr_base = input_data + p * input_depth;
    int8_t* out_ptr = output_data + p * output_depth;

    int out_c = 0;
    size_t out_c_rem = output_depth;

    while (out_c_rem > 0) {
      // 设置向量长度，以 output_depth 为维度
      const size_t vl = __riscv_vsetvl_e32m8(out_c_rem);

      // 1. 初始化 Accumulator
      vint32m8_t acc;
      if (bias_data) {
        acc = __riscv_vle32_v_i32m8(bias_data + out_c, vl);
      } else {
        acc = __riscv_vmv_v_x_i32m8(0, vl);
      }

      // 2. 权重指针设置
      // 权重是 [C_out, C_in]。对于当前的 out_c 块，起始地址是 filter_data + out_c * input_depth
      // 当我们在向量寄存器中处理 [out_c, out_c + vl] 时，
      // W[out_c, 0] 和 W[out_c+1, 0] 之间的内存距离是 input_depth。
      const int8_t* w_ptr = filter_data + out_c * input_depth;
      const ptrdiff_t w_stride = input_depth * sizeof(int8_t); 

      int in_c = 0;
      const int8_t* in_ptr = in_ptr_base;

      // 3. 计算循环 (Unrolling = 4)
      // 这里的策略是：广播输入的 int8 (scalar)，加载一列权重的 int8 (vector stride load)，进行 MAC。
      for (; in_c <= input_depth - 4; in_c += 4) {
        // 加载 4 个输入值 (Scalar)
        int8_t in_val_0 = in_ptr[0];
        int8_t in_val_1 = in_ptr[1];
        int8_t in_val_2 = in_ptr[2];
        int8_t in_val_3 = in_ptr[3];
        in_ptr += 4;

        // 应用 Input Offset
        int16_t in_16_0 = (int16_t)in_val_0 + (int16_t)input_offset;
        int16_t in_16_1 = (int16_t)in_val_1 + (int16_t)input_offset;
        int16_t in_16_2 = (int16_t)in_val_2 + (int16_t)input_offset;
        int16_t in_16_3 = (int16_t)in_val_3 + (int16_t)input_offset;

        // 加载 4 列权重 (Vector Strided Load)
        // 使用 vlse8 (vector load strided element 8-bit)
        vint8m2_t w_0 = __riscv_vlse8_v_i8m2(w_ptr + 0, w_stride, vl);
        vint8m2_t w_1 = __riscv_vlse8_v_i8m2(w_ptr + 1, w_stride, vl);
        vint8m2_t w_2 = __riscv_vlse8_v_i8m2(w_ptr + 2, w_stride, vl);
        vint8m2_t w_3 = __riscv_vlse8_v_i8m2(w_ptr + 3, w_stride, vl);
        w_ptr += 4; // 指针向 input_depth 方向移动 4

        // 扩展权重并 MAC (Vector * Scalar)
        // vwmacc.vx: acc += vector * scalar
        vint16m4_t w_16_0 = __riscv_vsext_vf2_i16m4(w_0, vl);
        acc = __riscv_vwmacc_vx_i32m8(acc, in_16_0, w_16_0, vl);

        vint16m4_t w_16_1 = __riscv_vsext_vf2_i16m4(w_1, vl);
        acc = __riscv_vwmacc_vx_i32m8(acc, in_16_1, w_16_1, vl);

        vint16m4_t w_16_2 = __riscv_vsext_vf2_i16m4(w_2, vl);
        acc = __riscv_vwmacc_vx_i32m8(acc, in_16_2, w_16_2, vl);

        vint16m4_t w_16_3 = __riscv_vsext_vf2_i16m4(w_3, vl);
        acc = __riscv_vwmacc_vx_i32m8(acc, in_16_3, w_16_3, vl);
      }

      // Cleanup Loop (处理剩余的 Input Channels)
      for (; in_c < input_depth; ++in_c) {
        int8_t in_val = *in_ptr++;
        int16_t in_val_16 = (int16_t)in_val + (int16_t)input_offset;

        vint8m2_t w_val_8 = __riscv_vlse8_v_i8m2(w_ptr, w_stride, vl);
        w_ptr++; 
        
        vint16m4_t w_val_16 = __riscv_vsext_vf2_i16m4(w_val_8, vl);
        acc = __riscv_vwmacc_vx_i32m8(acc, in_val_16, w_val_16, vl);
      }

      // 4. 量化 Pipeline (Requantization)
      vint32m8_t v_mult = __riscv_vle32_v_i32m8(output_multiplier + out_c, vl);
      vuint8m2_t v_lshift_8 = __riscv_vle8_v_u8m2(lshift_data.get() + out_c, vl);
      vuint8m2_t v_rshift_8 = __riscv_vle8_v_u8m2(rshift_data.get() + out_c, vl);
      
      // 扩展 Shift 参数到 32位 以便进行向量位移
      vuint32m8_t v_lshift_32 = __riscv_vzext_vf4_u32m8(v_lshift_8, vl);
      vuint32m8_t v_rshift_32 = __riscv_vzext_vf4_u32m8(v_rshift_8, vl);

      // 左移 -> 乘法 -> 右移 (rounding)
      acc = __riscv_vsll_vv_i32m8(acc, v_lshift_32, vl);
      acc = __riscv_vsmul_vv_i32m8(acc, v_mult, vxrm, vl);
      acc = __riscv_vssra_vv_i32m8(acc, v_rshift_32, vxrm, vl);
      
      // Output Offset
      acc = __riscv_vadd_vx_i32m8(acc, output_offset, vl);
      
      // 窄化 (Narrowing) 到 int8 并进行 Clamp
      vint16m4_t acc_16 = __riscv_vnclip_wx_i16m4(acc, 0, vxrm, vl);
      vint8m2_t acc_8 = __riscv_vnclip_wx_i8m2(acc_16, 0, vxrm, vl);
      
      acc_8 = __riscv_vmax_vx_i8m2(acc_8, (int8_t)output_activation_min, vl);
      acc_8 = __riscv_vmin_vx_i8m2(acc_8, (int8_t)output_activation_max, vl);

      // 存储结果
      __riscv_vse8_v_i8m2(out_ptr + out_c, acc_8, vl);

      out_c += vl;
      out_c_rem -= vl;
    }
  }
}

}  // namespace

// =========================================================
// 3. 主逻辑 Dispatcher
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

  // 检查是否为 1x1 卷积：Filter 1x1, Stride 1x1, Dilation 1x1
  if (filter_h == 1 && filter_w == 1 && 
      stride_h == 1 && stride_w == 1 &&
      dilation_h == 1 && dilation_w == 1) {
     Conv1x1PerChannelRVV(params, output_multiplier, output_shift, input_shape, 
         input_data, filter_shape, filter_data, bias_shape, bias_data, 
         output_shape, output_data);
  } else {
     // 其他情况（3x3 或 通用尺寸），调用 TFLite 参考实现
     // 如果未来实现了 3x3 优化，在此处添加分支
     tflite::reference_integer_ops::ConvPerChannel(
         params, output_multiplier, output_shift, input_shape, input_data,
         filter_shape, filter_data, bias_shape, bias_data, output_shape,
         output_data);
  }
}

TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
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

  // 检查数据类型支持：仅支持 Int8
  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
     MicroPrintf("Type %s (%d) not supported by optimized Conv.", TfLiteTypeGetName(input->type), input->type);
     return kTfLiteError;
  }

  // 重构 ConvParams
  // 注意：这里手动构建 ConvParams 是为了兼容性，确保参数传递正确。
  // 在原始代码中通常使用 ConvParamsQuantized(params, data)，但其位于 reference 实现中可能无法内联。
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
