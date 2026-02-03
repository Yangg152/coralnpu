#include "sw/opt/litert-micro/conv.h"

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/rvv_opt.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/conv.h" 

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
// 1. 内存管理辅助
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
// 2. 量化核心 (LMUL=4) - 仅用于 Cleanup 循环
// =========================================================
inline __attribute__((always_inline)) vint8m1_t QuantizeResult_m4(
    vint32m4_t acc, vint32m4_t v_mult, vuint32m4_t v_lshift, vuint32m4_t v_rshift,
    int32_t output_offset, int32_t output_min, int32_t output_max, size_t vl) {
    
    constexpr uint32_t vxrm = 0; 
    acc = __riscv_vsll_vv_i32m4(acc, v_lshift, vl);
    acc = __riscv_vsmul_vv_i32m4(acc, v_mult, vxrm, vl);
    acc = __riscv_vssra_vv_i32m4(acc, v_rshift, vxrm, vl);
    acc = __riscv_vadd_vx_i32m4(acc, output_offset, vl);
    
    vint16m2_t acc_16 = __riscv_vnclip_wx_i16m2(acc, 0, vxrm, vl);
    vint8m1_t acc_8 = __riscv_vnclip_wx_i8m1(acc_16, 0, vxrm, vl);
    
    acc_8 = __riscv_vmax_vx_i8m1(acc_8, (int8_t)output_min, vl);
    acc_8 = __riscv_vmin_vx_i8m1(acc_8, (int8_t)output_max, vl);
    return acc_8;
}

// =========================================================
// 3. 量化核心 (LMUL=2) - 用于通用卷积
// =========================================================
inline __attribute__((always_inline)) vint8mf2_t QuantizeResult_m2(
    vint32m2_t acc, vint32m2_t v_mult, vuint32m2_t v_lshift, vuint32m2_t v_rshift,
    int32_t output_offset, int32_t output_min, int32_t output_max, size_t vl) {
    
    constexpr uint32_t vxrm = 0; 
    acc = __riscv_vsll_vv_i32m2(acc, v_lshift, vl);
    acc = __riscv_vsmul_vv_i32m2(acc, v_mult, vxrm, vl);
    acc = __riscv_vssra_vv_i32m2(acc, v_rshift, vxrm, vl);
    acc = __riscv_vadd_vx_i32m2(acc, output_offset, vl);
    
    vint16m1_t acc_16 = __riscv_vnclip_wx_i16m1(acc, 0, vxrm, vl);
    vint8mf2_t acc_8 = __riscv_vnclip_wx_i8mf2(acc_16, 0, vxrm, vl);
    
    acc_8 = __riscv_vmax_vx_i8mf2(acc_8, (int8_t)output_min, vl);
    acc_8 = __riscv_vmin_vx_i8mf2(acc_8, (int8_t)output_max, vl);
    return acc_8;
}

// =========================================================
// 4. 1x1 卷积极限优化 (LMUL=4, Unroll=6 + 延迟量化加载)
// 策略：计算期间不持有量化参数，最大化累加器数量。
// =========================================================
void Conv1x1PerChannelRVV_Optimized(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* __restrict__ input_data, const RuntimeShape& filter_shape,
    const int8_t* __restrict__ filter_data, const RuntimeShape& bias_shape,
    const int32_t* __restrict__ bias_data, const RuntimeShape& output_shape,
    int8_t* __restrict__ output_data) {

  const int input_depth = input_shape.Dims(3);
  const int output_depth = output_shape.Dims(3);
  const int batches = input_shape.Dims(0);
  const int num_pixels = batches * input_shape.Dims(1) * input_shape.Dims(2);

  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_min = params.quantized_activation_min;
  const int32_t output_max = params.quantized_activation_max;

  auto lshift_data = make_aligned_array<uint8_t>(16, output_depth);
  auto rshift_data = make_aligned_array<uint8_t>(16, output_depth);
  if (!lshift_data || !rshift_data) return;
  PrepareShiftParams(lshift_data.get(), rshift_data.get(), output_shift, output_depth);

  int out_c = 0;
  size_t out_c_rem = output_depth;

  while (out_c_rem > 0) {
    // LMUL=4, 最大化向量长度
    const size_t vl = __riscv_vsetvl_e32m4(out_c_rem);

    // [优化点] 不在这里加载 Mult/Shift，节省 12 个寄存器

    vint32m4_t bias_vec;
    if (bias_data) {
      bias_vec = __riscv_vle32_v_i32m4(bias_data + out_c, vl);
    } else {
      bias_vec = __riscv_vmv_v_x_i32m4(0, vl);
    }

    const int8_t* w_ptr_base = filter_data + out_c * input_depth;
    const ptrdiff_t w_stride = input_depth * sizeof(int8_t);

    // Unroll 6 (Pixel)
    // 寄存器分配分析 (Total 32):
    // Accumulators: 6 * m4 = 24 regs (v0-v23)
    // Weights: m2 + m1 = 3 regs (v24-v26)
    // 剩余: 5 regs. 安全。
    int p = 0;
    const int p_loop_end = num_pixels - 6;

    for (; p <= p_loop_end; p += 6) {
      vint32m4_t acc0 = bias_vec;
      vint32m4_t acc1 = bias_vec;
      vint32m4_t acc2 = bias_vec;
      vint32m4_t acc3 = bias_vec;
      vint32m4_t acc4 = bias_vec;
      vint32m4_t acc5 = bias_vec;

      const int8_t* in_ptr0 = input_data + (p + 0) * input_depth;
      const int8_t* in_ptr1 = input_data + (p + 1) * input_depth;
      const int8_t* in_ptr2 = input_data + (p + 2) * input_depth;
      const int8_t* in_ptr3 = input_data + (p + 3) * input_depth;
      const int8_t* in_ptr4 = input_data + (p + 4) * input_depth;
      const int8_t* in_ptr5 = input_data + (p + 5) * input_depth;
      
      const int8_t* w_ptr = w_ptr_base;

      for (int ic = 0; ic < input_depth; ++ic) {
        // Strided Load Weights (开销大，被 6 个像素分摊)
        vint8m1_t w_8 = __riscv_vlse8_v_i8m1(w_ptr + ic, w_stride, vl);
        vint16m2_t w_16 = __riscv_vsext_vf2_i16m2(w_8, vl);

        // Scalar Load Inputs (Scalar + Offset)
        int16_t i0 = (int16_t)in_ptr0[ic] + input_offset;
        int16_t i1 = (int16_t)in_ptr1[ic] + input_offset;
        int16_t i2 = (int16_t)in_ptr2[ic] + input_offset;
        int16_t i3 = (int16_t)in_ptr3[ic] + input_offset;
        int16_t i4 = (int16_t)in_ptr4[ic] + input_offset;
        int16_t i5 = (int16_t)in_ptr5[ic] + input_offset;

        // Widening MAC
        acc0 = __riscv_vwmacc_vx_i32m4(acc0, i0, w_16, vl);
        acc1 = __riscv_vwmacc_vx_i32m4(acc1, i1, w_16, vl);
        acc2 = __riscv_vwmacc_vx_i32m4(acc2, i2, w_16, vl);
        acc3 = __riscv_vwmacc_vx_i32m4(acc3, i3, w_16, vl);
        acc4 = __riscv_vwmacc_vx_i32m4(acc4, i4, w_16, vl);
        acc5 = __riscv_vwmacc_vx_i32m4(acc5, i5, w_16, vl);
      }

      // [Delayed Quantization Pipeline]
      // 此时所有计算已完成，开始分阶段加载参数进行量化，避免寄存器溢出。
      
      // Phase 1: Left Shift
      {
          vuint32m4_t v_lshift = __riscv_vzext_vf4_u32m4(__riscv_vle8_v_u8m1(lshift_data.get() + out_c, vl), vl);
          acc0 = __riscv_vsll_vv_i32m4(acc0, v_lshift, vl);
          acc1 = __riscv_vsll_vv_i32m4(acc1, v_lshift, vl);
          acc2 = __riscv_vsll_vv_i32m4(acc2, v_lshift, vl);
          acc3 = __riscv_vsll_vv_i32m4(acc3, v_lshift, vl);
          acc4 = __riscv_vsll_vv_i32m4(acc4, v_lshift, vl);
          acc5 = __riscv_vsll_vv_i32m4(acc5, v_lshift, vl);
      } // v_lshift is dead here, registers freed

      // Phase 2: Multiply
      {
          vint32m4_t v_mult = __riscv_vle32_v_i32m4(output_multiplier + out_c, vl);
          acc0 = __riscv_vsmul_vv_i32m4(acc0, v_mult, 0, vl);
          acc1 = __riscv_vsmul_vv_i32m4(acc1, v_mult, 0, vl);
          acc2 = __riscv_vsmul_vv_i32m4(acc2, v_mult, 0, vl);
          acc3 = __riscv_vsmul_vv_i32m4(acc3, v_mult, 0, vl);
          acc4 = __riscv_vsmul_vv_i32m4(acc4, v_mult, 0, vl);
          acc5 = __riscv_vsmul_vv_i32m4(acc5, v_mult, 0, vl);
      } // v_mult is dead here

      // Phase 3: Right Shift & Output Store
      {
          vuint32m4_t v_rshift = __riscv_vzext_vf4_u32m4(__riscv_vle8_v_u8m1(rshift_data.get() + out_c, vl), vl);
          
          // Lambda to process remaining steps and store
          auto finish_quant = [&](vint32m4_t& acc, int offset_idx) {
              acc = __riscv_vssra_vv_i32m4(acc, v_rshift, 0, vl);
              acc = __riscv_vadd_vx_i32m4(acc, output_offset, vl);
              vint16m2_t acc_16 = __riscv_vnclip_wx_i16m2(acc, 0, 0, vl);
              vint8m1_t acc_8 = __riscv_vnclip_wx_i8m1(acc_16, 0, 0, vl);
              acc_8 = __riscv_vmax_vx_i8m1(acc_8, (int8_t)output_min, vl);
              acc_8 = __riscv_vmin_vx_i8m1(acc_8, (int8_t)output_max, vl);
              __riscv_vse8_v_i8m1(output_data + (p + offset_idx) * output_depth + out_c, acc_8, vl);
          };

          finish_quant(acc0, 0);
          finish_quant(acc1, 1);
          finish_quant(acc2, 2);
          finish_quant(acc3, 3);
          finish_quant(acc4, 4);
          finish_quant(acc5, 5);
      }
    }

    // Cleanup Loop (Single Pixel, Unroll 1)
    // 这里我们可以恢复使用预加载模式，因为只有 1 个 Acc，寄存器很空闲
    if (p < num_pixels) {
         vint32m4_t v_mult = __riscv_vle32_v_i32m4(output_multiplier + out_c, vl);
         vuint32m4_t v_lshift = __riscv_vzext_vf4_u32m4(__riscv_vle8_v_u8m1(lshift_data.get() + out_c, vl), vl);
         vuint32m4_t v_rshift = __riscv_vzext_vf4_u32m4(__riscv_vle8_v_u8m1(rshift_data.get() + out_c, vl), vl);

         for (; p < num_pixels; ++p) {
            vint32m4_t acc = bias_vec;
            const int8_t* in_ptr = input_data + p * input_depth;
            const int8_t* w_ptr = w_ptr_base;

            for (int ic = 0; ic < input_depth; ++ic) {
                vint8m1_t w_8 = __riscv_vlse8_v_i8m1(w_ptr + ic, w_stride, vl);
                vint16m2_t w_16 = __riscv_vsext_vf2_i16m2(w_8, vl);
                acc = __riscv_vwmacc_vx_i32m4(acc, (int16_t)in_ptr[ic] + input_offset, w_16, vl);
            }
            vint8m1_t out = QuantizeResult_m4(acc, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl);
            __riscv_vse8_v_i8m1(output_data + p * output_depth + out_c, out, vl);
         }
    }

    out_c += vl;
    out_c_rem -= vl;
  }
}

// =========================================================
// 5. 通用卷积极限优化 (保持 LMUL=2, Unroll=8)
// =========================================================
void ConvGeneralPerChannelRVV_Optimized_Unroll8(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* __restrict__ input_data, const RuntimeShape& filter_shape,
    const int8_t* __restrict__ filter_data, const RuntimeShape& bias_shape,
    const int32_t* __restrict__ bias_data, const RuntimeShape& output_shape,
    int8_t* __restrict__ output_data) {

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

  int8_t zero_point_val = (int8_t)(-input_offset);
  auto zero_buffer_ptr = make_aligned_array<int8_t>(16, input_depth);
  std::memset(zero_buffer_ptr.get(), zero_point_val, input_depth);
  const int8_t* zero_ptr = zero_buffer_ptr.get();

  const ptrdiff_t w_stride_oc = filter_height * filter_width * input_depth * sizeof(int8_t);

  int out_c = 0;
  size_t out_c_rem = output_depth;

  while (out_c_rem > 0) {
    const size_t vl = __riscv_vsetvl_e32m2(out_c_rem);

    vint32m2_t v_mult = __riscv_vle32_v_i32m2(output_multiplier + out_c, vl);
    vuint8mf2_t v_lshift_8 = __riscv_vle8_v_u8mf2(lshift_data.get() + out_c, vl);
    vuint8mf2_t v_rshift_8 = __riscv_vle8_v_u8mf2(rshift_data.get() + out_c, vl);
    vuint32m2_t v_lshift = __riscv_vzext_vf4_u32m2(v_lshift_8, vl);
    vuint32m2_t v_rshift = __riscv_vzext_vf4_u32m2(v_rshift_8, vl);

    vint32m2_t bias_vec;
    if (bias_data) bias_vec = __riscv_vle32_v_i32m2(bias_data + out_c, vl);
    else bias_vec = __riscv_vmv_v_x_i32m2(0, vl);

    const int8_t* w_base_block = filter_data + out_c * (filter_height * filter_width * input_depth);

    for (int b = 0; b < batches; ++b) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        
        const int in_y_origin = out_y * stride_h - pad_h;
        int8_t* out_row_ptr = output_data + ((b * output_height + out_y) * output_width) * output_depth + out_c;

        int out_x = 0;
        for (; out_x <= output_width - 8; out_x += 8) {
            
            vint32m2_t acc0 = bias_vec, acc1 = bias_vec, acc2 = bias_vec, acc3 = bias_vec;
            vint32m2_t acc4 = bias_vec, acc5 = bias_vec, acc6 = bias_vec, acc7 = bias_vec;

            int in_x_origin[8];
            for(int i=0; i<8; ++i) in_x_origin[i] = (out_x + i) * stride_w - pad_w;

            for (int ky = 0; ky < filter_height; ++ky) {
                const int in_y = in_y_origin + ky;
                const bool in_y_valid = (in_y >= 0 && in_y < input_height);
                const int8_t* in_row_ptr = in_y_valid ? input_data + (b * input_height + in_y) * input_width * input_depth : nullptr;

                for (int kx = 0; kx < filter_width; ++kx) {
                    const int8_t* ptrs[8];
                    if (!in_y_valid) {
                        for(int i=0;i<8;++i) ptrs[i] = zero_ptr;
                    } else {
                        for(int i=0; i<8; ++i) {
                            int ix = in_x_origin[i] + kx;
                            if (ix >= 0 && ix < input_width) {
                                ptrs[i] = in_row_ptr + ix * input_depth;
                            } else {
                                ptrs[i] = zero_ptr;
                            }
                        }
                    }

                    int w_offset_base = (ky * filter_width + kx) * input_depth;
                    const int8_t* w_ptr_curr = w_base_block + w_offset_base;

                    for (int ic = 0; ic < input_depth; ++ic) {
                        vint8mf2_t w_8 = __riscv_vlse8_v_i8mf2(w_ptr_curr + ic, w_stride_oc, vl);
                        vint16m1_t w_16 = __riscv_vsext_vf2_i16m1(w_8, vl);

                        acc0 = __riscv_vwmacc_vx_i32m2(acc0, (int16_t)ptrs[0][ic] + input_offset, w_16, vl);
                        acc1 = __riscv_vwmacc_vx_i32m2(acc1, (int16_t)ptrs[1][ic] + input_offset, w_16, vl);
                        acc2 = __riscv_vwmacc_vx_i32m2(acc2, (int16_t)ptrs[2][ic] + input_offset, w_16, vl);
                        acc3 = __riscv_vwmacc_vx_i32m2(acc3, (int16_t)ptrs[3][ic] + input_offset, w_16, vl);
                        acc4 = __riscv_vwmacc_vx_i32m2(acc4, (int16_t)ptrs[4][ic] + input_offset, w_16, vl);
                        acc5 = __riscv_vwmacc_vx_i32m2(acc5, (int16_t)ptrs[5][ic] + input_offset, w_16, vl);
                        acc6 = __riscv_vwmacc_vx_i32m2(acc6, (int16_t)ptrs[6][ic] + input_offset, w_16, vl);
                        acc7 = __riscv_vwmacc_vx_i32m2(acc7, (int16_t)ptrs[7][ic] + input_offset, w_16, vl);
                    }
                }
            }

            __riscv_vse8_v_i8mf2(out_row_ptr + 0 * output_depth, QuantizeResult_m2(acc0, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl), vl);
            __riscv_vse8_v_i8mf2(out_row_ptr + 1 * output_depth, QuantizeResult_m2(acc1, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl), vl);
            __riscv_vse8_v_i8mf2(out_row_ptr + 2 * output_depth, QuantizeResult_m2(acc2, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl), vl);
            __riscv_vse8_v_i8mf2(out_row_ptr + 3 * output_depth, QuantizeResult_m2(acc3, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl), vl);
            __riscv_vse8_v_i8mf2(out_row_ptr + 4 * output_depth, QuantizeResult_m2(acc4, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl), vl);
            __riscv_vse8_v_i8mf2(out_row_ptr + 5 * output_depth, QuantizeResult_m2(acc5, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl), vl);
            __riscv_vse8_v_i8mf2(out_row_ptr + 6 * output_depth, QuantizeResult_m2(acc6, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl), vl);
            __riscv_vse8_v_i8mf2(out_row_ptr + 7 * output_depth, QuantizeResult_m2(acc7, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl), vl);

            out_row_ptr += 8 * output_depth;
        }

        // Remainder loop
        for (; out_x < output_width; ++out_x) {
             vint32m2_t acc = bias_vec;
             const int in_x_origin = out_x * stride_w - pad_w;

             for (int ky = 0; ky < filter_height; ++ky) {
                const int in_y = in_y_origin + ky;
                const bool in_y_valid = (in_y >= 0 && in_y < input_height);
                const int8_t* in_row_ptr = in_y_valid ? input_data + (b * input_height + in_y) * input_width * input_depth : nullptr;

                for (int kx = 0; kx < filter_width; ++kx) {
                    const int in_x = in_x_origin + kx;
                    const int8_t* ptr;
                    if (in_y_valid && in_x >= 0 && in_x < input_width) {
                        ptr = in_row_ptr + in_x * input_depth;
                    } else {
                        ptr = zero_ptr;
                    }

                    int w_offset_base = (ky * filter_width + kx) * input_depth;
                    const int8_t* w_ptr_curr = w_base_block + w_offset_base;

                    for (int ic = 0; ic < input_depth; ++ic) {
                        vint8mf2_t w_8 = __riscv_vlse8_v_i8mf2(w_ptr_curr + ic, w_stride_oc, vl);
                        int16_t in_val = (int16_t)ptr[ic] + input_offset;
                        acc = __riscv_vwmacc_vx_i32m2(acc, in_val, __riscv_vsext_vf2_i16m1(w_8, vl), vl);
                    }
                }
             }
             __riscv_vse8_v_i8mf2(out_row_ptr, QuantizeResult_m2(acc, v_mult, v_lshift, v_rshift, output_offset, output_min, output_max, vl), vl);
             out_row_ptr += output_depth;
        }
      }
    }

    out_c += vl;
    out_c_rem -= vl;
  }
}

}  // namespace

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

  if (filter_h == 1 && filter_w == 1 && 
      stride_h == 1 && stride_w == 1 &&
      dilation_h == 1 && dilation_w == 1) {
     Conv1x1PerChannelRVV_Optimized(params, output_multiplier, output_shift, input_shape, 
         input_data, filter_shape, filter_data, bias_shape, bias_data, 
         output_shape, output_data);
  } 
  else if (dilation_h == 1 && dilation_w == 1) {
     ConvGeneralPerChannelRVV_Optimized_Unroll8(params, output_multiplier, output_shift, input_shape, 
         input_data, filter_shape, filter_data, bias_shape, bias_data, 
         output_shape, output_data);
  }
  else {
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

  const TfLiteEvalTensor* input = GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter = GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (tflite::micro::GetTensorData<int32_t>(GetEvalInput(context, node, kConvBiasTensor)) != nullptr)
          ? GetEvalInput(context, node, kConvBiasTensor) : nullptr;
  TfLiteEvalTensor* output = GetEvalOutput(context, node, kConvOutputTensor);

  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
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
