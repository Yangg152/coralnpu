// Copyright 2025 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cstring> 

// === 1. 引入优化算子头文件 ===
#include "sw/opt/litert-micro/conv.h"           
#include "sw/opt/litert-micro/depthwise_conv.h" 
#include "sw/opt/litert-micro/mul.h" 
#include "sw/opt/litert-micro/sub.h" 

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/compatibility.h" 

// 引入模型头文件
#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_128_quant.h"

// === 全局变量定义 ===
extern "C" {
constexpr size_t kTensorArenaSize = 512 * 1024; 

int8_t inference_status = -1;       
uint32_t inference_cycles = 0;      
int32_t output_class = -1;          
int8_t output_score = -128;         

char inference_status_message[64] 
    __attribute__((section(".data"), aligned(16)));

uint8_t tensor_arena[kTensorArenaSize]
    __attribute__((section(".data"), aligned(16)));

// Profiler 数据
struct OpLogEntry {
  char op_name[32];   
  uint32_t cycles;    
};
constexpr int kMaxLogEntries = 128; 
int32_t op_log_count = 0;
OpLogEntry op_logs[kMaxLogEntries] __attribute__((section(".data"), aligned(16)));

constexpr int kDebugBufferSize = 4096;
char debug_log_buffer[kDebugBufferSize] 
    __attribute__((section(".data"), aligned(16)));
int debug_log_index = 0;
}

namespace {

// 增加算子槽位到 32
using MobilenetOpResolver = tflite::MicroMutableOpResolver<32>;
using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_Mul;
using coralnpu_v2::opt::litert_micro::Register_Sub;

TfLiteStatus RegisterOps(MobilenetOpResolver& op_resolver) {
  // === 1. 核心卷积 (RVV优化) ===
  // TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  // TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  // TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  // TF_LITE_ENSURE_STATUS(op_resolver.AddSub()); 

  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_CONV_2D()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul(Register_Mul()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddSub(Register_Sub())); 

  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  return kTfLiteOk;
}

inline uint32_t read_cycles() {
  uint32_t cycles;
  asm volatile("csrr %0, mcycle" : "=r"(cycles));
  return cycles;
}
}  // namespace

// === Profiler ===
class CycleProfiler : public tflite::MicroProfiler {
 public:
  CycleProfiler() = default;
  ~CycleProfiler() override = default;
  TF_LITE_REMOVE_VIRTUAL_DELETE
  uint32_t BeginEvent(const char* tag) override {
    if (op_log_count < kMaxLogEntries) {
        const char* src = tag ? tag : "Unknown";
        char* dst = op_logs[op_log_count].op_name;
        int i = 0;
        while (*src && i < 31) { *dst++ = *src++; i++; }
        *dst = '\0';
    }
    last_start_ = read_cycles();
    return op_log_count; 
  }
  void EndEvent(uint32_t event_handle) override {
    uint32_t end = read_cycles();
    uint32_t duration = end - last_start_;
    if (event_handle < kMaxLogEntries && event_handle == (uint32_t)op_log_count) {
        op_logs[event_handle].cycles = duration;
        op_log_count++; 
    }
  }
 private:
  uint32_t last_start_ = 0;
};

// [新增] 重新实现 DebugLog
// TFLite Micro 内部会调用这个函数来打印错误
extern "C" void __wrap_DebugLog(const char* s) {
  // 将字符串追加到全局 buffer 中
  while (*s && debug_log_index < kDebugBufferSize - 1) {
    debug_log_buffer[debug_log_index++] = *s++;
  }
  debug_log_buffer[debug_log_index] = '\0'; // 确保结尾有结束符
}

int main(int argc, char** argv) {
  std::memset(inference_status_message, 0, sizeof(inference_status_message));
  tflite::InitializeTarget(); 

  const tflite::Model* model =
      tflite::GetModel(g_mobilenet_v1_025_128_quant_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
     std::strncpy(inference_status_message, "Model schema mismatch", 63);
     return -1;
  }

  MobilenetOpResolver op_resolver;
  if (RegisterOps(op_resolver) != kTfLiteOk) {
      std::strncpy(inference_status_message, "Op registration failed", 63);
      return -1;
  }

  CycleProfiler profiler;

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize, nullptr, &profiler);

  // 尝试分配 Tensor
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    // 如果这里失败，通常是因为缺算子(Missing Op)或 Arena 太小
    // 我们已经补全了算子，且 800KB 对该模型足够
    std::strncpy(inference_status_message, "AllocateTensors failed", 63);
    return -1;
  }

  std::strncpy(inference_status_message, "Running Invoke...", 63);

  uint32_t start_cycles = read_cycles();
  
  if (interpreter.Invoke() != kTfLiteOk) {
    std::strncpy(inference_status_message, "Invoke failed", 63);
    inference_status = -1;
    return -1;
  }
  
  uint32_t end_cycles = read_cycles();
  inference_cycles = end_cycles - start_cycles;

  // === 解析输出 ===
  TfLiteTensor* output = interpreter.output(0);
  if (output != nullptr) {
      intmax_t num_elements = 1;
      for (int i = 0; i < output->dims->size; ++i) {
          num_elements *= output->dims->data[i];
      }

      int best_idx = 0;
      int8_t best_val = -128;
      for (int i = 0; i < num_elements; ++i) {
          if (output->data.int8[i] > best_val) {
              best_val = output->data.int8[i];
              best_idx = i;
          }
      }
      output_class = best_idx;
      output_score = best_val;
  }

  std::strncpy(inference_status_message, "Invoke successful", 63);
  inference_status = 0;
  return 0;
}
