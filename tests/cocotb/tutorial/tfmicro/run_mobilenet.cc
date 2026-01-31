// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cstring> // For std::strncpy

#include "sw/opt/litert-micro/conv.h" 
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/compatibility.h" 

#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_partial_layers.h"

namespace {
using MobilenetOpResolver = tflite::MicroMutableOpResolver<5>; // 稍微增加一点大小以防万一
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_CONV_2D;

TfLiteStatus RegisterOps(MobilenetOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_CONV_2D()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
  return kTfLiteOk;
}
}  // namespace

// === 1. 定义性能日志结构体 ===
struct OpLogEntry {
  char op_name[32];   // 算子名称
  uint32_t cycles;    // 耗时周期
};

constexpr int kMaxLogEntries = 64; 

extern "C" {
// aligned(16)
constexpr size_t kTensorArenaSize = 256 * 1024;
int8_t inference_status = -1;       // 0 = Success, -1 = Fail
uint32_t inference_cycles = 0;      // 总耗时
int32_t output_class = -1;          // 结果类别
int8_t output_score = -128;         // 结果分数

char inference_status_message[64]
    __attribute__((section(".data"), aligned(16)));

uint8_t tensor_arena[kTensorArenaSize]
    __attribute__((section(".data"), aligned(16)));    

// === 全局日志数组 ===
int32_t op_log_count = 0;
OpLogEntry op_logs[kMaxLogEntries] __attribute__((section(".data"), aligned(16)));
}

inline uint32_t read_cycles() {
  uint32_t cycles;
  asm volatile("csrr %0, mcycle" : "=r"(cycles));
  return cycles;
}

// === 2. 自定义 Profiler ===
class CycleProfiler : public tflite::MicroProfiler {
 public:
  CycleProfiler() = default;
  ~CycleProfiler() override = default;

  // [关键修复] 添加此宏以允许派生类正确处理 TFLite Micro 的内存管理策略
  TF_LITE_REMOVE_VIRTUAL_DELETE

  uint32_t BeginEvent(const char* tag) override {
    // 保存算子名字到当前的 log entry
    if (op_log_count < kMaxLogEntries) {
        const char* src = tag ? tag : "Unknown";
        char* dst = op_logs[op_log_count].op_name;
        int i = 0;
        // 复制字符串，最多31个字符，保留结尾 \0
        while (*src && i < 31) { 
            *dst++ = *src++; 
            i++; 
        }
        *dst = '\0';
    }
    
    last_start_ = read_cycles();
    // 返回当前的 index 作为 handle
    return op_log_count; 
  }

  void EndEvent(uint32_t event_handle) override {
    uint32_t end = read_cycles();
    uint32_t duration = end - last_start_;
    
    // 确保 handle 匹配当前的 count (简单的校验)
    if (event_handle < kMaxLogEntries && event_handle == (uint32_t)op_log_count) {
        op_logs[event_handle].cycles = duration;
        // 只有在结束时才增加计数，这样 Python 端看到计数增加时，数据已准备好
        op_log_count++; 
    }
  }

 private:
  uint32_t last_start_ = 0;
};

int main(int argc, char** argv) {
  // 初始化状态信息
  std::memset(inference_status_message, 0, sizeof(inference_status_message));
  
  const tflite::Model* model =
      tflite::GetModel(g_mobilenet_v1_025_partial_layers_model_data);
      
  if (model->version() != TFLITE_SCHEMA_VERSION) {
     std::strncpy(inference_status_message, "Model schema mismatch", 63);
     return -1;
  }

  // 注册算子
  MobilenetOpResolver op_resolver;
  if (RegisterOps(op_resolver) != kTfLiteOk) {
      std::strncpy(inference_status_message, "Op registration failed", 63);
      return -1;
  }

  CycleProfiler profiler;

  // === 初始化解释器 ===
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize, nullptr, &profiler);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::strncpy(inference_status_message, "AllocateTensors failed", 63);
    return -1;
  }

  std::strncpy(inference_status_message, "Running Invoke...", 63);

  // === 执行推理 ===
  uint32_t start_cycles = read_cycles();
  
  if (interpreter.Invoke() != kTfLiteOk) {
    std::strncpy(inference_status_message, "Error during Invoke", 63);
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
  inference_status = 0; // 成功
  return 0;
}
