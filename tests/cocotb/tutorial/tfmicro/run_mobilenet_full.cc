// Copyright 2025 Google LLC
// Licensed under the Apache License, Version 2.0
// ... (License header omitted for brevity)

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cstring> 

// === 1. 引入优化算子头文件 ===
#include "sw/opt/litert-micro/conv.h"           
#include "sw/opt/litert-micro/depthwise_conv.h" 
#include "sw/opt/litert-micro/mul.h" 
#include "sw/opt/litert-micro/sub.h" 
#include "sw/opt/litert-micro/mean.h" 

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/compatibility.h" 

// 引入模型头文件
#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_128_quant.h"

// 增加 OpResolver 的容量，防止算子超出
using MobilenetOpResolver = tflite::MicroMutableOpResolver<12>;

using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_MUL;
using coralnpu_v2::opt::litert_micro::Register_SUB;
using coralnpu_v2::opt::litert_micro::Register_MEAN;

// === 定义调试阶段常量 ===
enum DebugStage {
  STAGE_INIT = 0,
  STAGE_MODEL_LOADED = 1,
  STAGE_OPS_REGISTERED = 2,
  STAGE_ALLOCATE_START = 3,
  STAGE_ALLOCATE_DONE = 4,
  STAGE_INVOKE_START = 5,
  STAGE_INVOKE_RUNNING = 6,
  STAGE_INVOKE_DONE = 7,
  STAGE_ERROR = 99
};

TfLiteStatus RegisterOps(MobilenetOpResolver& op_resolver) {
  // 建议：如果要测试特定的 MUL 问题，先确保使用标准算子能跑通，再替换为优化算子
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSub()); 
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax()); // MobileNet 通常结尾有 Softmax，以防万一

  // === 替换为你的优化算子时，取消下面注释 ===
  // TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_CONV_2D()));
  // TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
  // TF_LITE_ENSURE_STATUS(op_resolver.AddMul(Register_MUL()));
  // ...

  return kTfLiteOk;
}

struct OpLogEntry {
  char op_name[32];
  uint32_t cycles;
};

constexpr int kMaxLogEntries = 128; // 增加日志条目数

extern "C" {
// 增大 Arena 大小，防止内存分配失败导致莫名其妙的挂起
constexpr size_t kTensorArenaSize = 512 * 1024;

// === 全局状态变量 (volatile 防止被优化) ===
volatile int8_t inference_status = -1;       
volatile uint32_t inference_cycles = 0;      
volatile int32_t output_class = -1;          
volatile int8_t output_score = -128;         

// 调试状态：程序运行到了哪个阶段
volatile int32_t debug_stage __attribute__((section(".data"), aligned(16))) = STAGE_INIT;

// 算子索引：当前运行的是第几个算子
volatile int32_t current_op_index __attribute__((section(".data"), aligned(16))) = -1;

char inference_status_message[64]
    __attribute__((section(".data"), aligned(16)));

uint8_t tensor_arena[kTensorArenaSize]
    __attribute__((section(".data"), aligned(16)));    

volatile int32_t op_log_count = 0;
OpLogEntry op_logs[kMaxLogEntries] __attribute__((section(".data"), aligned(16)));

char current_running_op[32] __attribute__((section(".data"), aligned(16)));
}

inline uint32_t read_cycles() {
  uint32_t cycles;
  asm volatile("csrr %0, mcycle" : "=r"(cycles));
  return cycles;
}

// === 自定义 Profiler ===
class CycleProfiler : public tflite::MicroProfiler {
 public:
  CycleProfiler() = default;
  ~CycleProfiler() override = default;

  TF_LITE_REMOVE_VIRTUAL_DELETE

  uint32_t BeginEvent(const char* tag) override {
    const char* op_name = tag ? tag : "Unknown";

    // 更新调试信息
    current_op_index++; // 索引 +1
    debug_stage = STAGE_INVOKE_RUNNING; // 标记正在运行

    // 记录当前名字
    std::strncpy(current_running_op, op_name, 31);
    current_running_op[31] = '\0';
    
    // 同时也写到 Log 数组中（虽然还没结束，先占位名字）
    if (op_log_count < kMaxLogEntries) {
        std::strncpy(op_logs[op_log_count].op_name, op_name, 31);
        op_logs[op_log_count].op_name[31] = '\0';
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

    // 算子结束
    current_running_op[0] = '\0';
    // 注意：debug_stage 保持为 INVOKE_RUNNING，直到整个 Invoke 结束
  }

 private:
  uint32_t last_start_ = 0;
};

int main(int argc, char** argv) {
  // 1. 初始化
  debug_stage = STAGE_INIT;
  std::memset(inference_status_message, 0, sizeof(inference_status_message));
  std::memset(current_running_op, 0, sizeof(current_running_op));
  current_op_index = -1;
  
  const tflite::Model* model =
      tflite::GetModel(g_mobilenet_v1_025_128_quant_model_data);
      
  if (model->version() != TFLITE_SCHEMA_VERSION) {
     std::strncpy(inference_status_message, "Model schema mismatch", 63);
     debug_stage = STAGE_ERROR;
     return -1;
  }
  debug_stage = STAGE_MODEL_LOADED;

  // 2. 注册算子
  MobilenetOpResolver op_resolver;
  if (RegisterOps(op_resolver) != kTfLiteOk) {
      std::strncpy(inference_status_message, "Op registration failed", 63);
      debug_stage = STAGE_ERROR;
      return -1;
  }
  debug_stage = STAGE_OPS_REGISTERED;

  CycleProfiler profiler;

  // 3. 解释器构建
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize, nullptr, &profiler);

  // 4. 分配张量
  debug_stage = STAGE_ALLOCATE_START;
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::strncpy(inference_status_message, "AllocateTensors failed", 63);
    debug_stage = STAGE_ERROR;
    return -1;
  }
  debug_stage = STAGE_ALLOCATE_DONE;

  std::strncpy(inference_status_message, "Running Invoke...", 63);

  // 5. 执行推理
  uint32_t start_cycles = read_cycles();
  
  debug_stage = STAGE_INVOKE_START;
  current_op_index = -1; // 重置算子计数

  if (interpreter.Invoke() != kTfLiteOk) {
    std::strncpy(inference_status_message, "Error during Invoke", 63);
    inference_status = -1;
    debug_stage = STAGE_ERROR;
    return -1;
  }
  
  debug_stage = STAGE_INVOKE_DONE;
  uint32_t end_cycles = read_cycles();
  inference_cycles = end_cycles - start_cycles;

  // 6. 获取结果
  TfLiteTensor* output = interpreter.output(0);
  if (output != nullptr) {
      // 假设是分类输出
      int total_elements = 1;
      for (int i=0; i<output->dims->size; i++) total_elements *= output->dims->data[i];

      int best_idx = 0;
      int8_t best_val = -128;
      for (int i = 0; i < total_elements; ++i) {
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
