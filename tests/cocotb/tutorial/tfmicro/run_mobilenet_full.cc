// Copyright 2025 Google LLC
// Licensed under the Apache License, Version 2.0

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cstring> 

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
#include "tensorflow/lite/schema/schema_generated.h" 

#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_128_quant.h"

using MobilenetOpResolver = tflite::MicroMutableOpResolver<10>;

enum DebugStage {
  STAGE_INIT = 0,
  STAGE_MAGIC_CHECK = 1,     
  STAGE_STRUCTURE_CHECK = 2, 
  STAGE_OPS_REGISTERED = 3,
  STAGE_ALLOCATE_START = 4,
  STAGE_ALLOCATE_DONE = 5,
  STAGE_INVOKE_START = 6,
  STAGE_INVOKE_DONE = 8,
  STAGE_ERROR = 99
};

using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_MUL;
using coralnpu_v2::opt::litert_micro::Register_SUB;
using coralnpu_v2::opt::litert_micro::Register_MEAN;

TfLiteStatus RegisterOps(MobilenetOpResolver& op_resolver) {
  // TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  // TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  // TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  // TF_LITE_ENSURE_STATUS(op_resolver.AddSub()); 
  // TF_LITE_ENSURE_STATUS(op_resolver.AddMean());

  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_CONV_2D()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul(Register_MUL()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddSub(Register_SUB())); 
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean(Register_MEAN()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());             
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());  

  return kTfLiteOk;
}
// [已移除] append_int_to_str 函数，因为不再需要打印 Pre-check 日志

// 定义性能日志结构体
struct OpLogEntry {
  char op_name[32];   // 32 bytes
  uint32_t cycles;    // 4 bytes
};                    // Total 36 bytes

constexpr int kMaxLogEntries = 128;

extern "C" {
constexpr size_t kTensorArenaSize = 300 * 1024; 
uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".data"), aligned(16)));    

char debug_log_buffer[512] __attribute__((section(".data"), aligned(16)));

void __wrap_DebugLog(const char* s) {
    if (!s) return;
    static int global_log_idx = 0; 
    if (global_log_idx >= 510) return;
    for (int i = 0; s[i] != 0; i++) {
        if (global_log_idx < 511) debug_log_buffer[global_log_idx++] = s[i];
        else break;
    }
    debug_log_buffer[global_log_idx] = 0;
}

volatile int8_t inference_status = -1;       
volatile uint32_t inference_cycles = 0;      
volatile int32_t output_class = -1;          
volatile int8_t output_score = -128;         

volatile int32_t debug_stage __attribute__((section(".data"))) = STAGE_INIT;
volatile int32_t current_op_index __attribute__((section(".data"))) = -1;

volatile uint32_t debug_sp_addr __attribute__((section(".data"))) = 0;
volatile uint32_t debug_arena_start __attribute__((section(".data"))) = 0;
volatile uint32_t debug_model_addr __attribute__((section(".data"))) = 0;

char inference_status_message[64] __attribute__((section(".data"), aligned(16)));
char current_running_op[32] __attribute__((section(".data"), aligned(16)));

// 日志数组
volatile int32_t op_log_count = 0;
OpLogEntry op_logs[kMaxLogEntries] __attribute__((section(".data"), aligned(16)));
}

inline uint32_t read_cycles() {
  uint32_t cycles;
  asm volatile("csrr %0, mcycle" : "=r"(cycles));
  return cycles;
}

// =========================================================================
// Profiler: 记录每个算子的耗时
// =========================================================================
class CycleProfiler : public tflite::MicroProfiler {
 public:
  CycleProfiler() = default;
  ~CycleProfiler() override = default;
  TF_LITE_REMOVE_VIRTUAL_DELETE
  
  uint32_t BeginEvent(const char* tag) override {
    // 仅在 INVOKE 阶段记录
    if (debug_stage == STAGE_INVOKE_START) {
        current_op_index++;
        
        const char* name = tag ? tag : "Unknown";
        int i = 0;
        for (; i < 31 && name[i] != 0; i++) {
            current_running_op[i] = name[i];
        }
        current_running_op[i] = 0;

        // 准备日志条目
        if (op_log_count < kMaxLogEntries) {
             int j = 0;
             for (; j < 31 && name[j] != 0; j++) {
                 op_logs[op_log_count].op_name[j] = name[j];
             }
             op_logs[op_log_count].op_name[j] = 0;
        }
    }
    return read_cycles(); 
  }

  void EndEvent(uint32_t event_handle) override {
      if (debug_stage == STAGE_INVOKE_START) {
          uint32_t end_cycles = read_cycles();
          if (op_log_count < kMaxLogEntries) {
              op_logs[op_log_count].cycles = end_cycles - event_handle;
              op_log_count++;
          }
      }
  }
};

int main(int argc, char** argv) {
  debug_stage = STAGE_INIT;
  current_op_index = -1;
  std::memset(inference_status_message, 0, sizeof(inference_status_message));
  std::memset(current_running_op, 0, sizeof(current_running_op));
  std::memset(debug_log_buffer, 0, sizeof(debug_log_buffer));
  
  debug_sp_addr = (uint32_t)&tensor_arena[0]; 
  debug_arena_start = (uint32_t)&tensor_arena[0];
  debug_model_addr = (uint32_t)g_mobilenet_v1_025_128_quant_model_data;

  // -----------------------------------------------------------
  // Step 1: Magic Check
  // -----------------------------------------------------------
  debug_stage = STAGE_MAGIC_CHECK;
  const uint8_t* raw_data = g_mobilenet_v1_025_128_quant_model_data;
  
  if (raw_data[4] != 'T' || raw_data[5] != 'F' || raw_data[6] != 'L' || raw_data[7] != '3') {
      std::strncpy(inference_status_message, "FATAL: Bad Magic 'TFL3'!", 63);
      debug_stage = STAGE_ERROR;
      return -1;
  }
  
  const tflite::Model* model = tflite::GetModel(g_mobilenet_v1_025_128_quant_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
     std::strncpy(inference_status_message, "FATAL: Schema Version Mismatch", 63);
     debug_stage = STAGE_ERROR;
     return -1;
  }

  // -----------------------------------------------------------
  // Step 2: Structure Check
  // -----------------------------------------------------------
  debug_stage = STAGE_STRUCTURE_CHECK;
  if (model->subgraphs() == nullptr || model->subgraphs()->size() == 0) {
      std::strncpy(inference_status_message, "FATAL: No subgraphs found", 63);
      debug_stage = STAGE_ERROR;
      return -1;
  }

  // -----------------------------------------------------------
  // Step 3: Register Ops (Directly)
  // -----------------------------------------------------------
  MobilenetOpResolver op_resolver;
  RegisterOps(op_resolver);
  
  std::strncpy(inference_status_message, "Ops Registered", 63);
  debug_stage = STAGE_OPS_REGISTERED;

  // -----------------------------------------------------------
  // Step 4: Allocate
  // -----------------------------------------------------------
  CycleProfiler profiler;
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize, nullptr, &profiler);

  std::strncpy(inference_status_message, "Allocating...", 63);
  debug_stage = STAGE_ALLOCATE_START;
  
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::strncpy(inference_status_message, "AllocateTensors failed", 63);
    debug_stage = STAGE_ERROR;
    return -1;
  }
  debug_stage = STAGE_ALLOCATE_DONE;

  // -----------------------------------------------------------
  // Step 5: Invoke
  // -----------------------------------------------------------
  debug_stage = STAGE_INVOKE_START;
  
  current_op_index = -1; 
  op_log_count = 0; 
  std::strncpy(inference_status_message, "Invoking...", 63);
  
  uint32_t start_cycles = read_cycles();
  if (interpreter.Invoke() != kTfLiteOk) {
    std::strncpy(inference_status_message, "Invoke failed", 63);
    debug_stage = STAGE_ERROR;
    return -1;
  }
  uint32_t end_cycles = read_cycles();

  debug_stage = STAGE_INVOKE_DONE;
  inference_cycles = end_cycles - start_cycles;
  inference_status = 0; 
  std::strncpy(inference_status_message, "Test Finished SUCCESS", 63);
  
  return 0;
}
