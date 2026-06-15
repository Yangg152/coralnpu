#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/litert-micro/pooling.h"
#include "sw/opt/litert-micro/fully_connected.h"
#include "sw/opt/litert-micro/softmax.h"
#include "sw/opt/litert-micro/mxu.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/schema/schema_generated.h"
// 核心头文件：包含 MicroPrintf
#include "tensorflow/lite/micro/micro_log.h" 

// VWW 专用头文件
#include "tests/tflm/vww/vww_data/vww_model_data.h"
#include "tests/tflm/vww/vww_data/vww_input_data.h"
#include "tests/tflm/vww/vww_data/vww_model_settings.h"

using VwwOpResolver = tflite::MicroMutableOpResolver<10>;

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
using coralnpu_v2::opt::litert_micro::Register_FULLY_CONNECTED;
using coralnpu_v2::opt::litert_micro::Register_AVERAGE_POOL_2D;
using coralnpu_v2::opt::litert_micro::Register_SOFTMAX;
using coralnpu_v2::opt::litert_micro::Register_MXU_CONV_2D;

TfLiteStatus RegisterOps(VwwOpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_MXU_CONV_2D()));
    TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected(Register_FULLY_CONNECTED()));
    TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D(Register_AVERAGE_POOL_2D()));
    TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax(Register_SOFTMAX()));
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    return kTfLiteOk;
}

// 性能日志结构体
struct OpLogEntry {
  char op_name[32];   
  uint32_t cycles;    
};                    

constexpr int kMaxLogEntries = 128;

extern "C" {
// 内存与日志 Buffer 定义
constexpr size_t kTensorArenaSize = 512 * 1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".extdata"), aligned(16)));

// 增大 Buffer 以容纳完整的 TFLM 内存报告
char debug_log_buffer[2048] __attribute__((section(".extdata"), aligned(16)));
static int global_log_idx = 0;

// 实现 __wrap_DebugLog，它会捕获所有 MicroPrintf 的输出
void __wrap_DebugLog(const char* s) {
    if (!s) return;
    for (int i = 0; s[i] != 0; i++) {
        if (global_log_idx < (int)sizeof(debug_log_buffer) - 2) {
            debug_log_buffer[global_log_idx++] = s[i];
        } else {
            debug_log_buffer[sizeof(debug_log_buffer) - 2] = '!'; // 缓冲区溢出标记
            break;
        }
    }
    // 添加换行符，确保在内存查看时易读
    if (global_log_idx < (int)sizeof(debug_log_buffer) - 1) {
        debug_log_buffer[global_log_idx++] = '\n';
    }
    debug_log_buffer[global_log_idx] = 0;
}

volatile int8_t  inference_status  = -1;
volatile uint32_t inference_cycles = 0;
volatile int32_t output_class      = -1;
volatile int8_t  output_score      = -128;

volatile int32_t debug_stage       __attribute__((section(".data"))) = STAGE_INIT;
volatile int32_t current_op_index  __attribute__((section(".data"))) = -1;

volatile uint32_t debug_sp_addr    __attribute__((section(".data"))) = 0;
volatile uint32_t debug_arena_start __attribute__((section(".data"))) = 0;
volatile uint32_t debug_model_addr  __attribute__((section(".data"))) = 0;

char inference_status_message[64]  __attribute__((section(".data"), aligned(16)));
char current_running_op[32]        __attribute__((section(".data"), aligned(16)));

volatile int32_t op_log_count = 0;
OpLogEntry op_logs[kMaxLogEntries] __attribute__((section(".data"), aligned(16)));
}

inline uint32_t read_cycles() {
  uint32_t cycles;
  asm volatile("csrr %0, mcycle" : "=r"(cycles));
  return cycles;
}

// =========================================================================
// Profiler
// =========================================================================
class CycleProfiler : public tflite::MicroProfiler {
 public:
  CycleProfiler() = default;
  ~CycleProfiler() override = default;
  TF_LITE_REMOVE_VIRTUAL_DELETE

  uint32_t BeginEvent(const char* tag) override {
    if (debug_stage == STAGE_INVOKE_START) {
      current_op_index++;
      const char* name = tag ? tag : "Unknown";
      
      int i = 0;
      for (; i < 31 && name[i] != 0; i++) current_running_op[i] = name[i];
      current_running_op[i] = 0;

      if (op_log_count < kMaxLogEntries) {
        int j = 0;
        for (; j < 31 && name[j] != 0; j++)
          op_logs[op_log_count].op_name[j] = name[j];
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

// =========================================================================
// Main
// =========================================================================
int main(int argc, char** argv) {
  debug_stage = STAGE_INIT;
  current_op_index = -1;
  global_log_idx = 0; // 重置日志
  std::memset(inference_status_message, 0, sizeof(inference_status_message));
  std::memset(current_running_op, 0, sizeof(current_running_op));
  std::memset(debug_log_buffer, 0, sizeof(debug_log_buffer));

  debug_sp_addr     = (uint32_t)&tensor_arena[0];
  debug_arena_start = (uint32_t)&tensor_arena[0];
  debug_model_addr  = (uint32_t)vww_model_data;

  // 使用全局 MicroPrintf (不带 tflite:: 前缀)
  MicroPrintf("--- Starting VWW Inference ---");

  // -----------------------------------------------------------
  // Step 1: Magic Check
  // -----------------------------------------------------------
  debug_stage = STAGE_MAGIC_CHECK;
  const uint8_t* raw_data = vww_model_data;

  if (raw_data[4] != 'T' || raw_data[5] != 'F' || raw_data[6] != 'L' || raw_data[7] != '3') {
    MicroPrintf("ERR: Bad Magic! Got %c%c%c%c", raw_data[4], raw_data[5], raw_data[6], raw_data[7]);
    std::strncpy(inference_status_message, "FATAL: Bad Magic", 63);
    debug_stage = STAGE_ERROR;
    return -1;
  }

  const tflite::Model* model = tflite::GetModel(vww_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("ERR: Version Mismatch. Model: %d, Env: %d", model->version(), TFLITE_SCHEMA_VERSION);
    std::strncpy(inference_status_message, "FATAL: Schema Version Mismatch", 63);
    debug_stage = STAGE_ERROR;
    return -1;
  }

  // -----------------------------------------------------------
  // Step 2 & 3: Ops
  // -----------------------------------------------------------
  debug_stage = STAGE_STRUCTURE_CHECK;
  if (model->subgraphs() == nullptr || model->subgraphs()->size() == 0) {
    MicroPrintf("ERR: No subgraphs");
    debug_stage = STAGE_ERROR;
    return -1;
  }

  VwwOpResolver op_resolver;
  RegisterOps(op_resolver);
  debug_stage = STAGE_OPS_REGISTERED;

  // -----------------------------------------------------------
  // Step 4: Allocate
  // -----------------------------------------------------------
  CycleProfiler profiler;
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize, nullptr, &profiler);

  debug_stage = STAGE_ALLOCATE_START;
  MicroPrintf("Log: Allocating Tensors...");

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    // 这里非常重要：失败时 TFLM 内部会打印详细的内存布局报错到 debug_log_buffer
    MicroPrintf("ERR: AllocateTensors failed!");
    std::strncpy(inference_status_message, "AllocateTensors failed", 63);
    debug_stage = STAGE_ERROR;
    return -1;
  }
  debug_stage = STAGE_ALLOCATE_DONE;

  // -----------------------------------------------------------
  // Step 5: Fill Input
  // -----------------------------------------------------------
  TfLiteTensor* input = interpreter.input(0);
  if (!input) {
    MicroPrintf("ERR: Input tensor is null");
    debug_stage = STAGE_ERROR;
    return -1;
  }
  std::memcpy(input->data.int8, vww_input_data, vww_input_data_len[0]);

  // -----------------------------------------------------------
  // Step 6: Invoke
  // -----------------------------------------------------------
  debug_stage = STAGE_INVOKE_START;
  op_log_count = 0;
  MicroPrintf("Log: Invoking...");

  uint32_t start_cycles = read_cycles();
  TfLiteStatus invoke_status = interpreter.Invoke();
  uint32_t end_cycles = read_cycles();

  if (invoke_status != kTfLiteOk) {
    MicroPrintf("ERR: Invoke failed with status %d", invoke_status);
    std::strncpy(inference_status_message, "Invoke failed", 63);
    debug_stage = STAGE_ERROR;
    return -1;
  }
  
  inference_cycles = end_cycles - start_cycles;
  debug_stage = STAGE_INVOKE_DONE;

  // -----------------------------------------------------------
  // Step 7: Results
  // -----------------------------------------------------------
  TfLiteTensor* output = interpreter.output(0);
  int8_t score_no_person = output->data.int8[0];
  int8_t score_person    = output->data.int8[1];

  if (score_person > score_no_person) {
    output_class = 1;
    output_score = score_person;
  } else {
    output_class = 0;
    output_score = score_no_person;
  }

  MicroPrintf("SUCCESS: Class %d, Score %d", output_class, output_score);
  inference_status = 0;
  std::strncpy(inference_status_message, "Test Finished SUCCESS", 63);

  return 0;
}
