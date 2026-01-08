#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h> 

// === 1. 引入优化算子头文件 ===
#include "sw/opt/litert-micro/conv.h"           
#include "sw/opt/litert-micro/depthwise_conv.h" 

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"

// 引入模型头文件
#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_128_quant.h"

namespace {

// 定义 OpResolver
using MobilenetOpResolver = tflite::MicroMutableOpResolver<10>; // 稍微加大一点以防万一

// 引用命名空间中的注册函数
using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;

TfLiteStatus RegisterOps(MobilenetOpResolver& op_resolver) {
  // 1. 核心卷积 (RVV优化)
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_CONV_2D()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
  
  // 2. 结构算子
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  
  // 3. 必要的辅助算子 (防止模型初始化失败)
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad()); 
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());

  return kTfLiteOk;
}

inline uint32_t read_cycles() {
  uint32_t cycles;
  asm volatile("csrr %0, mcycle" : "=r"(cycles));
  return cycles;
}

}  // namespace

// === 定义 Profiling 数据结构 ===
struct LayerProfile {
    uint32_t op_code; // TFLite 内部的 Operator Code
    uint32_t cycles;  // 该层耗时
};

constexpr int kMaxLayers = 64; // 假设模型不超过 64 层

extern "C" {
constexpr size_t kTensorArenaSize = 512 * 1024; 

// === 全局变量供 Python 读取 ===
int8_t inference_status = -1;       
uint32_t inference_cycles = 0;      
int32_t output_class = -1;          
int8_t output_score = -128;         

// 状态消息
char inference_status_message[64] 
    __attribute__((section(".data"), aligned(16)));

// Tensor Arena
uint8_t tensor_arena[kTensorArenaSize]
    __attribute__((section(".data"), aligned(16)));

// Profiling 结果数组 (放到 .data 段)
LayerProfile profile_data[kMaxLayers] 
    __attribute__((section(".data"), aligned(16)));
int profile_count = 0;
}

// === 自定义 Profiler ===
class MemoryProfiler : public tflite::MicroProfiler {
public:
    // 重写 AddEvent，记录每个算子的耗时到内存数组
    void AddEvent(const char* tag, uint32_t start, uint32_t end, uint32_t event_tag) override {
        if (profile_count < kMaxLayers) {
            profile_data[profile_count].op_code = event_tag;
            profile_data[profile_count].cycles = end - start;
            profile_count++;
        }
        // 调用基类以保留默认的日志打印功能 (如果在仿真器中能看到的话)
        tflite::MicroProfiler::AddEvent(tag, start, end, event_tag);
    }
};

int main(int argc, char** argv) {
  // 必须初始化 Target，这通常会设置 timer，让 MicroProfiler 能读取时间
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

  // === 关键修改：实例化 MemoryProfiler ===
  MemoryProfiler profiler;

  // === 关键修改：将 profiler 指针传给 Interpreter ===
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize, nullptr, &profiler);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::strncpy(inference_status_message, "AllocateTensors failed", 63);
    return -1;
  }

  std::strncpy(inference_status_message, "Running Invoke...", 63);

  uint32_t start_cycles = read_cycles();
  
  // Invoke 时会自动调用 profiler.AddEvent
  if (interpreter.Invoke() != kTfLiteOk) {
    std::strncpy(inference_status_message, "Error during Invoke", 63);
    return -1;
  }
  
  uint32_t end_cycles = read_cycles();
  inference_cycles = end_cycles - start_cycles;

  // 解析输出
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
