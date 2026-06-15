// Copyright 2025 Google LLC
// Licensed under the Apache License, Version 2.0

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/litert-micro/mxu.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tests/posenet/posenet_int8.h"
#include "tests/posenet/posenet_input.h"

using MobilenetOpResolver = tflite::MicroMutableOpResolver<10>;

enum DebugStage {
  STAGE_INIT = 0,
  STAGE_MAGIC_CHECK = 1,
  STAGE_STRUCTURE_CHECK = 2,
  STAGE_OPS_REGISTERED = 3,
  STAGE_ALLOCATE_START = 4,
  STAGE_ALLOCATE_DONE = 5,
  STAGE_INPUT_LOADED = 55,
  STAGE_INVOKE_START = 6,
  STAGE_INVOKE_DONE = 8,
  STAGE_OUTPUT_PARSED = 9,
  STAGE_ERROR = 99
};

using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_MXU_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;

TfLiteStatus RegisterOps(MobilenetOpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_MXU_CONV_2D()));
    TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
    TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
    return kTfLiteOk;
}

// =========================================================================
// 常量
// =========================================================================
constexpr int kNumKeypoints = 17;
constexpr int kNumEdges = 16;
constexpr int kMaxDetections = 10;
constexpr int kPoseStride = 16;
constexpr int kMaxLogEntries = 128;
constexpr int kMidShortRefinementSteps = 5;

// Heatmap 尺寸 (353/16+1=23, 481/16+1=31)
constexpr int kHeatmapMaxH = 23;
constexpr int kHeatmapMaxW = 31;

// =========================================================================
// 性能日志结构体
// =========================================================================
struct OpLogEntry {
  char op_name[32];
  uint32_t cycles;
};

// =========================================================================
// Pose 数据结构
// =========================================================================
struct PosePoint {
  float y;
  float x;
};

struct PoseKeypointResult {
  PosePoint position;  // 像素坐标
  float score;         // sigmoid 后 [0,1]
};

struct PoseResult {
  float pose_score;
  PoseKeypointResult keypoints[kNumKeypoints];
};

// 旧的 keypoint 结构 (兼容)
struct KeypointResult {
  int16_t y;
  int16_t x;
  int8_t  score;
  uint8_t id;
};

// =========================================================================
// 边列表 (前向16 + 后向16 = 32条)
// =========================================================================
struct Edge {
  int parent;
  int child;
};

static const Edge kEdgeList[32] = {
    // Forward edges (0-15)
    {0, 1}, {1, 3}, {0, 2}, {2, 4},
    {0, 5}, {5, 7}, {7, 9}, {5, 11},
    {11, 13}, {13, 15}, {0, 6}, {6, 8},
    {8, 10}, {6, 12}, {12, 14}, {14, 16},
    // Backward edges (16-31)
    {1, 0}, {3, 1}, {2, 0}, {4, 2},
    {5, 0}, {7, 5}, {9, 7}, {11, 5},
    {13, 11}, {15, 13}, {6, 0}, {8, 6},
    {10, 8}, {12, 6}, {14, 12}, {16, 14}
};

// 静态邻接表
struct StaticAdjacencyList {
  int child_ids[kNumKeypoints][8];
  int edge_ids[kNumKeypoints][8];
  int num_children[kNumKeypoints];
};

// =========================================================================
// 优先队列 (静态数组, 最大堆)
// =========================================================================
struct KeypointCandidate {
  PosePoint point;
  int id;
  float score;
};

struct StaticPriorityQueue {
  static constexpr int kMaxSize = 4096;
  KeypointCandidate items[kMaxSize];
  int size;

  void init() { size = 0; }

  void push(const KeypointCandidate& item) {
    if (size >= kMaxSize) return;
    items[size] = item;
    size++;
    int i = size - 1;
    while (i > 0) {
      int parent = (i - 1) / 2;
      if (items[parent].score < items[i].score) {
        KeypointCandidate tmp = items[parent];
        items[parent] = items[i];
        items[i] = tmp;
        i = parent;
      } else {
        break;
      }
    }
  }

  KeypointCandidate pop() {
    KeypointCandidate top = items[0];
    size--;
    if (size > 0) {
      items[0] = items[size];
      int i = 0;
      while (true) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int largest = i;
        if (left < size && items[left].score > items[largest].score) largest = left;
        if (right < size && items[right].score > items[largest].score) largest = right;
        if (largest != i) {
          KeypointCandidate tmp = items[i];
          items[i] = items[largest];
          items[largest] = tmp;
          i = largest;
        } else {
          break;
        }
      }
    }
    return top;
  }

  bool empty() const { return size == 0; }
};

// =========================================================================
// extern "C" 全局变量
// =========================================================================
extern "C" {
constexpr size_t kTensorArenaSize = 8 * 1024 * 1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".extdata"), aligned(16)));

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

// 性能日志
volatile int32_t op_log_count = 0;
OpLogEntry op_logs[kMaxLogEntries] __attribute__((section(".data"), aligned(16)));

// 旧版关键点输出 (兼容)
volatile int32_t num_keypoints_found = 0;
KeypointResult keypoint_results[kNumKeypoints] __attribute__((section(".data"), aligned(16)));

// 输出 tensor 的元信息
volatile int32_t output_tensor_count = 0;
volatile int32_t heatmap_height = 0;
volatile int32_t heatmap_width = 0;

// 多人 Pose 检测结果
volatile int32_t num_poses_found = 0;
PoseResult pose_results[kMaxDetections] __attribute__((section(".data"), aligned(16)));

// 反量化用的 float buffer (静态分配在 extdata section)
float heatmap_float[kHeatmapMaxH * kHeatmapMaxW * kNumKeypoints]
    __attribute__((section(".extdata"), aligned(16)));
float short_offset_float[kHeatmapMaxH * kHeatmapMaxW * 2 * kNumKeypoints]
    __attribute__((section(".extdata"), aligned(16)));
float mid_offset_float[kHeatmapMaxH * kHeatmapMaxW * 2 * 2 * kNumEdges]
    __attribute__((section(".extdata"), aligned(16)));

volatile int32_t diag_num_outputs __attribute__((section(".data"))) = 0;
volatile int32_t diag_out_dims[4][4] __attribute__((section(".data")));  // 最多4个output，每个4维
volatile float diag_out_scale[4] __attribute__((section(".data")));
volatile int32_t diag_out_zp[4] __attribute__((section(".data")));
volatile int32_t diag_heatmap_min __attribute__((section(".data"))) = 0;
volatile int32_t diag_heatmap_max __attribute__((section(".data"))) = 0;

// 原始输出 tensor 数据 (后处理前)
int8_t raw_heatmap[kHeatmapMaxH * kHeatmapMaxW * kNumKeypoints]
    __attribute__((section(".extdata"), aligned(16)));
int8_t raw_short_offsets[kHeatmapMaxH * kHeatmapMaxW * 2 * kNumKeypoints]
    __attribute__((section(".extdata"), aligned(16)));
int8_t raw_mid_offsets[kHeatmapMaxH * kHeatmapMaxW * 2 * 2 * kNumEdges]
    __attribute__((section(".extdata"), aligned(16)));
volatile int32_t raw_heatmap_size __attribute__((section(".data"))) = 0;
volatile int32_t raw_short_offsets_size __attribute__((section(".data"))) = 0;
volatile int32_t raw_mid_offsets_size __attribute__((section(".data"))) = 0;
}  // extern "C"

// =========================================================================
// 辅助函数
// =========================================================================

inline uint32_t read_cycles() {
  uint32_t cycles;
  asm volatile("csrr %0, mcycle" : "=r"(cycles));
  return cycles;
}

inline float sigmoid_f(float x) {
  return 1.0f / (1.0f + expf(-x));
}

template <typename T>
inline T clamp_val(T v, T lo, T hi) {
  return v < lo ? lo : (hi < v ? hi : v);
}

// 反量化 int8 tensor 到 float buffer
inline void dequantize_tensor_int8(const TfLiteTensor* tensor, float* output,
                                   float extra_scale = 1.0f) {
  const int8_t* data = tensor->data.int8;
  const float scale = tensor->params.scale * extra_scale;
  const int32_t zp = tensor->params.zero_point;
  const int num_elements = static_cast<int>(tensor->bytes);

  for (int i = 0; i < num_elements; i++) {
    output[i] = (static_cast<float>(data[i]) - static_cast<float>(zp)) * scale;
  }
}

// 双线性插值采样 (多 channel)
inline void sample_tensor_multi(const float* tensor, int height, int width,
                                int num_channels, float y, float x,
                                const int* channels, int n_channels,
                                float* results) {
  const float yc = clamp_val(y, 0.0f, static_cast<float>(height - 1));
  const float xc = clamp_val(x, 0.0f, static_cast<float>(width - 1));

  const int y_floor = static_cast<int>(yc);
  const int y_ceil = (y_floor < height - 1) ? y_floor + 1 : y_floor;
  const float y_lerp = yc - static_cast<float>(y_floor);

  const int x_floor = static_cast<int>(xc);
  const int x_ceil = (x_floor < width - 1) ? x_floor + 1 : x_floor;
  const float x_lerp = xc - static_cast<float>(x_floor);

  const int tl = (y_floor * width + x_floor) * num_channels;
  const int tr = (y_floor * width + x_ceil) * num_channels;
  const int bl = (y_ceil * width + x_floor) * num_channels;
  const int br = (y_ceil * width + x_ceil) * num_channels;

  for (int i = 0; i < n_channels; i++) {
    int c = channels[i];
    results[i] = (1.0f - y_lerp) * ((1.0f - x_lerp) * tensor[tl + c] +
                                      x_lerp * tensor[tr + c]) +
                 y_lerp * ((1.0f - x_lerp) * tensor[bl + c] +
                            x_lerp * tensor[br + c]);
  }
}

// 双线性插值采样 (单 channel)
inline float sample_tensor_single(const float* tensor, int height, int width,
                                  int num_channels, float y, float x, int channel) {
  float result;
  sample_tensor_multi(tensor, height, width, num_channels, y, x, &channel, 1, &result);
  return result;
}

// =========================================================================
// 构建邻接表
// =========================================================================
void build_adjacency_list(StaticAdjacencyList* adj) {
  for (int i = 0; i < kNumKeypoints; i++) {
    adj->num_children[i] = 0;
  }
  for (int k = 0; k < 32; k++) {
    int parent = kEdgeList[k].parent;
    int child = kEdgeList[k].child;
    int idx = adj->num_children[parent];
    if (idx < 8) {
      adj->child_ids[parent][idx] = child;
      adj->edge_ids[parent][idx] = k;
      adj->num_children[parent]++;
    }
  }
}

// =========================================================================
// 通过 mid_offset 找到相邻关键点位置
// =========================================================================
PosePoint find_displaced_position(
    const float* short_offsets, const float* mid_offsets,
    int height, int width,
    PosePoint source, int edge_id, int target_id,
    int refinement_steps) {

  float y = source.y;
  float x = source.x;

  // mid_offsets layout: [H, W, 2*2*kNumEdges] = [H, W, 64]
  // 4 blocks of 16: [fwd_Y(0..15)][fwd_X(0..15)][bwd_Y(0..15)][bwd_X(0..15)]
  // For forward edge (edge_id in [0,16)): Y channel = edge_id, X channel = 16 + edge_id
  // For backward edge (adjusted edge_id in [32,48)): Y channel = edge_id, X channel = 16 + edge_id
  int channels[2] = {edge_id, kNumEdges + edge_id};
  float offsets[2];

  sample_tensor_multi(mid_offsets, height, width, 2 * 2 * kNumEdges,
                      y, x, channels, 2, offsets);
  y = clamp_val(y + offsets[0], 0.0f, static_cast<float>(height - 1));
  x = clamp_val(x + offsets[1], 0.0f, static_cast<float>(width - 1));

  // short offset refinement
  channels[0] = target_id;
  channels[1] = kNumKeypoints + target_id;
  for (int i = 0; i < refinement_steps; i++) {
    sample_tensor_multi(short_offsets, height, width, 2 * kNumKeypoints,
                        y, x, channels, 2, offsets);
    y = clamp_val(y + offsets[0], 0.0f, static_cast<float>(height - 1));
    x = clamp_val(x + offsets[1], 0.0f, static_cast<float>(width - 1));
  }

  return PosePoint{y, x};
}

// =========================================================================
// 从 root 关键点出发，通过 graph traversal 解码整个 pose
// =========================================================================
void backtrack_decode_pose(
    const float* scores, const float* short_offsets, const float* mid_offsets,
    int height, int width,
    const KeypointCandidate& root,
    const StaticAdjacencyList& adj,
    int refinement_steps,
    PoseResult* pose) {

  static StaticPriorityQueue queue;
  queue.init();
  queue.push(root);

  bool decoded[kNumKeypoints];
  for (int i = 0; i < kNumKeypoints; i++) decoded[i] = false;

  while (!queue.empty()) {
    KeypointCandidate current = queue.pop();
    if (decoded[current.id]) continue;

    pose->keypoints[current.id].position = current.point;
    pose->keypoints[current.id].score = current.score;
    decoded[current.id] = true;

    for (int j = 0; j < adj.num_children[current.id]; j++) {
      int child_id = adj.child_ids[current.id][j];
      int edge_id = adj.edge_ids[current.id][j];
      if (decoded[child_id]) continue;

      // backward edge: edge_id >= kNumEdges 时，mid_offset 中的索引要跳过 kNumEdges
      int mid_edge_id = edge_id;
      if (edge_id >= kNumEdges) {
        mid_edge_id += kNumEdges;
      }

      PosePoint child_point = find_displaced_position(
          short_offsets, mid_offsets, height, width,
          current.point, mid_edge_id, child_id, refinement_steps);

      float child_score = sample_tensor_single(scores, height, width,
                                               kNumKeypoints,
                                               child_point.y, child_point.x,
                                               child_id);

      KeypointCandidate child_candidate;
      child_candidate.point = child_point;
      child_candidate.id = child_id;
      child_candidate.score = child_score;
      queue.push(child_candidate);
    }
  }
}

// =========================================================================
// 完整多人 PoseNet 后处理
// =========================================================================
int decode_posenet_multi_pose(
    const TfLiteTensor* heatmap_tensor,
    const TfLiteTensor* short_offset_tensor,
    const TfLiteTensor* mid_offset_tensor,
    PoseResult* results,
    int max_detections,
    float score_threshold,
    float nms_radius) {

  const int height = heatmap_tensor->dims->data[1];
  const int width = heatmap_tensor->dims->data[2];

  // Step 1: 反量化
  dequantize_tensor_int8(heatmap_tensor, heatmap_float, 1.0f);
  dequantize_tensor_int8(short_offset_tensor, short_offset_float,
                         1.0f / static_cast<float>(kPoseStride));
  dequantize_tensor_int8(mid_offset_tensor, mid_offset_float,
                         1.0f / static_cast<float>(kPoseStride));

  // Step 2: 构建邻接表
  StaticAdjacencyList adj;
  build_adjacency_list(&adj);

  // Step 3: 构建候选关键点队列 (local maximum filtering)
  const float min_score_logit = -logf(1.0f / (score_threshold + 1e-6f) - 1.0f);
  const int local_max_radius = 1;

  static StaticPriorityQueue root_queue;
  root_queue.init();

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      for (int kp = 0; kp < kNumKeypoints; kp++) {
        float score = heatmap_float[(row * width + col) * kNumKeypoints + kp];
        if (score < min_score_logit) continue;

        // 局部最大值检测
        bool is_max = true;
        int y_start = (row - local_max_radius > 0) ? row - local_max_radius : 0;
        int y_end = (row + local_max_radius + 1 < height) ? row + local_max_radius + 1 : height;
        for (int yr = y_start; yr < y_end && is_max; yr++) {
          int x_start = (col - local_max_radius > 0) ? col - local_max_radius : 0;
          int x_end = (col + local_max_radius + 1 < width) ? col + local_max_radius + 1 : width;
          for (int xr = x_start; xr < x_end; xr++) {
            if (heatmap_float[(yr * width + xr) * kNumKeypoints + kp] > score) {
              is_max = false;
              break;
            }
          }
        }

        if (is_max) {
          // short_offset refinement for root position
          int idx = (row * width + col) * 2 * kNumKeypoints;
          float dy = short_offset_float[idx + kp];
          float dx = short_offset_float[idx + kNumKeypoints + kp];
          float refined_y = clamp_val(static_cast<float>(row) + dy, 0.0f,
                                      static_cast<float>(height - 1));
          float refined_x = clamp_val(static_cast<float>(col) + dx, 0.0f,
                                      static_cast<float>(width - 1));

          KeypointCandidate candidate;
          candidate.point = PosePoint{refined_y, refined_x};
          candidate.id = kp;
          candidate.score = score;
          root_queue.push(candidate);
        }
      }
    }
  }

  // Step 4: 贪心多人解码
  const float nms_radius_block = nms_radius / static_cast<float>(kPoseStride);
  const float squared_nms = nms_radius_block * nms_radius_block;
  int pose_count = 0;

  // 记录已解码 pose 的 block space 坐标用于 NMS
  static PosePoint decoded_positions[kMaxDetections][kNumKeypoints];

  while (pose_count < max_detections && pose_count < kMaxDetections && !root_queue.empty()) {
    KeypointCandidate root = root_queue.pop();

    // NMS: 检查是否已被已有 pose 覆盖
    bool rejected = false;
    for (int p = 0; p < pose_count; p++) {
      float dy = root.point.y - decoded_positions[p][root.id].y;
      float dx = root.point.x - decoded_positions[p][root.id].x;
      if (dy * dy + dx * dx <= squared_nms) {
        rejected = true;
        break;
      }
    }
    if (rejected) continue;

    // Decode this pose
    PoseResult* pose = &results[pose_count];
    for (int k = 0; k < kNumKeypoints; k++) {
      pose->keypoints[k].position = PosePoint{-1.0f, -1.0f};
      pose->keypoints[k].score = -1e5f;
    }

    backtrack_decode_pose(heatmap_float, short_offset_float,
                          mid_offset_float, height, width,
                          root, adj, kMidShortRefinementSteps, pose);

    // 计算 pose score: sigmoid 所有 keypoint score，取平均
    float score_sum = 0.0f;
    for (int k = 0; k < kNumKeypoints; k++) {
      pose->keypoints[k].score = sigmoid_f(pose->keypoints[k].score);
      score_sum += pose->keypoints[k].score;
    }
    pose->pose_score = score_sum / kNumKeypoints;

    if (pose->pose_score < score_threshold) continue;

    // 保存 block space 坐标并转换为像素坐标
    for (int k = 0; k < kNumKeypoints; k++) {
      decoded_positions[pose_count][k] = pose->keypoints[k].position;
      pose->keypoints[k].position.y *= kPoseStride;
      pose->keypoints[k].position.x *= kPoseStride;
    }

    pose_count++;
  }

  return pose_count;
}

// =========================================================================
// 简化版单人 decode (fallback，仅用 heatmap + short_offset)
// =========================================================================
int decode_posenet_single_pose(
    const TfLiteTensor* heatmap_tensor,
    const TfLiteTensor* short_offset_tensor,
    PoseResult* result) {

  const int height = heatmap_tensor->dims->data[1];
  const int width = heatmap_tensor->dims->data[2];
  const int num_kp = heatmap_tensor->dims->data[3];

  dequantize_tensor_int8(heatmap_tensor, heatmap_float, 1.0f);
  dequantize_tensor_int8(short_offset_tensor, short_offset_float,
                         1.0f / static_cast<float>(kPoseStride));

  float score_sum = 0.0f;
  for (int kp = 0; kp < kNumKeypoints && kp < num_kp; kp++) {
    float best_score = -1e9f;
    int best_row = 0;
    int best_col = 0;

    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        float val = heatmap_float[(row * width + col) * num_kp + kp];
        if (val > best_score) {
          best_score = val;
          best_row = row;
          best_col = col;
        }
      }
    }

    // Offset refinement
    int idx = (best_row * width + best_col) * 2 * kNumKeypoints;
    float dy = short_offset_float[idx + kp];
    float dx = short_offset_float[idx + kNumKeypoints + kp];
    float refined_y = clamp_val(static_cast<float>(best_row) + dy, 0.0f,
                                static_cast<float>(height - 1));
    float refined_x = clamp_val(static_cast<float>(best_col) + dx, 0.0f,
                                static_cast<float>(width - 1));

    result->keypoints[kp].position.y = refined_y * kPoseStride;
    result->keypoints[kp].position.x = refined_x * kPoseStride;
    result->keypoints[kp].score = sigmoid_f(best_score);
    score_sum += result->keypoints[kp].score;
  }

  result->pose_score = score_sum / kNumKeypoints;
  return 1;
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
        for (; i < 31 && name[i] != 0; i++) {
            current_running_op[i] = name[i];
        }
        current_running_op[i] = 0;

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

// =========================================================================
// Main
// =========================================================================
int main(int argc, char** argv) {
  debug_stage = STAGE_INIT;
  current_op_index = -1;
  std::memset(inference_status_message, 0, sizeof(inference_status_message));
  std::memset(current_running_op, 0, sizeof(current_running_op));
  std::memset(debug_log_buffer, 0, sizeof(debug_log_buffer));
  std::memset((void*)keypoint_results, 0, sizeof(keypoint_results));
  std::memset((void*)pose_results, 0, sizeof(pose_results));

  debug_sp_addr = (uint32_t)&tensor_arena[0];
  debug_arena_start = (uint32_t)&tensor_arena[0];
  debug_model_addr = (uint32_t)g_posenet_075_353_481_int8_model_data;

  // -----------------------------------------------------------
  // Step 1: Magic Check
  // -----------------------------------------------------------
  debug_stage = STAGE_MAGIC_CHECK;
  const uint8_t* bmp_data = g_posenet_075_353_481_int8_model_data;

  if (bmp_data[4] != 'T' || bmp_data[5] != 'F' || bmp_data[6] != 'L' || bmp_data[7] != '3') {
      std::strncpy(inference_status_message, "FATAL: Bad Magic 'TFL3'!", 63);
      debug_stage = STAGE_ERROR;
      return -1;
  }

  const tflite::Model* model = tflite::GetModel(g_posenet_075_353_481_int8_model_data);
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
  // Step 3: Register Ops
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
  // Step 4.5: 加载输入图片 (uint8 -> int8 转换)
  // -----------------------------------------------------------
  TfLiteTensor* input = interpreter.input(0);
  const size_t input_bytes = input->bytes;

  if (input_bytes != g_test_input_353x481_image_data_size) {
      std::strncpy(inference_status_message, "Input size mismatch!", 63);
      debug_stage = STAGE_ERROR;
      return -1;
  }

  // 输入 tensor 是 int8 类型 (zero_point = -128, scale ≈ 0.00784)
  // 原始图片数据是 uint8 [0, 255]
  // 需要转换: int8_val = uint8_val - 128
  // 这等价于 Python 中: input_data = (img_array.astype(np.int16) - 128).astype(np.int8)
  if (input->type == kTfLiteInt8) {
    const uint8_t* src = g_test_input_353x481_image_data;
    int8_t* dst = input->data.int8;
    for (size_t i = 0; i < input_bytes; i++) {
      dst[i] = static_cast<int8_t>(static_cast<int16_t>(src[i]) - 128);
    }
  } else {
    // 如果输入仍是 uint8 类型 (不应该发生在 int8 模型中)，直接 memcpy
    std::memcpy(input->data.uint8, g_test_input_353x481_image_data, input_bytes);
  }

  std::strncpy(inference_status_message, "Input loaded", 63);
  debug_stage = STAGE_INPUT_LOADED;

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

  inference_cycles = end_cycles - start_cycles;
  debug_stage = STAGE_INVOKE_DONE;

 diag_num_outputs = (int32_t)interpreter.outputs_size();
  for (int i = 0; i < diag_num_outputs && i < 4; i++) {
    TfLiteTensor* t = interpreter.output(i);
    for (int d = 0; d < 4 && d < t->dims->size; d++) {
      diag_out_dims[i][d] = t->dims->data[d];
    }
    diag_out_scale[i] = t->params.scale;
    diag_out_zp[i] = t->params.zero_point;
  }
  // 看第一个 output 的 int8 原始值范围
  TfLiteTensor* t0 = interpreter.output(0);
  int8_t vmin = 127, vmax = -128;
  for (int i = 0; i < (int)t0->bytes; i++) {
    int8_t v = t0->data.int8[i];
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  diag_heatmap_min = vmin;
  diag_heatmap_max = vmax;

  // -----------------------------------------------------------
  // Step 6: 解析输出 - PoseNet 完整后处理
  // -----------------------------------------------------------
  output_tensor_count = (int32_t)interpreter.outputs_size();

  // PoseNet 无 decoder 模型输出:
  //   output[0]: heatmap       [1, 23, 31, 17]   - keypoint heatmaps (logits)
  //   output[1]: short_offsets [1, 23, 31, 34]   - short-range offsets
  //   output[2]: mid_offsets   [1, 23, 31, 64]   - mid-range offsets
  //   (output[3]: long_offsets  [1, 23, 31, 34] - optional, for instance masks)

  TfLiteTensor* heatmap_t = interpreter.output(0);
  heatmap_height = heatmap_t->dims->data[1];
  heatmap_width = heatmap_t->dims->data[2];

  // ★ 保存原始 int8 数据 (后处理前)
  raw_heatmap_size = (int32_t)heatmap_t->bytes;
  std::memcpy(raw_heatmap, heatmap_t->data.int8, heatmap_t->bytes);

  if (output_tensor_count >= 2) {
    TfLiteTensor* short_offset_t = interpreter.output(1);
    raw_short_offsets_size = (int32_t)short_offset_t->bytes;
    std::memcpy(raw_short_offsets, short_offset_t->data.int8, short_offset_t->bytes);
  }

  if (output_tensor_count >= 3) {
    TfLiteTensor* mid_offset_t = interpreter.output(2);
    raw_mid_offsets_size = (int32_t)mid_offset_t->bytes;
    std::memcpy(raw_mid_offsets, mid_offset_t->data.int8, mid_offset_t->bytes);
  }

  // 后处理解码
  if (output_tensor_count >= 3) {
    // 完整多人解码
    TfLiteTensor* short_offset_t = interpreter.output(1);
    TfLiteTensor* mid_offset_t = interpreter.output(2);

    num_poses_found = decode_posenet_multi_pose(
        heatmap_t, short_offset_t, mid_offset_t,
        pose_results, kMaxDetections,
        /*score_threshold=*/0.25f,
        /*nms_radius=*/20.0f);
  } else if (output_tensor_count >= 2) {
    // 仅有 heatmap + short_offset
    TfLiteTensor* short_offset_t = interpreter.output(1);
    num_poses_found = decode_posenet_single_pose(
        heatmap_t, short_offset_t, &pose_results[0]);
  } else {
    // 仅有 heatmap, 退化为纯 argmax
    dequantize_tensor_int8(heatmap_t, heatmap_float, 1.0f);
    const int h = heatmap_t->dims->data[1];
    const int w = heatmap_t->dims->data[2];
    const int nk = heatmap_t->dims->data[3];
    float score_sum = 0.0f;
    for (int kp = 0; kp < kNumKeypoints && kp < nk; kp++) {
      float best = -1e9f;
      int br = 0, bc = 0;
      for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
          float v = heatmap_float[(r * w + c) * nk + kp];
          if (v > best) { best = v; br = r; bc = c; }
        }
      }
      pose_results[0].keypoints[kp].position.y = static_cast<float>(br) * kPoseStride;
      pose_results[0].keypoints[kp].position.x = static_cast<float>(bc) * kPoseStride;
      pose_results[0].keypoints[kp].score = sigmoid_f(best);
      score_sum += pose_results[0].keypoints[kp].score;
    }
    pose_results[0].pose_score = score_sum / kNumKeypoints;
    num_poses_found = 1;
  }

  // 填充旧版 keypoint_results 结构 (兼容调试接口)
  if (num_poses_found > 0) {
    for (int i = 0; i < kNumKeypoints; i++) {
      keypoint_results[i].id = (uint8_t)i;
      keypoint_results[i].y = (int16_t)(pose_results[0].keypoints[i].position.y);
      keypoint_results[i].x = (int16_t)(pose_results[0].keypoints[i].position.x);
      float s = pose_results[0].keypoints[i].score;
      int8_t qs = (int8_t)clamp_val(static_cast<int>(s * 254.0f - 127.0f), -128, 127);
      keypoint_results[i].score = qs;
    }
    num_keypoints_found = kNumKeypoints;
  }

  // 找最高分 pose
  float best_pose_score = 0.0f;
  int best_pose_idx = 0;
  for (int i = 0; i < num_poses_found; i++) {
    if (pose_results[i].pose_score > best_pose_score) {
      best_pose_score = pose_results[i].pose_score;
      best_pose_idx = i;
    }
  }

  output_score = (int8_t)clamp_val(static_cast<int>(best_pose_score * 254.0f - 127.0f), -128, 127);
  output_class = best_pose_idx;

  debug_stage = STAGE_OUTPUT_PARSED;
  inference_status = 0;
  std::strncpy(inference_status_message, "Test Finished SUCCESS", 63);

  return 0;
}