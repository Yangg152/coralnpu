import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import heapq

# 模型和输入路径
MODEL_PATH = "posenet_075_353_481_int8.tflite"
IMAGE_PATH = "test_couple.jpg"
OUTPUT_PATH = "posenet_result.jpg"

NUM_KEYPOINTS = 17
NUM_EDGES = 16
LOCAL_MAXIMUM_RADIUS = 1

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 骨架连接 (用于画图)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# 32条边: 前16条forward, 后16条backward (与C++一致)
EDGE_LIST = [
    # Forward edges (0-15)
    (0, 1),   # nose -> left_eye
    (1, 3),   # left_eye -> left_ear
    (0, 2),   # nose -> right_eye
    (2, 4),   # right_eye -> right_ear
    (0, 5),   # nose -> left_shoulder
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    (5, 11),  # left_shoulder -> left_hip
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    (0, 6),   # nose -> right_shoulder
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    (6, 12),  # right_shoulder -> right_hip
    (12, 14), # right_hip -> right_knee
    (14, 16), # right_knee -> right_ankle
    # Backward edges (16-31)
    (1, 0),   # left_eye -> nose
    (3, 1),   # left_ear -> left_eye
    (2, 0),   # right_eye -> nose
    (4, 2),   # right_ear -> right_eye
    (5, 0),   # left_shoulder -> nose
    (7, 5),   # left_elbow -> left_shoulder
    (9, 7),   # left_wrist -> left_elbow
    (11, 5),  # left_hip -> left_shoulder
    (13, 11), # left_knee -> left_hip
    (15, 13), # left_ankle -> left_knee
    (6, 0),   # right_shoulder -> nose
    (8, 6),   # right_elbow -> right_shoulder
    (10, 8),  # right_wrist -> right_elbow
    (12, 6),  # right_hip -> right_shoulder
    (14, 12), # right_knee -> right_hip
    (16, 14), # right_ankle -> right_knee
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def logodds(x):
    return -np.log(1.0 / (x + 1e-6) - 1.0)


def build_adjacency_list():
    """构建邻接表: child_ids[node] = [child_nodes...], edge_ids[node] = [edge_ids...]"""
    child_ids = [[] for _ in range(NUM_KEYPOINTS)]
    edge_ids = [[] for _ in range(NUM_KEYPOINTS)]
    for k, (parent, child) in enumerate(EDGE_LIST):
        child_ids[parent].append(child)
        edge_ids[parent].append(k)
    return child_ids, edge_ids


def sample_tensor_bilinear(tensor, y, x, channels):
    """
    双线性插值采样 tensor[y, x, channels]
    tensor shape: (H, W, C)
    返回 len(channels) 个采样值
    """
    height, width, _ = tensor.shape
    y_clamped = np.clip(y, 0.0, height - 1.0)
    x_clamped = np.clip(x, 0.0, width - 1.0)

    y_floor = int(np.floor(y_clamped))
    y_ceil = int(np.ceil(y_clamped))
    x_floor = int(np.floor(x_clamped))
    x_ceil = int(np.ceil(x_clamped))

    y_lerp = y_clamped - y_floor
    x_lerp = x_clamped - x_floor

    results = []
    for c in channels:
        top_left = tensor[y_floor, x_floor, c]
        top_right = tensor[y_floor, x_ceil, c]
        bottom_left = tensor[y_ceil, x_floor, c]
        bottom_right = tensor[y_ceil, x_ceil, c]

        val = ((1 - y_lerp) * ((1 - x_lerp) * top_left + x_lerp * top_right) +
               y_lerp * ((1 - x_lerp) * bottom_left + x_lerp * bottom_right))
        results.append(val)
    return results


def find_displaced_position(short_offsets, mid_offsets, height, width,
                            source_y, source_x, edge_id, target_id,
                            mid_short_offset_refinement_steps):
    """
    沿 mid_offsets 移动，然后用 short_offsets 精修。
    所有坐标在 block space 中。
    mid_offsets shape: (H, W, 2*2*NUM_EDGES) = (H, W, 64)
      布局: [fwd_Y_0..15][fwd_X_0..15][bwd_Y_0..15][bwd_X_0..15]
    short_offsets shape: (H, W, 2*NUM_KEYPOINTS) = (H, W, 34)
      布局: [Y_0..16][X_0..16]
    """
    y = source_y
    x = source_x

    # mid_offsets 通道: edge_id 是 Y, NUM_EDGES + edge_id 是 X
    # 但 edge_id 如果 >= NUM_EDGES (backward), 需要额外加 NUM_EDGES
    # C++ 注释:
    #   The mid-offsets block is organized as 4 blocks of kNumEdges:
    #   [fwd Y offsets][fwd X offsets][bwd Y offsets][bwd X offsets]
    #   edge_id is [0,kNumEdges) for forward edges and
    #   [kNumEdges, 2*kNumEdges) for backward edges.
    #   Thus if the edge is a backward edge (>=kNumEdges) then we need
    #   to start 16 indices later to be correctly aligned with the mid-offsets.
    # 在 BacktrackDecodePose 里已经做了调整:
    #   if (edge_id >= kNumEdges) edge_id += kNumEdges;
    # 所以传入这里的 edge_id 已经调整过了。
    # mid_offsets channels: [edge_id] = Y, [NUM_EDGES + edge_id] = X
    # 但由于总共64通道，分4块:
    #   [0..15] = fwd Y
    #   [16..31] = fwd X
    #   [32..47] = bwd Y
    #   [48..63] = bwd X
    # 调整后: forward edge_id in [0,16), channels = [edge_id, 16+edge_id]
    #         backward edge_id in [32,48), channels = [edge_id, 16+edge_id]
    # 即统一 channels = [edge_id, NUM_EDGES + edge_id]

    mid_channels = [edge_id, NUM_EDGES + edge_id]
    offsets = sample_tensor_bilinear(mid_offsets, y, x, mid_channels)
    y = np.clip(y + offsets[0], 0.0, height - 1.0)
    x = np.clip(x + offsets[1], 0.0, width - 1.0)

    # short_offsets 精修
    short_channels = [target_id, NUM_KEYPOINTS + target_id]
    for _ in range(mid_short_offset_refinement_steps):
        offsets = sample_tensor_bilinear(short_offsets, y, x, short_channels)
        y = np.clip(y + offsets[0], 0.0, height - 1.0)
        x = np.clip(x + offsets[1], 0.0, width - 1.0)

    return y, x


def build_keypoint_queue(scores, short_offsets, height, width,
                         score_threshold, local_maximum_radius):
    """
    构建关键点优先队列 (分数递减)。
    scores shape: (H, W, 17) - 已是 logit 空间
    short_offsets shape: (H, W, 34) - block space
    """
    queue = []  # (-score, y_refined, x_refined, keypoint_id)

    for y in range(height):
        for x in range(width):
            for j in range(NUM_KEYPOINTS):
                score = scores[y, x, j]
                if score < score_threshold:
                    continue

                # 检查局部最大值
                local_maximum = True
                y_start = max(y - local_maximum_radius, 0)
                y_end = min(y + local_maximum_radius + 1, height)
                for y_curr in range(y_start, y_end):
                    x_start = max(x - local_maximum_radius, 0)
                    x_end = min(x + local_maximum_radius + 1, width)
                    for x_curr in range(x_start, x_end):
                        if scores[y_curr, x_curr, j] > score:
                            local_maximum = False
                            break
                    if not local_maximum:
                        break

                if local_maximum:
                    # 用 short offset 精修位置
                    dy = short_offsets[y, x, j]
                    dx = short_offsets[y, x, j + NUM_KEYPOINTS]
                    y_refined = np.clip(y + dy, 0.0, height - 1.0)
                    x_refined = np.clip(x + dx, 0.0, width - 1.0)
                    # 用负分数实现最大堆
                    heapq.heappush(queue, (-score, y_refined, x_refined, j))

    return queue


def backtrack_decode_pose(scores, short_offsets, mid_offsets, height, width,
                          root_point_y, root_point_x, root_id, root_score,
                          child_ids_adj, edge_ids_adj,
                          mid_short_offset_refinement_steps):
    """
    从根关键点出发，用优先队列 + BFS 解码整个人体姿态。
    """
    pose_keypoints = np.full((NUM_KEYPOINTS, 2), -1.0, dtype=np.float32)  # (y, x)
    keypoint_scores = np.full(NUM_KEYPOINTS, -1e5, dtype=np.float32)

    # 优先队列
    decode_queue = []  # (-score, y, x, kp_id)
    heapq.heappush(decode_queue, (-root_score, root_point_y, root_point_x, root_id))

    keypoint_decoded = [False] * NUM_KEYPOINTS

    while decode_queue:
        neg_score, cur_y, cur_x, cur_id = heapq.heappop(decode_queue)
        cur_score = -neg_score

        if keypoint_decoded[cur_id]:
            continue

        pose_keypoints[cur_id] = [cur_y, cur_x]
        keypoint_scores[cur_id] = cur_score
        keypoint_decoded[cur_id] = True

        # 遍历当前节点的子节点
        for child_id, edge_id in zip(child_ids_adj[cur_id], edge_ids_adj[cur_id]):
            if keypoint_decoded[child_id]:
                continue

            # 调整 edge_id: backward edge (>= NUM_EDGES) 需要加 NUM_EDGES
            adjusted_edge_id = edge_id
            if edge_id >= NUM_EDGES:
                adjusted_edge_id += NUM_EDGES

            child_y, child_x = find_displaced_position(
                short_offsets, mid_offsets, height, width,
                cur_y, cur_x, adjusted_edge_id, child_id,
                mid_short_offset_refinement_steps)

            # 在目标位置采样 score
            child_score = sample_tensor_bilinear(scores, child_y, child_x, [child_id])[0]
            heapq.heappush(decode_queue, (-child_score, child_y, child_x, child_id))

    return pose_keypoints, keypoint_scores


def pass_keypoint_nms(poses, keypoint_id, point_y, point_x, squared_nms_radius):
    """检查新关键点是否与已有 pose 的同类关键点距离足够远"""
    for pose in poses:
        dy = point_y - pose['keypoints'][keypoint_id, 0]
        dx = point_x - pose['keypoints'][keypoint_id, 1]
        if dy * dy + dx * dx <= squared_nms_radius:
            return False
    return True


def decode_all_poses(scores, short_offsets, mid_offsets, height, width,
                     max_detections=10, score_threshold=0.25,
                     mid_short_offset_refinement_steps=5,
                     nms_radius=20, stride=16):
    """
    完整多人姿态解码 (对应 C++ DecodeAllPoses)
    scores, short_offsets, mid_offsets 均已在 block space 中 (反量化 + 除以 stride)
    """
    # score_threshold 转为 logit
    min_score_logit = logodds(score_threshold)

    # 构建优先队列
    queue = build_keypoint_queue(scores, short_offsets, height, width,
                                min_score_logit, LOCAL_MAXIMUM_RADIUS)

    # 构建邻接表
    child_ids_adj, edge_ids_adj = build_adjacency_list()

    # nms_radius 转到 block space
    nms_radius_block = nms_radius / stride
    squared_nms_radius = nms_radius_block * nms_radius_block

    poses = []

    while len(poses) < max_detections and queue:
        neg_score, root_y, root_x, root_id = heapq.heappop(queue)
        root_score = -neg_score

        # NMS 检查
        if not pass_keypoint_nms(poses, root_id, root_y, root_x, squared_nms_radius):
            continue

        # 解码整个 pose
        pose_keypoints, keypoint_scores_raw = backtrack_decode_pose(
            scores, short_offsets, mid_offsets, height, width,
            root_y, root_x, root_id, root_score,
            child_ids_adj, edge_ids_adj,
            mid_short_offset_refinement_steps)

        # 将 keypoint scores 从 logit 转为概率
        kp_scores_prob = sigmoid(keypoint_scores_raw)

        # 计算 instance score (top-k 平均, k=NUM_KEYPOINTS)
        sorted_scores = np.sort(kp_scores_prob)[::-1]
        instance_score = np.mean(sorted_scores[:NUM_KEYPOINTS])

        if instance_score >= score_threshold:
            poses.append({
                'keypoints': pose_keypoints,      # block space
                'scores': kp_scores_prob,
                'instance_score': instance_score
            })

    # Soft keypoint NMS 重新打分
    for i in range(len(poses)):
        occluded = np.zeros(NUM_KEYPOINTS, dtype=bool)
        for j in range(i):
            # 检查 pose i 的关键点是否与更高分 pose j 的对应关键点重叠
            for k in range(NUM_KEYPOINTS):
                dy = poses[i]['keypoints'][k, 0] - poses[j]['keypoints'][k, 0]
                dx = poses[i]['keypoints'][k, 1] - poses[j]['keypoints'][k, 1]
                if dy * dy + dx * dx <= squared_nms_radius:
                    occluded[k] = True

        # 重新计算 instance score
        sorted_indices = np.argsort(poses[i]['scores'])[::-1]
        total = 0.0
        for k in range(NUM_KEYPOINTS):
            idx = sorted_indices[k]
            if not occluded[idx]:
                total += poses[i]['scores'][idx]
        poses[i]['instance_score'] = total / NUM_KEYPOINTS

    # 按 instance_score 降序排列，过滤低分
    poses.sort(key=lambda p: p['instance_score'], reverse=True)
    poses = [p for p in poses if p['instance_score'] >= score_threshold]

    # 将坐标转到像素空间
    for pose in poses:
        pose['keypoints'] *= stride

    return poses


# ============================================================
# 主流程
# ============================================================

# 加载模型
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]  # 353
input_width = input_details[0]['shape'][2]   # 481
stride = input_height // output_details[0]['quantization_parameters']['zero_points'].shape[0]
# 更可靠的方式: stride = input_height / output_height
output_height = output_details[0]['shape'][1]
output_width = output_details[0]['shape'][2]
stride = round(input_height / output_height)  # 应该是 16 (353/23 ≈ 15.3, 但 PoseNet 标准 stride=16)
stride = 16  # PoseNet 固定 stride

print(f"Input: {input_height}x{input_width}")
print(f"Output: {output_height}x{output_width}")
print(f"Stride: {stride}")

# 加载原图
img_orig = Image.open(IMAGE_PATH)
orig_width, orig_height = img_orig.size

# 预处理
img_resized = img_orig.resize((input_width, input_height))
img_array = np.array(img_resized)

input_dtype = input_details[0]['dtype']
if input_dtype == np.int8:
    input_data = (img_array.astype(np.int16) - 128).astype(np.int8)
elif input_dtype == np.uint8:
    input_data = img_array.astype(np.uint8)
else:
    input_data = (img_array / 255.0).astype(np.float32)

input_data = np.expand_dims(input_data, axis=0)

# 推理
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 获取输出并反量化
# Output 0: heatmaps (1, 23, 31, 17) - logits
# Output 1: short_offsets (1, 23, 31, 34) - 像素单位, 需要除以stride转block space
# Output 2: mid_offsets (1, 23, 31, 64) - 像素单位, 需要除以stride转block space

heatmaps_raw = interpreter.get_tensor(output_details[0]['index'])
offsets_raw = interpreter.get_tensor(output_details[1]['index'])
mid_offsets_raw = interpreter.get_tensor(output_details[2]['index'])

# 反量化 heatmaps (不需要额外 scale)
hm_qp = output_details[0]['quantization_parameters']
heatmaps = (heatmaps_raw.astype(np.float32) - hm_qp['zero_points'][0]) * hm_qp['scales'][0]

# 反量化 short_offsets, 然后除以 stride (转到 block space)
off_qp = output_details[1]['quantization_parameters']
short_offsets = (offsets_raw.astype(np.float32) - off_qp['zero_points'][0]) * off_qp['scales'][0] / stride

# 反量化 mid_offsets, 然后除以 stride (转到 block space)
mid_qp = output_details[2]['quantization_parameters']
mid_offsets = (mid_offsets_raw.astype(np.float32) - mid_qp['zero_points'][0]) * mid_qp['scales'][0] / stride

# 去除 batch 维度
heatmaps = heatmaps[0]       # (23, 31, 17)
short_offsets = short_offsets[0]  # (23, 31, 34)
mid_offsets = mid_offsets[0]      # (23, 31, 64)

print(f"Heatmaps shape: {heatmaps.shape}")
print(f"Short offsets shape: {short_offsets.shape}")
print(f"Mid offsets shape: {mid_offsets.shape}")
print(f"Heatmaps range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")

# 解码
poses = decode_all_poses(
    heatmaps, short_offsets, mid_offsets,
    output_height, output_width,
    max_detections=10,
    score_threshold=0.2,
    mid_short_offset_refinement_steps=5,
    nms_radius=20,
    stride=stride
)

print(f"\n检测到 {len(poses)} 个人")

# 可视化
img_draw = img_orig.copy()
draw = ImageDraw.Draw(img_draw)

COLORS = [
    ((255, 0, 0), (0, 255, 0)),     # 红点 绿线
    ((0, 0, 255), (255, 165, 0)),   # 蓝点 橙线
    ((255, 0, 255), (0, 255, 255)), # 紫点 青线
    ((255, 255, 0), (128, 0, 128)), # 黄点 紫线
]

THRESHOLD = 0.2

for person_id, pose in enumerate(poses):
    color_idx = person_id % len(COLORS)
    point_color = COLORS[color_idx][0]
    line_color = COLORS[color_idx][1]

    # pose['keypoints'] 在像素空间 (相对于输入图 353x481)
    # 转到原图坐标
    coords = pose['keypoints'].copy()  # (17, 2): (y, x) in input image pixels
    kp_scores = pose['scores']

    orig_coords_y = coords[:, 0] * orig_height / input_height
    orig_coords_x = coords[:, 1] * orig_width / input_width

    # 画骨架
    for (i, j) in SKELETON:
        if kp_scores[i] > THRESHOLD and kp_scores[j] > THRESHOLD:
            x1, y1 = orig_coords_x[i], orig_coords_y[i]
            x2, y2 = orig_coords_x[j], orig_coords_y[j]
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=3)

    # 画关键点
    for kp_id in range(NUM_KEYPOINTS):
        if kp_scores[kp_id] > THRESHOLD:
            x = orig_coords_x[kp_id]
            y = orig_coords_y[kp_id]
            r = 5
            draw.ellipse([(x - r, y - r), (x + r, y + r)],
                         fill=point_color, outline=(255, 255, 255))

    print(f"\n人物 {person_id + 1} (instance_score={pose['instance_score']:.3f}):")
    for kp_id in range(NUM_KEYPOINTS):
        s = kp_scores[kp_id]
        status = "✓" if s > THRESHOLD else "✗"
        print(f"  {status} {KEYPOINT_NAMES[kp_id]:16s}  ({orig_coords_x[kp_id]:6.1f}, {orig_coords_y[kp_id]:6.1f})  score={s:.3f}")

# 保存
img_draw.save(OUTPUT_PATH)
print(f"\n结果已保存到: {OUTPUT_PATH}")
