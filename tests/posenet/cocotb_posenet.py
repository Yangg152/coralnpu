# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

import cocotb
import sys
import struct
from cocotb.triggers import ClockCycles
from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles

STAGE_MAP = {
    0: "INIT",
    1: "MAGIC_CHECK",
    2: "STRUCTURE_CHECK",
    3: "OPS_REGISTERED",
    4: "ALLOCATE_START",
    5: "ALLOCATE_DONE",
    55: "INPUT_LOADED",
    6: "INVOKE_START",
    8: "INVOKE_DONE",
    9: "OUTPUT_PARSED",
    99: "ERROR"
}

KEYPOINT_NAMES = [
    'NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'
]

# 参考结果 (从 posenet_mobilenet_v1_075_353_481_16_quant_decoder_reference.csv)
REFERENCE_POSES = [
    {
        'pose_score': 0.89025635,
        'keypoints': [
            {'name': 'NOSE',           'score': 0.98346347, 'x': 203.6254,  'y': 97.58575},
            {'name': 'LEFT_EYE',       'score': 0.7460481,  'x': 204.85681, 'y': 93.172226},
            {'name': 'RIGHT_EYE',      'score': 0.9811637,  'x': 198.64923, 'y': 93.115715},
            {'name': 'LEFT_EAR',       'score': 0.027278204,'x': 205.33545, 'y': 96.13378},
            {'name': 'RIGHT_EAR',      'score': 0.9885768,  'x': 185.00697, 'y': 95.23009},
            {'name': 'LEFT_SHOULDER',  'score': 0.99565434, 'x': 210.94872, 'y': 123.0651},
            {'name': 'RIGHT_SHOULDER', 'score': 0.99710876, 'x': 169.0196,  'y': 127.60784},
            {'name': 'LEFT_ELBOW',     'score': 0.98094475, 'x': 221.25198, 'y': 161.45421},
            {'name': 'RIGHT_ELBOW',    'score': 0.99468464, 'x': 165.06767, 'y': 165.89093},
            {'name': 'LEFT_WRIST',     'score': 0.9902507,  'x': 222.60283, 'y': 192.184},
            {'name': 'RIGHT_WRIST',    'score': 0.9918166,  'x': 159.34328, 'y': 203.07043},
            {'name': 'LEFT_HIP',       'score': 0.9964401,  'x': 206.2957,  'y': 197.58525},
            {'name': 'RIGHT_HIP',      'score': 0.9954336,  'x': 185.0755,  'y': 199.01413},
            {'name': 'LEFT_KNEE',      'score': 0.98581785, 'x': 206.33186, 'y': 242.46378},
            {'name': 'RIGHT_KNEE',     'score': 0.9810116,  'x': 189.19565, 'y': 242.3359},
            {'name': 'LEFT_ANKLE',     'score': 0.6927644,  'x': 198.82545, 'y': 281.08133},
            {'name': 'RIGHT_ANKLE',    'score': 0.80590034, 'x': 193.33891, 'y': 281.91632},
        ]
    },
    {
        'pose_score': 0.83582795,
        'keypoints': [
            {'name': 'NOSE',           'score': 0.9871651,  'x': 257.9147,  'y': 105.13625},
            {'name': 'LEFT_EYE',       'score': 0.99250984, 'x': 263.05246, 'y': 100.76674},
            {'name': 'RIGHT_EYE',      'score': 0.60081387, 'x': 256.27286, 'y': 99.909096},
            {'name': 'LEFT_EAR',       'score': 0.9919642,  'x': 277.13528, 'y': 103.992805},
            {'name': 'RIGHT_EAR',      'score': 0.023253076,'x': 255.78479, 'y': 103.75027},
            {'name': 'LEFT_SHOULDER',  'score': 0.9973737,  'x': 290.0752,  'y': 129.24666},
            {'name': 'RIGHT_SHOULDER', 'score': 0.9954184,  'x': 256.33075, 'y': 127.63359},
            {'name': 'LEFT_ELBOW',     'score': 0.9890182,  'x': 300.71857, 'y': 156.12056},
            {'name': 'RIGHT_ELBOW',    'score': 0.97245365, 'x': 247.20091, 'y': 161.64403},
            {'name': 'LEFT_WRIST',     'score': 0.98531264, 'x': 309.1438,  'y': 183.21922},
            {'name': 'RIGHT_WRIST',    'score': 0.74013567, 'x': 237.23274, 'y': 192.07008},
            {'name': 'LEFT_HIP',       'score': 0.9969537,  'x': 286.61514, 'y': 190.32948},
            {'name': 'RIGHT_HIP',      'score': 0.9957957,  'x': 264.94116, 'y': 191.60785},
            {'name': 'LEFT_KNEE',      'score': 0.98287255, 'x': 282.4872,  'y': 234.24887},
            {'name': 'RIGHT_KNEE',     'score': 0.9265601,  'x': 275.66415, 'y': 234.81589},
            {'name': 'LEFT_ANKLE',     'score': 0.5236073,  'x': 283.315,   'y': 278.22263},
            {'name': 'RIGHT_ANKLE',    'score': 0.5078647,  'x': 284.92114, 'y': 278.03003},
        ]
    }
]


@cocotb.test()
async def core_mini_rvv_posenet(dut):
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()

    async def read_u32(symbol):
        raw = await fixture.read(symbol, 4)
        return int.from_bytes(bytes(raw), byteorder='little')

    async def read_i32(symbol):
        raw = await fixture.read(symbol, 4)
        return int.from_bytes(bytes(raw), byteorder='little', signed=True)

    async def read_str(symbol, max_len=64):
        raw = await fixture.read(symbol, max_len)
        return bytes(raw).split(b'\0')[0].decode(errors='replace')

    async def read_f32(symbol):
        raw = await fixture.read(symbol, 4)
        return struct.unpack('<f', bytes(raw))[0]

    async def monitor_profiler():
        print(f"\n{'='*90}", flush=True)
        print(f"DEBUG MONITOR ACTIVATED (PoseNet Inference)", flush=True)
        print(f"{'='*90}", flush=True)

        last_status_msg = ""
        last_stage = -1
        stage_stuck_counter = 0
        SAMPLE_PERIOD = 20000
        total_monitored_cycles = 0

        while True:
            await ClockCycles(dut.io_aclk, SAMPLE_PERIOD)
            total_monitored_cycles += SAMPLE_PERIOD

            if total_monitored_cycles % 5_000_000 == 0:
                sys.stdout.write(f"\r{' '*90}\r")
                print(f"[HEARTBEAT] Simulation Running... Total Cycles: {total_monitored_cycles/1_000_000:.1f} M")

            try:
                stage_code = await read_i32('debug_stage')
                op_idx = await read_i32('current_op_index')
                status_msg = await read_str('inference_status_message')
                current_op_name = await read_str('current_running_op', 32)

                if stage_code == last_stage:
                    stage_stuck_counter += 1
                else:
                    stage_stuck_counter = 0
                    last_stage = stage_code

                if stage_code == 4 and stage_stuck_counter % 200 == 0 and stage_stuck_counter > 0:
                    elapsed_cycles = stage_stuck_counter * SAMPLE_PERIOD
                    sys.stdout.write(f"\r{' '*90}\r")
                    print(f"[INFO] Memory Allocating... {elapsed_cycles/1000000:.1f}M cycles elapsed")

                if status_msg != last_status_msg and len(status_msg) > 0:
                    sys.stdout.write(f"\r{' '*90}\r")
                    print(f"[STATUS] {STAGE_MAP.get(stage_code, 'UNKNOWN')}: {status_msg}")
                    last_status_msg = status_msg

                stage_str = STAGE_MAP.get(stage_code, f"CODE_{stage_code}")
                spinner = "|/-\\"[(stage_stuck_counter // 5) % 4]
                op_display = ""
                if stage_code == 6:
                    op_display = f"Op:{op_idx} [{current_op_name}]"

                sys.stdout.write(f"\r {spinner} [{stage_str}] {op_display} | Msg: {status_msg[:30]}")
                sys.stdout.flush()

            except Exception:
                pass

    elf_file = 'posenet_int8_binary.elf'
    elf_path = r.Rlocation('coralnpu_hw/tests/posenet/' + elf_file)

    await fixture.load_elf_and_lookup_symbols(
        elf_path,
        [
            'inference_status', 'inference_status_message',
            'debug_stage', 'current_op_index', 'current_running_op',
            'inference_cycles',
            'op_logs', 'op_log_count', 'debug_log_buffer',
            'keypoint_results', 'num_keypoints_found',
            'output_tensor_count', 'heatmap_height', 'heatmap_width',
            'num_poses_found', 'pose_results',
            # 诊断
            'diag_num_outputs', 'diag_out_dims', 'diag_out_scale',
            'diag_out_zp', 'diag_heatmap_min', 'diag_heatmap_max',
            'raw_heatmap', 'raw_short_offsets', 'raw_mid_offsets',
            'raw_heatmap_size', 'raw_short_offsets_size', 'raw_mid_offsets_size',
        ])

    monitor_task = cocotb.start_soon(monitor_profiler())

    halt_cycle_count = await fixture.run_to_halt(timeout_cycles=500_000_000_00)
    monitor_task.kill()
    print(f"\n\nExecution halted at: {halt_cycle_count} cycles", flush=True)

    # ===== 读取结果 =====
    raw_status = await fixture.read('inference_status', 1)
    status = int.from_bytes(bytes(raw_status), byteorder='little', signed=True)
    msg = await read_str('inference_status_message')
    total_cycles = await read_u32('inference_cycles')

    # ===== 读取 heatmap 维度 (后续多处使用) =====
    hm_h = await read_i32('heatmap_height')
    hm_w = await read_i32('heatmap_width')
    num_outputs = await read_i32('output_tensor_count')
    num_poses = await read_i32('num_poses_found')

    # ===== 算子性能表 =====
    if status == 0:
        log_count = await read_u32('op_log_count')
        print("\n" + "="*60)
        print(f" PER-OPERATOR PERFORMANCE REPORT (Total: {log_count} ops)")
        print("="*60)
        print(f"{'Idx':<4} | {'Operator Name':<25} | {'Cycles':>12} | {'% Total':>7}")
        print("-" * 60)

        STRUCT_SIZE = 36  # 32 bytes name + 4 bytes cycles
        raw_logs = await fixture.read('op_logs', STRUCT_SIZE * log_count)

        sum_cycles = 0
        for i in range(log_count):
            offset = i * STRUCT_SIZE
            entry_bytes = raw_logs[offset: offset + STRUCT_SIZE]
            op_name_bytes, cycles = struct.unpack('<32sI', entry_bytes)
            op_name = op_name_bytes.split(b'\0')[0].decode(errors='replace')
            sum_cycles += cycles
            pct = (cycles / total_cycles * 100) if total_cycles > 0 else 0
            print(f"{i:<4} | {op_name:<25} | {cycles:>12,} | {pct:>6.1f}%")

        print("-" * 60)
        print(f"{'SUM':<4} | {'All Operators':<25} | {sum_cycles:>12,} |")
        print(f"{'TOT':<4} | {'Total Inference':<25} | {total_cycles:>12,} |")
        print("=" * 60)

    # ===== 诊断 output tensor =====
    if status == 0:
        n_out = await read_i32('diag_num_outputs')
        print(f"\n{'='*60}")
        print(f" OUTPUT TENSOR DIAGNOSTICS ({n_out} outputs)")
        print(f"{'='*60}")

        # diag_out_dims: 4 outputs x 4 dims = 16 x int32 = 64 bytes
        raw_dims = await fixture.read('diag_out_dims', 64)
        # diag_out_scale: 4 x float32 = 16 bytes
        raw_scale = await fixture.read('diag_out_scale', 16)
        # diag_out_zp: 4 x int32 = 16 bytes
        raw_zp = await fixture.read('diag_out_zp', 16)

        for i in range(min(n_out, 4)):
            dims = struct.unpack_from('<4i', bytes(raw_dims), i * 16)
            scale = struct.unpack_from('<f', bytes(raw_scale), i * 4)[0]
            zp = struct.unpack_from('<i', bytes(raw_zp), i * 4)[0]
            print(f"  Output[{i}]: shape=({dims[0]}, {dims[1]}, {dims[2]}, {dims[3]})  "
                  f"scale={scale:.8f}  zero_point={zp}")

        hm_min = await read_i32('diag_heatmap_min')
        hm_max = await read_i32('diag_heatmap_max')
        print(f"  Output[0] int8 raw range: [{hm_min}, {hm_max}]")
        print(f"{'='*60}")

    # ===== 导出原始 int8 tensor 数据 =====
    if status == 0:
        hm_size = await read_i32('raw_heatmap_size')
        so_size = await read_i32('raw_short_offsets_size')
        mo_size = await read_i32('raw_mid_offsets_size')

        print(f"\n Raw tensor sizes: heatmap={hm_size}, short_offsets={so_size}, mid_offsets={mo_size}")

        try:
            import numpy as np

            if hm_size > 0 and hm_h > 0 and hm_w > 0:
                raw_hm = await fixture.read('raw_heatmap', hm_size)
                hm_array = np.frombuffer(bytes(raw_hm), dtype=np.int8)
                np.save('/tmp/posenet_raw_heatmap.npy', hm_array.reshape(hm_h, hm_w, 17))
                print(f"  Saved: /tmp/posenet_raw_heatmap.npy  shape=({hm_h},{hm_w},17)")

            if so_size > 0 and hm_h > 0 and hm_w > 0:
                raw_so = await fixture.read('raw_short_offsets', so_size)
                so_array = np.frombuffer(bytes(raw_so), dtype=np.int8)
                np.save('/tmp/posenet_raw_short_offsets.npy', so_array.reshape(hm_h, hm_w, 34))
                print(f"  Saved: /tmp/posenet_raw_short_offsets.npy  shape=({hm_h},{hm_w},34)")

            if mo_size > 0 and hm_h > 0 and hm_w > 0:
                raw_mo = await fixture.read('raw_mid_offsets', mo_size)
                mo_array = np.frombuffer(bytes(raw_mo), dtype=np.int8)
                np.save('/tmp/posenet_raw_mid_offsets.npy', mo_array.reshape(hm_h, hm_w, 64))
                print(f"  Saved: /tmp/posenet_raw_mid_offsets.npy  shape=({hm_h},{hm_w},64)")

            print(f"  These are the raw int8 values BEFORE dequantization and post-processing.")
            print(f"  To dequantize: float_val = (int8_val - zero_point) * scale")
        except ImportError:
            print("  [WARN] numpy not available, skipping raw tensor export.")
        except Exception as e:
            print(f"  [WARN] Raw tensor export failed: {e}")

    # ===== Pose 结果输出与参考对比 =====
    if status == 0:
        print(f"\n{'='*80}")
        print(f" POSENET MULTI-POSE RESULTS")
        print(f" Heatmap: {hm_h}x{hm_w}, Output tensors: {num_outputs}, Poses found: {num_poses}")
        print(f"{'='*80}")

        # PoseResult 结构体布局:
        #   float pose_score;                    // 4 bytes
        #   PoseKeypointResult keypoints[17];    // 17 * (float y, float x, float score) = 17*12 = 204 bytes
        # Total per pose: 4 + 204 = 208 bytes
        POSE_RESULT_SIZE = 4 + 17 * 12  # 208 bytes

        all_poses = []
        if num_poses > 0:
            raw_poses = await fixture.read('pose_results', POSE_RESULT_SIZE * num_poses)

            for p in range(num_poses):
                pose_offset = p * POSE_RESULT_SIZE
                pose_score = struct.unpack_from('<f', bytes(raw_poses), pose_offset)[0]

                keypoints = []
                for k in range(17):
                    kp_offset = pose_offset + 4 + k * 12
                    ky, kx, ks = struct.unpack_from('<fff', bytes(raw_poses), kp_offset)
                    keypoints.append({'y': ky, 'x': kx, 'score': ks})

                all_poses.append({'pose_score': pose_score, 'keypoints': keypoints})

        # 打印结果 (CSV 兼容格式)
        if all_poses:
            print(f"\n{'pose_id':<8},{'pose_score':<12},{'keypoint_label':<18},{'kp_score':<12},{'kp_x':<12},{'kp_y':<12}")
            print("-" * 80)
            for p_idx, pose in enumerate(all_poses):
                for k_idx, kp in enumerate(pose['keypoints']):
                    name = KEYPOINT_NAMES[k_idx]
                    print(f"{p_idx:<8},{pose['pose_score']:<12.6f},{name:<18},{kp['score']:<12.6f},{kp['x']:<12.4f},{kp['y']:<12.4f}")

        # ===== 与参考结果对比 =====
        print(f"\n{'='*80}")
        print(f" ACCURACY COMPARISON vs REFERENCE")
        print(f"{'='*80}")

        expected_num_poses = len(REFERENCE_POSES)
        print(f" Expected poses: {expected_num_poses}, Detected poses: {num_poses}")

        if num_poses == 0:
            print(" WARNING: No poses detected!")
        else:
            num_compare = min(num_poses, expected_num_poses)
            score_threshold = 0.4
            position_tolerance = 30.0

            total_checked = 0
            total_passed = 0

            for p_idx in range(num_compare):
                ref_pose = REFERENCE_POSES[p_idx]
                det_pose = all_poses[p_idx]

                print(f"\n Pose {p_idx}: ref_score={ref_pose['pose_score']:.4f}, "
                      f"det_score={det_pose['pose_score']:.4f}, "
                      f"score_diff={abs(ref_pose['pose_score'] - det_pose['pose_score']):.4f}")
                print(f"  {'Keypoint':<18} | {'Ref(x,y)':<20} | {'Det(x,y)':<20} | "
                      f"{'Dist':>6} | {'RefScr':>7} | {'DetScr':>7} | {'Status'}")
                print(f"  {'-'*95}")

                for k_idx in range(17):
                    ref_kp = ref_pose['keypoints'][k_idx]
                    det_kp = det_pose['keypoints'][k_idx]
                    name = KEYPOINT_NAMES[k_idx]

                    ref_x, ref_y = ref_kp['x'], ref_kp['y']
                    det_x, det_y = det_kp['x'], det_kp['y']
                    ref_s = ref_kp['score']
                    det_s = det_kp['score']

                    dist = ((ref_x - det_x)**2 + (ref_y - det_y)**2)**0.5

                    if ref_s > score_threshold and det_s > score_threshold:
                        total_checked += 1
                        passed = dist < position_tolerance
                        if passed:
                            total_passed += 1
                        status_str = "PASS" if passed else f"FAIL (dist={dist:.1f})"
                    elif ref_s <= score_threshold:
                        status_str = "SKIP (ref low)"
                    else:
                        status_str = "SKIP (det low)"

                    print(f"  {name:<18} | ({ref_x:7.1f},{ref_y:7.1f}) | "
                          f"({det_x:7.1f},{det_y:7.1f}) | {dist:6.1f} | "
                          f"{ref_s:7.4f} | {det_s:7.4f} | {status_str}")

            # 总结
            print(f"\n{'='*80}")
            print(f" ACCURACY SUMMARY")
            print(f"{'='*80}")
            print(f"  Keypoints checked (both scores > {score_threshold}): {total_checked}")
            print(f"  Keypoints passed (dist < {position_tolerance}px): {total_passed}")
            if total_checked > 0:
                accuracy = total_passed / total_checked * 100
                print(f"  Accuracy: {accuracy:.1f}%")
                min_pass_rate = 50.0
                print(f"  Required pass rate: {min_pass_rate}%")
                if accuracy < min_pass_rate:
                    print(f"  WARNING: Accuracy below threshold!")
            print(f"{'='*80}")

    # ===== 旧版兼容输出 (简化) =====
    if status == 0:
        num_kp = await read_i32('num_keypoints_found')
        if num_kp > 0:
            print(f"\n Legacy keypoint_results (pose 0, pixel coords):")
            KP_STRUCT_SIZE = 6  # int16 y, int16 x, int8 score, uint8 id
            raw_kp = await fixture.read('keypoint_results', KP_STRUCT_SIZE * num_kp)
            for i in range(num_kp):
                offset = i * KP_STRUCT_SIZE
                entry = raw_kp[offset: offset + KP_STRUCT_SIZE]
                y, x, score_raw, kp_id = struct.unpack('<hhbB', entry)
                name = KEYPOINT_NAMES[kp_id] if kp_id < 17 else f"KP_{kp_id}"
                print(f"   {name:<18}: ({x:>4}, {y:>4})  score_q={score_raw}")

    # ===== 可视化输出 =====
    if status == 0 and num_poses > 0 and len(all_poses) > 0:
        try:
            from PIL import Image, ImageDraw

            SKELETON_EDGES = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (0, 5), (0, 6),
                (5, 7), (7, 9),
                (6, 8), (8, 10),
                (5, 11), (6, 12),
                (11, 12),
                (11, 13), (13, 15),
                (12, 14), (14, 16),
            ]

            POSE_COLORS = [
                (255, 50, 50),
                (50, 255, 50),
                (50, 100, 255),
                (255, 255, 50),
                (255, 50, 255),
                (50, 255, 255),
            ]

            # 加载原始输入图片作为背景
            img = None
            try:
                img_path = r.Rlocation('coralnpu_hw/tests/posenet/models/test_input_353x481.bmp')
                if img_path:
                    img = Image.open(img_path).convert('RGB')
            except Exception:
                pass

            if img is None:
                try:
                    img_path = r.Rlocation('coralnpu_hw/tests/posenet/models/test_couple.jpg')
                    if img_path:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((481, 353))
                except Exception:
                    pass

            if img is None:
                img = Image.new('RGB', (481, 353), (40, 40, 40))

            draw = ImageDraw.Draw(img)
            vis_score_thresh = 0.3

            for p_idx, pose in enumerate(all_poses):
                color = POSE_COLORS[p_idx % len(POSE_COLORS)]
                dim_color = tuple(c // 2 for c in color)
                kps = pose['keypoints']

                # 画骨骼连线
                for (i, j) in SKELETON_EDGES:
                    if kps[i]['score'] > vis_score_thresh and kps[j]['score'] > vis_score_thresh:
                        x1, y1 = kps[i]['x'], kps[i]['y']
                        x2, y2 = kps[j]['x'], kps[j]['y']
                        draw.line([(x1, y1), (x2, y2)], fill=dim_color, width=3)

                # 画关键点圆点
                for k_idx, kp in enumerate(kps):
                    if kp['score'] > vis_score_thresh:
                        x, y = kp['x'], kp['y']
                        radius = 5
                        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                                     fill=color, outline=(255, 255, 255))

                # 标注 pose 编号和分数
                nose = kps[0]
                if nose['score'] > vis_score_thresh:
                    label = f"P{p_idx}: {pose['pose_score']:.2f}"
                    draw.text((nose['x'] + 10, nose['y'] - 15), label, fill=color)

            # 同时画参考结果 (用灰色虚线表示)
            for p_idx, ref_pose in enumerate(REFERENCE_POSES):
                ref_kps = ref_pose['keypoints']
                for (i, j) in SKELETON_EDGES:
                    if ref_kps[i]['score'] > vis_score_thresh and ref_kps[j]['score'] > vis_score_thresh:
                        x1, y1 = ref_kps[i]['x'], ref_kps[i]['y']
                        x2, y2 = ref_kps[j]['x'], ref_kps[j]['y']
                        # 虚线效果：画小段
                        seg_len = ((x2-x1)**2 + (y2-y1)**2)**0.5
                        steps = max(int(seg_len / 6), 1)
                        for s in range(0, steps, 2):
                            t1 = s / steps
                            t2 = min((s + 1) / steps, 1.0)
                            sx1 = x1 + (x2 - x1) * t1
                            sy1 = y1 + (y2 - y1) * t1
                            sx2 = x1 + (x2 - x1) * t2
                            sy2 = y1 + (y2 - y1) * t2
                            draw.line([(sx1, sy1), (sx2, sy2)], fill=(200, 200, 200), width=1)

                # 参考关键点用小方块
                for kp in ref_kps:
                    if kp['score'] > vis_score_thresh:
                        x, y = kp['x'], kp['y']
                        r_size = 3
                        draw.rectangle([(x - r_size, y - r_size), (x + r_size, y + r_size)],
                                       outline=(200, 200, 200), width=1)

            # 图例
            draw.text((10, 10), "Solid circles = Detected", fill=(255, 255, 255))
            draw.text((10, 25), "Gray squares  = Reference", fill=(200, 200, 200))
            for p_idx in range(min(num_poses, len(POSE_COLORS))):
                draw.text((10, 40 + p_idx * 15),
                          f"Pose {p_idx} (score: {all_poses[p_idx]['pose_score']:.3f})",
                          fill=POSE_COLORS[p_idx])

            output_vis_path = '/tmp/posenet_result.png'
            img.save(output_vis_path)
            print(f"\n {'='*60}")
            print(f"  VISUALIZATION")
            print(f" {'='*60}")
            print(f"  Pose visualization saved to: {output_vis_path}")
            print(f"  Open with: eog {output_vis_path}")
            print(f" {'='*60}")

        except ImportError:
            print("\n [VIS] Pillow not installed, skipping visualization.")
            print("       Install with: pip install Pillow")
        except Exception as e:
            print(f"\n [VIS] Visualization failed: {e}")

    print(f"\nFINAL STATUS: {status} ({msg})")
    print(f"Total inference cycles: {total_cycles:,}")

    if status != 0:
        debug_log_raw = await fixture.read('debug_log_buffer', 512)
        debug_log_str = bytes(debug_log_raw).split(b'\x00')[0].decode(errors='replace')
        print(f"\n{'='*60}")
        print(f" TFLM DebugLog Output:")
        print(f"{'='*60}")
        print(debug_log_str if debug_log_str else "(empty)")
        print(f"{'='*60}\n")
        assert False, f"Test Failed with status {status}: {msg}"