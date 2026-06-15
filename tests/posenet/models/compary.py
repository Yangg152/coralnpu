import numpy as np
import os
import tensorflow as tf


def compare_models():
    interp_u8 = tf.lite.Interpreter(model_path="posenet_075_353_481_quant.tflite")
    interp_u8.allocate_tensors()

    # 禁用 XNNPACK delegate
    interp_i8 = tf.lite.Interpreter(
        model_path="posenet_075_353_481_int8.tflite",
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    interp_i8.allocate_tensors()

    in_u8_details = interp_u8.get_input_details()[0]
    in_i8_details = interp_i8.get_input_details()[0]

    np.random.seed(42)
    input_u8 = np.random.randint(0, 256, size=in_u8_details['shape'], dtype=np.uint8)
    input_i8 = (input_u8.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)

    interp_u8.set_tensor(in_u8_details['index'], input_u8)
    interp_u8.invoke()
    interp_i8.set_tensor(in_i8_details['index'], input_i8)
    interp_i8.invoke()

    outs_u8 = interp_u8.get_output_details()
    outs_i8 = interp_i8.get_output_details()

    for i, (ou8, oi8) in enumerate(zip(outs_u8, outs_i8)):
        out_u8 = interp_u8.get_tensor(ou8['index']).flatten().astype(np.float64)
        out_i8 = interp_i8.get_tensor(oi8['index']).flatten().astype(np.float64)

        u8_zp = float(ou8['quantization_parameters']['zero_points'][0])
        u8_scale = float(ou8['quantization_parameters']['scales'][0])
        i8_zp = float(oi8['quantization_parameters']['zero_points'][0])
        i8_scale = float(oi8['quantization_parameters']['scales'][0])

        float_u8 = (out_u8 - u8_zp) * u8_scale
        float_i8 = (out_i8 - i8_zp) * i8_scale

        diff = float_i8 - float_u8
        abs_diff = np.abs(diff)

        print(f"\nOutput[{i}]: shape={ou8['shape']}, name={ou8['name']}")
        print(f"  u8: zp={u8_zp}, scale={u8_scale:.6f}")
        print(f"  i8: zp={i8_zp}, scale={i8_scale:.6f}")
        print(f"  float_u8 range: [{float_u8.min():.4f}, {float_u8.max():.4f}]")
        print(f"  float_i8 range: [{float_i8.min():.4f}, {float_i8.max():.4f}]")
        print(f"  abs diff: mean={abs_diff.mean():.6f}, max={abs_diff.max():.6f}, "
              f"median={np.median(abs_diff):.6f}")
        print(f"  relative to u8 range: {abs_diff.max() / (float_u8.max() - float_u8.min() + 1e-10) * 100:.2f}%")

        raw_u8_sample = interp_u8.get_tensor(ou8['index']).flatten()[:10]
        raw_i8_sample = interp_i8.get_tensor(oi8['index']).flatten()[:10]
        print(f"  raw u8[:10]: {raw_u8_sample}")
        print(f"  raw i8[:10]: {raw_i8_sample}")
        expected_i8 = (raw_u8_sample.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
        print(f"  expected i8 (u8-128)[:10]: {expected_i8}")
        print(f"  match u8-128: {np.array_equal(raw_i8_sample, expected_i8)}")


if __name__ == "__main__":
    compare_models()
