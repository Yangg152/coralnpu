import numpy as np
from tensorflow.lite.python.schema_py_generated import ModelT
import flatbuffers


def convert_uint8_to_int8(input_path, output_path):
    with open(input_path, 'rb') as f:
        buf = bytearray(f.read())

    model = ModelT.InitFromPackedBuf(buf, 0)
    sg = model.subgraphs[0]

    # Identify filter tensors (CONV_2D and DEPTHWISE_CONV_2D input[1])
    filter_indices = set()
    for op in sg.operators:
        oc = model.operatorCodes[op.opcodeIndex]
        code = oc.builtinCode if oc.builtinCode != 0 else oc.deprecatedBuiltinCode
        if code in (3, 4):  # CONV_2D, DEPTHWISE_CONV_2D
            filter_indices.add(op.inputs[1])

    for i, tensor in enumerate(sg.tensors):
        q = tensor.quantization
        if q is None or q.scale is None or len(q.scale) == 0:
            continue

        if tensor.type == 3:  # UINT8
            tensor.type = 9  # INT8

            buf_data = model.buffers[tensor.buffer].data
            if buf_data is not None and len(buf_data) > 0:
                u8_data = np.frombuffer(bytes(buf_data), dtype=np.uint8)

                if i in filter_indices:
                    # Filter: w_i8 = w_u8 - w_zp (so that real_value = w_i8 * scale)
                    zp = int(q.zeroPoint[0]) if q.zeroPoint is not None and len(q.zeroPoint) > 0 else 128
                    i8_data = (u8_data.astype(np.int16) - zp).clip(-128, 127).astype(np.int8)
                else:
                    # Activation: val_i8 = val_u8 - 128
                    i8_data = (u8_data.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)

                model.buffers[tensor.buffer].data = i8_data.tobytes()

            # Adjust zero_point
            if q.zeroPoint is not None and len(q.zeroPoint) > 0:
                if i in filter_indices:
                    # Filter zp must be 0 for int8
                    q.zeroPoint = np.array([0] * len(q.zeroPoint), dtype=np.int64)
                else:
                    # Activation: zp_i8 = zp_u8 - 128
                    new_zps = [int(zp) - 128 for zp in q.zeroPoint]
                    q.zeroPoint = np.array(new_zps, dtype=np.int64)

    # Bias (INT32) stays unchanged:
    # (in_u8 - in_zp_u8) * (w_u8 - w_zp_u8) = (in_i8 - in_zp_i8) * w_i8
    # Both sides are identical, so bias needs no adjustment.

    # Serialize
    builder = flatbuffers.Builder(1024 * 1024 * 10)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, b"TFL3")
    output_buf = builder.Output()
    with open(output_path, 'wb') as f:
        f.write(bytes(output_buf))
    print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    convert_uint8_to_int8("posenet_075_353_481_quant.tflite", "posenet_075_353_481_int8.tflite")
