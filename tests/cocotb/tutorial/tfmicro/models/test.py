import tensorflow as tf
import numpy as np

# 替换为你的 .tflite 文件路径
model_path = "mobilenet_v1_025_128_quant.tflite"

try:
    # 加载模型
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 获取输入和输出的详细信息
    input_details = interpreter.get_input_details()
    tensor_details = interpreter.get_tensor_details()

    print("=" * 60)
    print(f"MODEL: {model_path}")
    print("=" * 60)

    # 1. 检查模型输入 (Model Input)
    print("\n[Model Input Info]")
    for i, detail in enumerate(input_details):
        dtype = detail['dtype']
        shape = detail['shape']
        quant = detail['quantization']  # (scale, zero_point)
        
        type_str = "Unknown"
        if dtype == np.uint8: type_str = "UINT8 (0 ~ 255)"
        elif dtype == np.int8: type_str = "INT8 (-128 ~ 127)"
        elif dtype == np.float32: type_str = "FLOAT32"
        
        print(f"  Input {i}: {detail['name']}")
        print(f"    - Type: {dtype} -> 【 {type_str} 】")
        print(f"    - Shape: {shape}")
        print(f"    - Quantization: Scale={quant[0]}, ZeroPoint={quant[1]}")

    # 2. 检查第一层运算 (Op 0) 的输入
    # 我们遍历所有 Tensor，找到被 Op 0 使用的 Tensor
    # 注意：Interpreter API 不直接按 Op 排序暴露，但我们可以打印前几个 Tensor 的类型
    
    print("\n[First 5 Tensors Check]")
    for i in range(min(5, len(tensor_details))):
        t = tensor_details[i]
        dtype = t['dtype']
        name = t['name']
        
        type_str = "Unknown"
        if dtype == np.uint8: type_str = "UINT8"
        elif dtype == np.int8: type_str = "INT8"
        
        print(f"  Tensor {i} ({name}): {dtype} -> {type_str}")

except Exception as e:
    print(f"Error loading model: {e}")
    print("提示: 确保你安装了 tensorflow (pip install tensorflow)")

