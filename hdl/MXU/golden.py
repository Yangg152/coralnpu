#!/usr/bin/env python3
import numpy as np
import os

os.makedirs("cases", exist_ok=True)

def pack_weight_row(w_row):
    """W[k][:] 长度16"""
    val = 0
    for j in range(16):
        b = int(w_row[j]) & 0xFF
        val |= (b << (j * 8))
    return f"{val:032x}"

def pack_act_lines(A):
    """A尺寸为 [16, Tk]，将其打包为 Tk/16 个16×16的块"""
    lines = []
    Tk = A.shape[1]
    num_chunks = Tk // 16
    for c in range(num_chunks):
        for m in range(16):
            val = 0
            for k in range(16):
                b = int(A[m][c*16 + k]) & 0xFF
                val |= (b << (k * 8))
            lines.append(f"{val:032x}")
    return lines

def pack_result_bus(acc):
    """ACC为 [16, 16] 的 INT32。分4拍输出，每拍16条总线。"""
    lines = []
    for f in range(4):       
        for r in range(4):   
            m = f * 4 + r
            for c in range(4): 
                val = 0
                for nc in range(4):
                    a = int(acc[m][c*4+nc]) & 0xFFFFFFFF
                    val |= (a << (nc * 32))
                lines.append(f"{val:032x}")
    return lines

def compute_golden(W, A):
    """
    [修正]: 在执行 dot 前强制转换为 int32，避免 numpy 默认使用 int8 累加导致溢出！
    """
    return np.dot(A.astype(np.int32), W.astype(np.int32)).astype(np.int64)

def gen_tc(tc_name, Tk, seed):
    print(f"生成 {tc_name} (Tk={Tk})...")
    np.random.seed(seed)
    
    W = np.random.randint(-128, 128, (Tk, 16), dtype=np.int8)
    A = np.random.randint(-128, 128, (16, Tk), dtype=np.int8)
    
    if "tc3" in tc_name:
        W = np.full((Tk, 16), -128, dtype=np.int8)
        A = np.full((16, Tk), 127, dtype=np.int8)
    elif "tc4" in tc_name:
        A = np.zeros((16, Tk), dtype=np.int8)

    acc = compute_golden(W, A)

    write_hex(f"cases/{tc_name}_weight.hex", [pack_weight_row(W[k]) for k in range(Tk)])
    write_hex(f"cases/{tc_name}_act.hex", pack_act_lines(A))
    write_hex(f"cases/{tc_name}_golden.hex", pack_result_bus(acc))
    print(f"  {tc_name} ACC[0][0:4] = {acc[0,0:4]}")
    return acc

def write_hex(fname, lines):
    with open(fname, 'w') as f:
        for l in lines:
            f.write(l + '\n')

if __name__ == "__main__":
    gen_tc("tc1", 16, 42)
    gen_tc("tc2", 64, 123)
    gen_tc("tc3", 16, 1)
    gen_tc("tc4", 16, 77)

    print("生成 tc5...")
    Tk = 64
    acc5 = np.zeros((16, 16), dtype=np.int64)
    for tile in range(4):
        W = np.random.randint(-128, 128, (Tk, 16), dtype=np.int8)
        A = np.random.randint(-128, 128, (16, Tk), dtype=np.int8)
        acc5 += compute_golden(W, A)
        write_hex(f"cases/tc5_w{tile}.hex", [pack_weight_row(W[k]) for k in range(Tk)])
        write_hex(f"cases/tc5_a{tile}.hex", pack_act_lines(A))
    write_hex("cases/tc5_golden.hex", pack_result_bus(acc5))
    print(f"  tc5 ACC[0][0:4] = {acc5[0,0:4]}")
    print("\n测试向量生成完成！")
