#!/usr/bin/env python3
"""Generate MXU test vectors and golden results."""

import numpy as np
import os

CASES_DIR = os.path.join('.', 'cases')
os.makedirs(CASES_DIR, exist_ok=True)

MASK_32 = 0xFFFFFFFF
MASK_128 = (1 << 128) - 1

def to_hex128(val):
    """Force exactly 32 hex chars for 128-bit value."""
    return f'{val & MASK_128:032x}'

def gen_case(name, Tk, act, weight, debug=False):
    golden = act.astype(np.int32) @ weight.astype(np.int32)
    num_a_chunks = (Tk + 15) // 16
    cfg_Tk_val = 0 if Tk == 256 else Tk

    # Config
    with open(os.path.join(CASES_DIR, f'{name}_cfg.hex'), 'w') as f:
        f.write(f'{cfg_Tk_val:02x}\n')

    # Weight: Tk rows, each 128 bits = {w[k][15], ..., w[k][0]}
    with open(os.path.join(CASES_DIR, f'{name}_weight.hex'), 'w') as f:
        for k in range(Tk):
            val = 0
            for b in range(16):
                val |= (int(weight[k, b]) & 0xFF) << (b * 8)
            f.write(to_hex128(val) + '\n')

    # Activation: 16 rows x num_a_chunks beats
    # beat order: row0-chunk0, row0-chunk1, ..., row1-chunk0, ...
    # each beat: {act[row, chunk*16+15], ..., act[row, chunk*16+0]}
    with open(os.path.join(CASES_DIR, f'{name}_act.hex'), 'w') as f:
        for row in range(16):
            for chunk in range(num_a_chunks):
                val = 0
                for b in range(16):
                    idx = chunk * 16 + b
                    if idx < Tk:
                        val |= (int(act[row, idx]) & 0xFF) << (b * 8)
                f.write(to_hex128(val) + '\n')

    # Golden: 64 beats (16 rows x 4 groups), each 128 bits = 4 x int32
    with open(os.path.join(CASES_DIR, f'{name}_golden.hex'), 'w') as f:
        for row in range(16):
            for grp in range(4):
                val = 0
                for c in range(4):
                    col = grp * 4 + c
                    word = int(golden[row, col]) & MASK_32
                    val |= word << (c * 32)
                f.write(to_hex128(val) + '\n')

    if debug:
        print(f'\n[DEBUG] {name}: Tk={Tk}')
        # Expected abuf content after transpose store:
        # abuf[k] = {{act[15][k], act[14][k], ..., act[0][k]}}
        # wbuf[k] = {{w[k][15], w[k][14], ..., w[k][0]}}
        for k in range(min(3, Tk)):
            w_val = 0
            for b in range(16):
                w_val |= (int(weight[k, b]) & 0xFF) << (b * 8)
            a_val = 0
            for m in range(16):
                a_val |= (int(act[m, k]) & 0xFF) << (m * 8)
            print(f'  k={k}: wbuf={to_hex128(w_val)} abuf={to_hex128(a_val)}')
        print(f'  golden[0][0:4] = {golden[0, 0:4]}')
        # Show expected beat0: row0, grp0 = {golden[0][3], golden[0][2], golden[0][1], golden[0][0]}
        val = 0
        for c in range(4):
            val |= (int(golden[0, c]) & MASK_32) << (c * 32)
        print(f'  golden beat0 = {to_hex128(val)}')

    print(f'[golden] {name}: Tk={Tk}, golden shape={golden.shape}')
    return golden


np.random.seed(42)

Tk = 16
act1 = np.random.randint(-64, 64, size=(16, Tk), dtype=np.int8)
wt1  = np.random.randint(-64, 64, size=(Tk, 16), dtype=np.int8)
gen_case('tc1_tk16', Tk, act1, wt1, debug=False)

Tk = 64
act2 = np.random.randint(-128, 127, size=(16, Tk), dtype=np.int8)
wt2  = np.random.randint(-128, 127, size=(Tk, 16), dtype=np.int8)
gen_case('tc2_tk64', Tk, act2, wt2)

Tk = 256
act3 = np.random.randint(-50, 50, size=(16, Tk), dtype=np.int8)
wt3  = np.random.randint(-50, 50, size=(Tk, 16), dtype=np.int8)
gen_case('tc3_tk256', Tk, act3, wt3)

Tk = 16
act4 = np.random.randint(0, 127, size=(16, Tk), dtype=np.int8)
wt4  = np.random.randint(0, 127, size=(Tk, 16), dtype=np.int8)
gen_case('tc4_unsigned', Tk, act4, wt4)

print('\n[golden] All test cases generated.')