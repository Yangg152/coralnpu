#include <riscv_vector.h>
#include <string.h>
#include <stdint.h>

// ============================================================
// MXU 指令编码
// 格式: {funct6[31:26], vm[25], vs2[24:20], vs1[19:15], funct3[14:12]=001, vd[11:7], opcode[6:0]=1010111}
//
// funct6 定义 (从 TB 的 MXU_MCFG 等常量推断):
//   MCFG     = 6'b000000 = 0x00
//   MLOAD_W  = 6'b000001 = 0x01
//   MLOAD_A  = 6'b000010 = 0x02
//   MZERO    = 6'b000011 = 0x03
//   MMA      = 6'b000100 = 0x04
//   MFENCE   = 6'b000101 = 0x05
//   MSTORE   = 6'b000110 = 0x06  (注意: TB 用 OPIVI funct3)
//
// OPMXU funct3 = 3'b001 = 1
// OPIVI funct3 = 3'b011 = 3  (MSTORE 用这个)
// opcode = 7'b1010111 = 0x57
// ============================================================

// 构建 MXU 指令 (OPMXU funct3=001)
// vm: 0=is_last, 1=not_last
#define MXU_INST(funct6, vm, vs2, vs1, vd) \
    (((funct6) << 26) | ((vm) << 25) | ((vs2) << 20) | ((vs1) << 15) | (1 << 12) | ((vd) << 7) | 0x57)

// MSTORE 用 OPIVI funct3=011
#define MXU_MSTORE_INST(vm, vd) \
    (((0x06) << 26) | ((vm) << 25) | (0 << 20) | (0 << 15) | (3 << 12) | ((vd) << 7) | 0x57)

// ============================================================
// 数据区
// ============================================================
int8_t  weight_data[16 * 16] __attribute__((aligned(16)));  // Tk=16, 16 列
int8_t  act_data[16 * 16]    __attribute__((aligned(16)));   // 16 行, Tk=16
int32_t result_buf[16 * 16]  __attribute__((aligned(16)));   // 16x16 int32 结果
int32_t ref_buf[16 * 16];                                    // 参考结果

// ============================================================
// MXU 原语
// ============================================================

// MCFG: 配置 Tk 和 signed 模式
// rs1_val = (Tk & 0xFF) | (is_signed ? 0x100 : 0)
// 指令编码中 rs1 字段 = a0 = x10
static inline void mxu_mcfg(uint32_t cfg_val) {
    // MCFG: funct6=0, vm=1, vs2=0, vs1=0, vd=0
    // 但 rs1 字段需要编码为 x10(a0) 来匹配 rs1_val 传递
    // 实际上 TB 里 rs1_val 是通过 cmd.rs1 传的，不是指令编码
    // 在真实硬件上，mcfg 从 rs1 寄存器读值
    register uint32_t a0 asm("a0") = cfg_val;
    // funct6=0, vm=1, vs2=0, rs1=a0(x10), funct3=001, vd=0
    // = 000000 1 00000 01010 001 00000 1010111
    // = 0x02051057
    asm volatile(".word 0x02051057" :: "r"(a0) : "memory");
}

// MZERO: 清零累加器
static inline void mxu_mzero(void) {
    // funct6=3, vm=1, vs2=0, vs1=0, vd=0
    // = 000011 1 00000 00000 001 00000 1010111
    // = 0x0E001057
    asm volatile(".word 0x0E001057" ::: "memory");
}

// MLOAD_W: 从 vs2 加载权重到 MXU weight buffer
// vm=1 表示 not_last, vm=0 表示 is_last
static inline void mxu_mload_w(int vs2, int is_last) {
    // funct6=1, vm=!is_last, vs2=vs2, vs1=0, vd=0
    // 需要动态构建... 用 v16 作为 vs2
    // 简化: 总是用 v16 作为中转寄存器
    // funct6=1=000001, vm, vs2=10000(v16), vs1=0, funct3=001, vd=0
    // vm=1: 000001 1 10000 00000 001 00000 1010111 = 0x07001057
    // vm=0: 000001 0 10000 00000 001 00000 1010111 = 0x05001057
    if (is_last)
        asm volatile(".word 0x05001057" ::: "memory");
    else
        asm volatile(".word 0x07001057" ::: "memory");
}

// MLOAD_A: 从 vs2 加载激活到 MXU activation buffer
static inline void mxu_mload_a(int is_last) {
    // funct6=2=000010, vm, vs2=10000(v16), vs1=0, funct3=001, vd=0
    // vm=1: 000010 1 10000 00000 001 00000 1010111 = 0x0B001057
    // vm=0: 000010 0 10000 00000 001 00000 1010111 = 0x09001057
    if (is_last)
        asm volatile(".word 0x09001057" ::: "memory");
    else
        asm volatile(".word 0x0B001057" ::: "memory");
}

// MMA: 执行矩阵乘累加
static inline void mxu_mma(void) {
    // funct6=4=000100, vm=1, vs2=0, vs1=0, vd=0
    // = 000100 1 00000 00000 001 00000 1010111
    // = 0x12001057
    asm volatile(".word 0x12001057" ::: "memory");
}

// MFENCE: 等待 MXU 完成
static inline void mxu_mfence(void) {
    // funct6=5=000101, vm=1, vs2=0, vs1=0, vd=0
    // = 000101 1 00000 00000 001 00000 1010111
    // = 0x16001057
    asm volatile(".word 0x16001057" ::: "memory");
}

// MSTORE: 从 MXU 累加器读出结果到 vd
static inline void mxu_mstore(int vd) {
    // funct6=6, vm=1, vs2=0, vs1=0, funct3=011(OPIVI), vd=vd
    // 简化: 总是写到 v16
    // = 000110 1 00000 00000 011 10000 1010111
    // = 0x1A001857  (vd=16=10000)
    asm volatile(".word 0x1A001857" ::: "memory");
}

// ============================================================
// 辅助: 把 16 字节从内存加载到 v16
// ============================================================
static inline void load_v16(const int8_t* ptr) {
    // vle8.v v16, (ptr)  with vl=16
    // 先设 vl=16: vsetvli t0, x0, e8, m1
    asm volatile(
        "vsetivli zero, 16, e8, m1, ta, ma\n\t"
        "vle8.v v16, (%0)"
        :: "r"(ptr) : "memory", "v16"
    );
}

// 辅助: 把 v16 的 128 位 (作为 4 个 int32) 存到内存
static inline void store_v16_i32(int32_t* ptr) {
    asm volatile(
        "vsetivli zero, 4, e32, m1, ta, ma\n\t"
        "vse32.v v16, (%0)"
        :: "r"(ptr) : "memory"
    );
}

// ============================================================
// 测试 1: MZERO -> MSTORE, 验证全零
// ============================================================
int test_mzero(void) {
    int errors = 0;

    // 填 sentinel
    memset(result_buf, 0xAB, sizeof(result_buf));

    mxu_mcfg(16 | 0x100);  // Tk=16, signed
    mxu_mzero();

    // MSTORE 64 次, 每次读 4 个 int32 到 v16, 再存到内存
    for (int i = 0; i < 64; i++) {
        mxu_mstore(16);
        store_v16_i32(&result_buf[i * 4]);
    }
    mxu_mfence();

    // 检查全零
    for (int i = 0; i < 256; i++) {
        if (result_buf[i] != 0) {
            errors++;
        }
    }
    return errors;
}

// ============================================================
// 测试 2: 全 1 权重 × 全 1 激活, Tk=16
// 期望: 每个元素 = 1*1*16 = 16
// ============================================================
int test_ones(void) {
    int errors = 0;

    memset(result_buf, 0xAB, sizeof(result_buf));

    // 准备数据: 全 1
    int8_t ones[16] __attribute__((aligned(16)));
    for (int i = 0; i < 16; i++) ones[i] = 1;

    mxu_mcfg(16 | 0x100);

    // MLOAD_W: 16 次 (Tk=16), 每次加载 v16
    for (int k = 0; k < 16; k++) {
        load_v16(ones);
        mxu_mload_w(16, (k == 15) ? 1 : 0);
    }
    mxu_mfence();

    mxu_mzero();

    // MLOAD_A: 16 行 × 1 chunk = 16 次
    for (int beat = 0; beat < 16; beat++) {
        load_v16(ones);
        mxu_mload_a((beat == 15) ? 1 : 0);
    }

    mxu_mma();
    mxu_mfence();

    // MSTORE
    for (int i = 0; i < 64; i++) {
        mxu_mstore(16);
        store_v16_i32(&result_buf[i * 4]);
    }
    mxu_mfence();

    // 检查: 每个元素应该是 16
    for (int i = 0; i < 256; i++) {
        if (result_buf[i] != 16) {
            errors++;
        }
    }
    return errors;
}

// ============================================================
// 测试 3: 单位矩阵权重 × 递增激活
// W = I (identity), A[m][k] = m+1 for all k
// 期望: result[m][n] = (m+1) 当 n==对应列, 其余 0
// 实际: result[m][n] = sum_k(A[m][k] * W[k][n]) = A[m][n] = m+1
// ============================================================
int test_identity(void) {
    int errors = 0;

    memset(result_buf, 0xAB, sizeof(result_buf));

    int8_t w_row[16] __attribute__((aligned(16)));
    int8_t a_row[16] __attribute__((aligned(16)));

    mxu_mcfg(16 | 0x100);

    // MLOAD_W: 单位矩阵, w[k][n] = (k==n) ? 1 : 0
    for (int k = 0; k < 16; k++) {
        memset(w_row, 0, 16);
        w_row[k] = 1;
        load_v16(w_row);
        mxu_mload_w(16, (k == 15) ? 1 : 0);
    }
    mxu_mfence();

    mxu_mzero();

    // MLOAD_A: a[m][k] = m+1 for all k
    for (int m = 0; m < 16; m++) {
        int8_t val = (int8_t)(m + 1);
        memset(a_row, val, 16);
        load_v16(a_row);
        mxu_mload_a((m == 15) ? 1 : 0);
    }

    mxu_mma();
    mxu_mfence();

    for (int i = 0; i < 64; i++) {
        mxu_mstore(16);
        store_v16_i32(&result_buf[i * 4]);
    }
    mxu_mfence();

    // 检查: result[m][n] = m+1 for all n
    for (int m = 0; m < 16; m++) {
        for (int n = 0; n < 16; n++) {
            int32_t expected = m + 1;
            if (result_buf[m * 16 + n] != expected) {
                errors++;
            }
        }
    }
    return errors;
}

// ============================================================
// 测试 4: 负数权重
// W[k][n] = -1, A[m][k] = 2
// 期望: result[m][n] = (-1)*2*16 = -32
// ============================================================
int test_negative(void) {
    int errors = 0;

    memset(result_buf, 0xAB, sizeof(result_buf));

    int8_t w_neg[16] __attribute__((aligned(16)));
    int8_t a_pos[16] __attribute__((aligned(16)));
    memset(w_neg, -1, 16);   // 0xFF = -1 signed
    memset(a_pos, 2, 16);

    mxu_mcfg(16 | 0x100);

    for (int k = 0; k < 16; k++) {
        load_v16(w_neg);
        mxu_mload_w(16, (k == 15) ? 1 : 0);
    }
    mxu_mfence();

    mxu_mzero();

    for (int beat = 0; beat < 16; beat++) {
        load_v16(a_pos);
        mxu_mload_a((beat == 15) ? 1 : 0);
    }

    mxu_mma();
    mxu_mfence();

    for (int i = 0; i < 64; i++) {
        mxu_mstore(16);
        store_v16_i32(&result_buf[i * 4]);
    }
    mxu_mfence();

    for (int i = 0; i < 256; i++) {
        if (result_buf[i] != -32) {
            errors++;
        }
    }
    return errors;
}

// ============================================================
// 主函数
// ============================================================
volatile int test_results[4];

int main() {
    test_results[0] = test_mzero();
    test_results[1] = test_ones();
    test_results[2] = test_identity();
    test_results[3] = test_negative();

    // 总错误数
    int total = test_results[0] + test_results[1] + test_results[2] + test_results[3];

    return total;  // 0 = all pass
}