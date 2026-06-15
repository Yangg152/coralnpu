#include <riscv_vector.h>
#include <stdint.h>

#define N 16

int8_t  A[N * N] __attribute__((aligned(16)));
int8_t  B[N * N] __attribute__((aligned(16)));
int16_t C[N * N] __attribute__((aligned(16)));   // int8*int8 累加16次，用 int16 防溢出
int16_t ref[N * N] __attribute__((aligned(16)));

int main() {
    // 初始化
    for (int i = 0; i < N * N; i++) {
        A[i] = (int8_t)((i % 5) + 1);   // 1~5 循环，避免溢出
        B[i] = (int8_t)((i % 3) + 1);   // 1~3 循环
    }

    // 标量参考结果
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int16_t sum = 0;
            for (int k = 0; k < N; k++)
                sum += (int16_t)A[i * N + k] * (int16_t)B[k * N + j];
            ref[i * N + j] = sum;
        }
    }

    // RVV 矩阵乘法
    // 思路：对 C 的每一行 i，逐列向量化
    //   C[i][0..15] = sum_k( A[i][k] * B[k][0..15] )
    //   把 B 的第 k 行整行 load 成向量，乘以标量 A[i][k]，累加到结果向量
    for (int i = 0; i < N; i++) {
        size_t vl = __riscv_vsetvl_e16m1(N);
        vint16m1_t vc = __riscv_vmv_v_x_i16m1(0, vl);   // 累加器清零

        for (int k = 0; k < N; k++) {
            // load B 的第 k 行 (16 x int8) -> 宽化到 int16
            vint8mf2_t vb8 = __riscv_vle8_v_i8mf2(&B[k * N], vl);
            vint16m1_t vb  = __riscv_vsext_vf2_i16m1(vb8, vl);

            // 标量 A[i][k] 广播乘加
            int16_t a_ik = (int16_t)A[i * N + k];
            vc = __riscv_vmacc_vx_i16m1(vc, a_ik, vb, vl);
        }

        // store 结果行
        __riscv_vse16_v_i16m1(&C[i * N], vc, vl);
    }

    __asm__ volatile ("fence rw, rw");

    // 验证
    int errors = 0;
    for (int i = 0; i < N * N; i++) {
        if (C[i] != ref[i])
            errors++;
    }
    return errors;
}
