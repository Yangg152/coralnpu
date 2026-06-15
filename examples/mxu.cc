#include <riscv_vector.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// =============================================================================
// MXU instruction primitives (same as your mxu.h)
// =============================================================================

static inline __attribute__((always_inline))
void mxu_mcfg(uint32_t config_val) {
    register uint32_t a0_val asm("a0") = config_val;
    asm volatile(
        "vsetvli zero, %0, e8, m1, ta, ma\n\t"
        ".word 0x02051057"
        :: "r"(a0_val) : "memory");
}

static inline __attribute__((always_inline))
void mxu_mload_a_from_mem(const int8_t* src, bool is_last) {
    if (is_last) {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x09001057"
            :: "r"(src), "r"(16) : "memory", "v16");
    } else {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x0B001057"
            :: "r"(src), "r"(16) : "memory", "v16");
    }
}

static inline __attribute__((always_inline))
void mxu_mload_w_from_mem(const int8_t* src, bool is_last) {
    if (is_last) {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x05001057"
            :: "r"(src), "r"(16) : "memory", "v16");
    } else {
        asm volatile(
            "vsetvli zero, %1, e8, m1, ta, ma\n\t"
            "vle8.v v16, (%0)\n\t"
            ".word 0x07001057"
            :: "r"(src), "r"(16) : "memory", "v16");
    }
}

static inline __attribute__((always_inline))
void mxu_mzero() { asm volatile(".word 0x0E001057" ::: "memory"); }

static inline __attribute__((always_inline))
void mxu_mma() { asm volatile(".word 0x12001057" ::: "memory"); }

static inline __attribute__((always_inline))
void mxu_mstore_to_mem(int32_t* dst) {
    asm volatile(
        ".word 0x16001857\n\t"
        "vsetvli zero, %1, e32, m1, ta, ma\n\t"
        "vse32.v v16, (%0)"
        :: "r"(dst), "r"(4) : "memory", "v16");
}

// =============================================================================
// Test data — all 16-byte aligned
// =============================================================================

int8_t A[16][16] __attribute__((aligned(16)));
int8_t W[16][16] __attribute__((aligned(16)));
int32_t acc_buf[16 * 16] __attribute__((aligned(16)));
int32_t ref[16][16];

static const int REPEAT = 3;

int main() {
    int total_errors = 0;

    for (int iter = 0; iter < REPEAT; iter++) {
        // =========================================================
        // 1. Fill test data — use iter to vary the pattern each round
        //    A[i][j] = (i + 1 + iter) clamped to int8 range
        //    W[k][n] = (k + 1 + iter) clamped to int8 range
        // =========================================================
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                A[i][j] = (int8_t)((i + 1 + iter) & 0x7F);

        for (int k = 0; k < 16; k++)
            for (int n = 0; n < 16; n++)
                W[k][n] = (int8_t)((k + 1 + iter) & 0x7F);

        // CPU reference
        for (int i = 0; i < 16; i++)
            for (int n = 0; n < 16; n++) {
                int32_t sum = 0;
                for (int k = 0; k < 16; k++)
                    sum += (int32_t)A[i][k] * (int32_t)W[k][n];
                ref[i][n] = sum;
            }

        // =========================================================
        // 2. Configure MXU: Tk = 16
        // =========================================================
        mxu_mcfg((uint32_t)16 | 0x100);

        // =========================================================
        // 3. Load activation matrix A
        // =========================================================
        for (int row = 0; row < 16; row++) {
            bool last = (row == 15);
            mxu_mload_a_from_mem(&A[row][0], last);
        }

        // =========================================================
        // 4. Load weight matrix W
        // =========================================================
        for (int k = 0; k < 16; k++) {
            bool last = (k == 15);
            mxu_mload_w_from_mem(&W[k][0], last);
        }

        // =========================================================
        // 5. Zero accumulators, run matrix multiply
        // =========================================================
        mxu_mzero();
        mxu_mma();

        // =========================================================
        // 6. Read back results
        // =========================================================
        for (int i = 0; i < 64; i++)
            mxu_mstore_to_mem(&acc_buf[i * 4]);

        // =========================================================
        // 7. Verify against CPU reference
        // =========================================================
        for (int i = 0; i < 16; i++) {
            for (int n = 0; n < 16; n++) {
                int32_t mxu_val = acc_buf[i * 16 + n];
                int32_t ref_val = ref[i][n];
                if (mxu_val != ref_val) {
                    total_errors++;
                }
            }
        }
    }

    return total_errors;
}