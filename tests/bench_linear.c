/* Linear micro-benchmark: correctness vs the generic kernel + timing for
 * the Qwen DiT shapes at 4096-token rows. Run in three modes:
 *   ./tests/bench_linear                        -> MPS fast path (default)
 *   H3_DISABLE_MPS_LINEAR=1 ./tests/bench_linear -> generic scalar kernel
 *   H3_NAX=1 ./tests/bench_linear               -> nax TensorOps kernel
 */
#include "h3_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

typedef struct {
    uint32_t in_dim, out_dim;
    const char *name;
} shape;

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu: %s\n", error);
        return 1;
    }
    if (!h3_gpu_begin(gpu)) {
        printf("begin: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    uint32_t rows = 4119;  /* 4096 image tokens + 23 text */
    shape shapes[] = {
        {4096, 24576, "fc1"},
        {12288, 4096, "down"},
        {4096, 4096, "qkv/o"},
        {4096, 1024, "kv"},
    };
    /* Deterministic pseudo-random bf16 inputs. */
    unsigned long long rng = 88172645463325252ull;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        uint32_t in_dim = shapes[s].in_dim, out_dim = shapes[s].out_dim;
        size_t in_count = (size_t)rows * in_dim;
        size_t w_count = (size_t)out_dim * in_dim;
        size_t out_count = (size_t)rows * out_dim;
        h3_gpu_tensor *input = h3_gpu_tensor_new_bf16(gpu, in_count);
        h3_gpu_tensor *weight = h3_gpu_tensor_new_bf16(gpu, w_count);
        h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, out_count);
        h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, out_count);
        uint16_t *host_in = malloc(sizeof(uint16_t) * in_count);
        uint16_t *host_w = malloc(sizeof(uint16_t) * w_count);
        for (size_t i = 0; i < in_count; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            host_in[i] = (uint16_t)(0x3f00u | (rng & 0x7fu));  /* ~[1,2) */
        }
        for (size_t i = 0; i < w_count; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            host_w[i] = (uint16_t)(0x3b80u | (rng & 0x7fu));  /* ~small */
        }
        h3_gpu_tensor_write_bf16(input, host_in, in_count);
        h3_gpu_tensor_write_bf16(weight, host_w, w_count);

        /* Reference: the generic scalar kernel (dtype/layout ground truth). */
        if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) {
            printf("sync: %s\n", h3_gpu_error(gpu));
            return 1;
        }
        if (!h3_gpu_linear_bf16(gpu, reference, input, weight, NULL, rows,
                                in_dim, out_dim)) {
            printf("%s: reference failed: %s\n", shapes[s].name,
                   h3_gpu_error(gpu));
            return 1;
        }
        if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) return 1;

        /* Candidate (MPS / nax / scalar depending on env). */
        if (!h3_gpu_linear_bf16(gpu, output, input, weight, NULL, rows,
                                in_dim, out_dim)) {
            printf("%s: candidate failed: %s\n", shapes[s].name,
                   h3_gpu_error(gpu));
            return 1;
        }
        if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) return 1;
        uint16_t *host_ref = malloc(sizeof(uint16_t) * out_count);
        uint16_t *host_out = malloc(sizeof(uint16_t) * out_count);
        h3_gpu_tensor_read_bf16(reference, host_ref, out_count);
        h3_gpu_tensor_read_bf16(output, host_out, out_count);
        double sum_diff = 0.0, sum_ref = 0.0;
        for (size_t i = 0; i < out_count; i++) {
            float a, b;
            uint32_t ba = (uint32_t)host_ref[i] << 16;
            uint32_t bb = (uint32_t)host_out[i] << 16;
            memcpy(&a, &ba, 4); memcpy(&b, &bb, 4);
            double diff = (double)a - (double)b;
            sum_diff += diff * diff;
            sum_ref += (double)a * a;
        }
        double rel = sqrt(sum_diff) / (sqrt(sum_ref) + 1e-30);

        /* Timing: 10 iterations. */
        int iterations = 10;
        double start = now_seconds();
        for (int i = 0; i < iterations; i++) {
            if (!h3_gpu_linear_bf16(gpu, output, input, weight, NULL, rows,
                                    in_dim, out_dim)) {
                printf("%s: timed call failed\n", shapes[s].name);
                return 1;
            }
        }
        if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) return 1;
        double per_call = (now_seconds() - start) / iterations;
        double flops = 2.0 * rows * in_dim * out_dim;
        printf("%-8s relL2 %.3g  %6.1f ms  %5.1f TFLOPS\n", shapes[s].name,
               rel, per_call * 1e3, flops / per_call / 1e12);
        free(host_in); free(host_w); free(host_ref); free(host_out);
        h3_gpu_tensor_free(input); h3_gpu_tensor_free(weight);
        h3_gpu_tensor_free(output); h3_gpu_tensor_free(reference);
    }
    return 0;
}
