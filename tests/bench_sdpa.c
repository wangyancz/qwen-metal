/* SDPA micro-benchmark: the hand-written flash2 kernel family vs the
 * MPSGraph fused SDPA operator (qw_gpu_sdpa_rect_bf16) at the Qwen DiT
 * production shapes (32 heads x 128 head dim, 1024 latent tokens).
 *
 *   ./tests/bench_sdpa
 *
 * Configs (1024² production row counts):
 *   t2i        q=8188 image rows vs kv=8211 (text rows run through the
 *              causal gqa kernel and never reach SDPA)
 *   ti2i full  q=kv=16418 joint rows (the QWEN_NO_PREFIX probe shape)
 *   ti2i pre   production prefix kernel with a representative
 *              block-causal kv_limit table (34 text rows causal + four
 *              4096-row blocks attending to their block end); throughput
 *              is reported against the effective FLOPs actually computed.
 * flash2 and the fused operator cross-check each other (relL2) on the
 * full-bidirectional configs.
 */
#include "h3_gpu.h"
#include "qwen_gpu_ext.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define HEADS 32
#define HEAD_DIM 128
#define SCALE (1.0f / sqrtf((float)HEAD_DIM))
#define WARMUP 2
#define ITERS 10

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint16_t bf16_bits(unsigned long long *rng) {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    return (uint16_t)(0x3f00u | (*rng & 0x7fu)); /* ~[1, 2) */
}

static h3_gpu_tensor *fill_bf16(h3_gpu *gpu, size_t count,
                                unsigned long long *rng) {
    uint16_t *host = malloc(count * sizeof(uint16_t));
    if (!host) return NULL;
    for (size_t i = 0; i < count; i++) host[i] = bf16_bits(rng);
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, count);
    if (tensor) h3_gpu_tensor_write_bf16(tensor, host, count);
    free(host);
    return tensor;
}

static double rel_l2(h3_gpu_tensor *a, h3_gpu_tensor *b, size_t count) {
    uint16_t *ha = malloc(count * sizeof(uint16_t));
    uint16_t *hb = malloc(count * sizeof(uint16_t));
    if (!ha || !hb) {
        free(ha);
        free(hb);
        return -1.0;
    }
    h3_gpu_tensor_read_bf16(a, ha, count);
    h3_gpu_tensor_read_bf16(b, hb, count);
    double sum_diff = 0.0, sum_ref = 0.0;
    for (size_t i = 0; i < count; i++) {
        float x, y;
        uint32_t bx = (uint32_t)ha[i] << 16;
        uint32_t by = (uint32_t)hb[i] << 16;
        memcpy(&x, &bx, 4);
        memcpy(&y, &by, 4);
        double diff = (double)x - (double)y;
        sum_diff += diff * diff;
        sum_ref += (double)x * x;
    }
    free(ha);
    free(hb);
    return sqrt(sum_diff) / (sqrt(sum_ref) + 1e-30);
}

typedef struct {
    const char *name;
    uint32_t q_rows;   /* real (useful) query rows */
    uint32_t kv_rows;
    int prefix;        /* benchmark the production prefix kernel */
} config;

static int sync_gpu(h3_gpu *gpu) {
    if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) {
        printf("sync: %s\n", h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

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

    config configs[] = {
        {"t2i img      q= 8188 kv= 8211 full", 8188, 8211, 0},
        {"ti2i full    q=16418 kv=16418 full", 16418, 16418, 0},
        {"ti2i prefix  q=16418 kv=16418 block-causal", 16418, 16418, 1},
        {"1024 img     q= 4096 kv= 4119 full", 4096, 4119, 0},
        {" 768 img     q= 2304 kv= 2351 full", 2304, 2351, 0},
        {" 512 img     q= 1024 kv= 1047 full", 1024, 1047, 0},
    };

    unsigned long long rng = 88172645463325252ull;
    for (size_t c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        uint32_t q_rows = configs[c].q_rows;
        uint32_t kv_rows = configs[c].kv_rows;
        uint32_t q_pad_rows = (q_rows + 63u) & ~63u;
        size_t q_count = (size_t)q_rows * HEADS * HEAD_DIM;
        size_t q_pad_count = (size_t)q_pad_rows * HEADS * HEAD_DIM;
        size_t kv_count = (size_t)kv_rows * HEADS * HEAD_DIM;

        h3_gpu_tensor *query = fill_bf16(gpu, q_pad_count, &rng);
        h3_gpu_tensor *key = fill_bf16(gpu, kv_count, &rng);
        h3_gpu_tensor *value = fill_bf16(gpu, kv_count, &rng);
        h3_gpu_tensor *out_flash = h3_gpu_tensor_new_bf16(gpu, q_count);
        h3_gpu_tensor *out_fused = configs[c].prefix
            ? NULL : h3_gpu_tensor_new_bf16(gpu, q_count);
        h3_gpu_tensor *kv_limit = NULL;
        uint32_t *limits = NULL;
        double limit_sum = 0.0;
        int ok = query && key && value && out_flash &&
                 (configs[c].prefix ? 1 : out_fused != NULL);
        if (ok && configs[c].prefix) {
            /* Representative ti2i table: 34 text rows causal, then
             * 4096-row image blocks attending through their block end. */
            uint32_t text_rows = 34;
            uint32_t block = 4096;
            limits = malloc(sizeof(uint32_t) * q_rows);
            if (limits) {
                for (uint32_t r = 0; r < q_rows; r++) {
                    limits[r] = r < text_rows
                        ? r + 1
                        : text_rows + (((r - text_rows) / block) + 1) * block;
                    if (limits[r] > kv_rows) limits[r] = kv_rows;
                    limit_sum += limits[r];
                }
                kv_limit = h3_gpu_tensor_from_u32(gpu, limits, q_rows);
            }
            ok = limits && kv_limit;
        }
        if (!ok) {
            printf("%s: alloc failed\n", configs[c].name);
            return 1;
        }

        double flops_flash = configs[c].prefix
            ? 4.0 * limit_sum * HEADS * HEAD_DIM
            : 4.0 * q_rows * kv_rows * HEADS * HEAD_DIM;

        /* flash2 (or production prefix kernel), warmup + timed. */
        for (int i = 0; i < WARMUP; i++) {
            ok = configs[c].prefix
                ? qw_gpu_flash_sdpa_prefix_bf16(gpu, out_flash, query, key,
                                                value, kv_limit, q_rows,
                                                kv_rows, HEADS, HEADS,
                                                HEAD_DIM, SCALE)
                : qw_gpu_flash_sdpa_bf16(gpu, out_flash, query, key, value,
                                         q_rows, kv_rows, HEADS, HEAD_DIM,
                                         SCALE);
            if (!ok) {
                printf("%s: flash warmup: %s\n", configs[c].name,
                       h3_gpu_error(gpu));
                return 1;
            }
        }
        if (!sync_gpu(gpu)) return 1;
        double start = now_seconds();
        for (int i = 0; i < ITERS; i++) {
            ok = configs[c].prefix
                ? qw_gpu_flash_sdpa_prefix_bf16(gpu, out_flash, query, key,
                                                value, kv_limit, q_rows,
                                                kv_rows, HEADS, HEADS,
                                                HEAD_DIM, SCALE)
                : qw_gpu_flash_sdpa_bf16(gpu, out_flash, query, key, value,
                                         q_rows, kv_rows, HEADS, HEAD_DIM,
                                         SCALE);
            if (!ok) {
                printf("%s: flash timed: %s\n", configs[c].name,
                       h3_gpu_error(gpu));
                return 1;
            }
        }
        if (!sync_gpu(gpu)) return 1;
        double flash_ms = (now_seconds() - start) * 1e3 / ITERS;

        double fused_ms = -1.0, rel = -1.0;
        if (!configs[c].prefix) {
            for (int i = 0; i < WARMUP; i++) {
                if (!qw_gpu_sdpa_rect_bf16(gpu, out_fused, query, key, value,
                                           q_rows, kv_rows, HEADS, HEAD_DIM,
                                           SCALE)) {
                    printf("%s: fused warmup: %s\n", configs[c].name,
                           h3_gpu_error(gpu));
                    return 1;
                }
            }
            if (!sync_gpu(gpu)) return 1;
            start = now_seconds();
            for (int i = 0; i < ITERS; i++) {
                if (!qw_gpu_sdpa_rect_bf16(gpu, out_fused, query, key, value,
                                           q_rows, kv_rows, HEADS, HEAD_DIM,
                                           SCALE)) {
                    printf("%s: fused timed: %s\n", configs[c].name,
                           h3_gpu_error(gpu));
                    return 1;
                }
            }
            if (!sync_gpu(gpu)) return 1;
            fused_ms = (now_seconds() - start) * 1e3 / ITERS;
            rel = rel_l2(out_flash, out_fused, q_count);
        }

        printf("%s\n", configs[c].name);
        printf("  flash2    %8.1f ms  %6.2f TFLOPS (effective)\n",
               flash_ms, flops_flash / (flash_ms * 1e-3) / 1e12);
        if (fused_ms >= 0) {
            printf("  mps fused %8.1f ms  %6.2f TFLOPS  relL2(flash) %.3g\n",
                   fused_ms, flops_flash / (fused_ms * 1e-3) / 1e12, rel);
            if (rel < 0 || rel > 0.1) {
                printf("  FAIL: fused output disagrees with flash2\n");
                return 1;
            }
        } else {
            printf("  prefix FLOPs %.4g of full (limit_sum %.4g / %.4g)\n",
                   limit_sum / ((double)q_rows * kv_rows), limit_sum,
                   (double)q_rows * kv_rows);
        }

        free(limits);
        h3_gpu_tensor_free(query);
        h3_gpu_tensor_free(key);
        h3_gpu_tensor_free(value);
        h3_gpu_tensor_free(out_flash);
        h3_gpu_tensor_free(out_fused);
        h3_gpu_tensor_free(kv_limit);
    }
    return 0;
}
