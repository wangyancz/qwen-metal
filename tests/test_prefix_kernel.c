/* Prefix-visible SDPA kernel isolation test: small joint sequence,
 * CPU-computed expectation, no DiT involvement. */
#include "qwen_gpu_ext.h"

#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADS 32
#define D 128
#define ROWS 149 /* dit ti2i scale: 3 query groups, non-multiple of 64 */

static uint32_t rng_state = 20261028;
static float host_rand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, 4);
    return value;
}

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu create failed: %s\n", error);
        return 1;
    }

    uint16_t *q = malloc(sizeof(uint16_t) * ROWS * HEADS * D);
    uint16_t *k = malloc(sizeof(uint16_t) * ROWS * HEADS * D);
    uint16_t *v = malloc(sizeof(uint16_t) * ROWS * HEADS * D);
    uint32_t *limits = malloc(sizeof(uint32_t) * ROWS);
    float *q_f = malloc(sizeof(float) * ROWS * HEADS * D);
    float *k_f = malloc(sizeof(float) * ROWS * HEADS * D);
    float *v_f = malloc(sizeof(float) * ROWS * HEADS * D);
    for (uint32_t i = 0; i < ROWS * HEADS * D; i++) {
        q_f[i] = host_rand();
        k_f[i] = host_rand();
        v_f[i] = host_rand();
        q[i] = f32_to_bf16(q_f[i]);
        k[i] = f32_to_bf16(k_f[i]);
        v[i] = f32_to_bf16(v_f[i]);
    }
    for (uint32_t i = 0; i < ROWS * HEADS * D; i++) {
        q_f[i] = bf16_to_f32(q[i]);
        k_f[i] = bf16_to_f32(k[i]);
        v_f[i] = bf16_to_f32(v[i]);
    }

    /* ti2i-like segments: 8 text (causal), 12 image (block end), rest
     * text (causal). Row limits: causal / run end. */
    for (uint32_t r = 0; r < 20; r++) limits[r] = r + 1;
    for (uint32_t r = 20; r < 68; r++) limits[r] = 68;
    for (uint32_t r = 68; r < 85; r++) limits[r] = 68 + (r - 68) + 1;
    for (uint32_t r = 85; r < ROWS; r++) limits[r] = ROWS;

    /* CPU reference: softmax over [0, limit) with the same bf16 inputs. */
    float *ref = calloc(ROWS * HEADS * D, sizeof(float));
    for (uint32_t r = 0; r < ROWS; r++) {
        for (uint32_t h = 0; h < HEADS; h++) {
            float scores[ROWS];
            float m = -3.402823466e+38f;
            for (uint32_t s = 0; s < limits[r]; s++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++)
                    dot += q_f[(r * HEADS + h) * D + d] *
                           k_f[(s * HEADS + h) * D + d];
                scores[s] = dot / sqrtf((float)D);
                if (scores[s] > m) m = scores[s];
            }
            float denom = 0.0f;
            for (uint32_t s = 0; s < limits[r]; s++) {
                scores[s] = expf(scores[s] - m);
                denom += scores[s];
            }
            for (uint32_t s = 0; s < limits[r]; s++)
                for (uint32_t d = 0; d < D; d++)
                    ref[(r * HEADS + h) * D + d] +=
                        scores[s] / denom *
                        v_f[(s * HEADS + h) * D + d];
        }
    }

    uint32_t q_pad_rows = (ROWS + 63u) & ~63u;
    h3_gpu_tensor *qt =
        h3_gpu_tensor_new_bf16(gpu, (size_t)q_pad_rows * HEADS * D);
    h3_gpu_tensor *kt =
        h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * HEADS * D);
    h3_gpu_tensor *vt =
        h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * HEADS * D);
    h3_gpu_tensor *ot =
        h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * HEADS * D);
    h3_gpu_tensor *lt = h3_gpu_tensor_from_u32(gpu, limits, ROWS);
    if (!qt || !kt || !vt || !ot || !lt) {
        printf("alloc failed\n");
        return 1;
    }
    if (!h3_gpu_begin(gpu)) {
        printf("begin: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    h3_gpu_tensor_write_bf16(qt, q, (size_t)ROWS * HEADS * D);
    h3_gpu_tensor_write_bf16(kt, k, (size_t)ROWS * HEADS * D);
    h3_gpu_tensor_write_bf16(vt, v, (size_t)ROWS * HEADS * D);
    if (!qw_gpu_flash_sdpa_prefix_bf16(gpu, ot, qt, kt, vt, lt, ROWS, ROWS,
                                       HEADS, HEADS, D,
                                       1.0f / sqrtf((float)D))) {
        printf("kernel failed: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!h3_gpu_submit(gpu)) {
        printf("submit: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    uint16_t *host = malloc(sizeof(uint16_t) * ROWS * HEADS * D);
    h3_gpu_tensor_read_bf16(ot, host, (size_t)ROWS * HEADS * D);

    double num = 0, den = 0;
    for (size_t i = 0; i < (size_t)ROWS * HEADS * D; i++) {
        float g = bf16_to_f32(host[i]);
        double diff = (double)g - ref[i];
        num += diff * diff;
        den += (double)ref[i] * ref[i];
    }
    double rel = sqrt(num / (den > 0 ? den : 1));
    printf("prefix kernel relL2 %.3e vs CPU reference\n", rel);
    printf(rel < 5e-3 ? "PASS\n" : "FAIL\n");

    free(q);
    free(k);
    free(v);
    free(limits);
    free(q_f);
    free(k_f);
    free(v_f);
    free(ref);
    free(host);
    h3_gpu_tensor_free(qt);
    h3_gpu_tensor_free(kt);
    h3_gpu_tensor_free(vt);
    h3_gpu_tensor_free(ot);
    h3_gpu_tensor_free(lt);
    h3_gpu_free(gpu);
    return 0;
}
