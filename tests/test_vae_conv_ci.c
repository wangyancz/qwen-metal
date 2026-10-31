/* CI=4 conv MM 内核最小复现：随机数据跑 qw_gpu_vae_conv_mm_f32，
 * CI=4（encode conv_in 形状）与 CI=96（decode 形状）对照 numpy 参考实现。 */
#include "qwen_gpu_ext.h"
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 12345;
static float rnd(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / 8388608.0f) - 1.0f;
}

static int run_case(h3_gpu *gpu, uint32_t ci, uint32_t co, uint32_t h,
                    uint32_t w) {
    char error[512] = {0};
    uint32_t co_pad = (co + 63u) & ~63u;
    uint32_t Wp = w + 2;
    size_t x_pad_count = (size_t)(h + 2) * Wp * ci + 64;
    float *x_pad = calloc(x_pad_count, sizeof(float));
    float *w9 = calloc((size_t)9 * co_pad * ci, sizeof(float));
    float *bias = calloc(co, sizeof(float));
    /* 输入：CHW [C,H,W]（pad_ring 内核读这个布局写圈零 NHWC pad） */
    float *x_host = calloc((size_t)h * w * ci, sizeof(float));
    float *x_chw = calloc((size_t)ci * h * w, sizeof(float));
    for (size_t i = 0; i < (size_t)h * w * ci; i++) x_host[i] = rnd();
    for (size_t i = 0; i < (size_t)ci * h * w; i++) x_chw[i] = 0;
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++)
            for (uint32_t c = 0; c < ci; c++)
                x_chw[((size_t)y * w + x) * ci + c] =
                    x_host[((size_t)y * w + x) * ci + c];
    /* 权重 [CO,CI,9] 行主序 -> repack */
    float *w_host = calloc((size_t)co * ci * 9, sizeof(float));
    for (size_t i = 0; i < (size_t)co * ci * 9; i++) w_host[i] = rnd() * 0.3f;
    for (uint32_t c = 0; c < co; c++) bias[c] = rnd() * 0.1f;

    h3_gpu_tensor *x_t = h3_gpu_tensor_from_f32(gpu, x_chw,
                                                (size_t)ci * h * w);
    h3_gpu_tensor *w_t = h3_gpu_tensor_from_f32(gpu, w9, (size_t)9 * co_pad * ci);
    /* w9 先填 [CO,CI,9] 再由 repack 内核重排——直接在 GPU 上做 */
    float *w_src = calloc((size_t)co * ci * 9, sizeof(float));
    memcpy(w_src, w_host, sizeof(float) * (size_t)co * ci * 9);
    h3_gpu_tensor *w_src_t = h3_gpu_tensor_from_f32(gpu, w_src, (size_t)co * ci * 9);
    h3_gpu_tensor *b_t = h3_gpu_tensor_from_f32(gpu, bias, co);
    h3_gpu_tensor *out_t = h3_gpu_tensor_new_f32(gpu, (size_t)h * w * co);
    uint32_t m_span2 = (uint32_t)(((size_t)(h - 1) * Wp + w + 63) / 64 * 64);
    h3_gpu_tensor *arena = h3_gpu_tensor_new_f32(gpu, (size_t)(m_span2 + 2 * Wp + 2) * ci);
    if (!x_t || !w_t || !w_src_t || !b_t || !out_t || !arena) {
        printf("  alloc failed\n");
        return 1;
    }
    if (!h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) { printf("  begin0 failed\n"); return 1; }
    if (!qw_gpu_vae_conv_repack_f32(gpu, w_t, w_src_t, ci, co, co_pad)) {
        printf("  repack failed: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!h3_gpu_submit(gpu)) { printf("  submit: %s\n", h3_gpu_error(gpu)); return 1; }
    if (!h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) {
        printf("  begin failed\n");
        return 1;
    }
    if (!qw_gpu_vae_conv_mm_f32(gpu, out_t, x_t, w_t, b_t, h, w, ci, co,
                                arena)) {
        printf("  conv_mm failed: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!h3_gpu_submit(gpu)) { printf("  submit2: %s\n", h3_gpu_error(gpu)); return 1; }
    float *out = malloc(sizeof(float) * (size_t)h * w * co);
    if (!h3_gpu_tensor_read_f32(out_t, out, (size_t)h * w * co)) {
        printf("  read failed\n");
        return 1;
    }
    /* numpy 参考：3x3 conv 零 pad、行主序 [H*W, CO] */
    double max_err = 0, sum2 = 0, ref2 = 0;
    size_t err_count = 0;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            for (uint32_t oc = 0; oc < co; oc++) {
                double acc = bias[oc];
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = (int)y + dy, xx = (int)x + dx;
                        if (yy < 0 || xx < 0 || yy >= (int)h || xx >= (int)w)
                            continue;
                        for (uint32_t ic = 0; ic < ci; ic++) {
                            float xv = x_host[((size_t)yy * w + xx) * ci + ic];
                            float wv = w_host[((size_t)oc * ci + ic) * 9 +
                                              (size_t)(dy + 1) * 3 + (dx + 1)];
                            acc += (double)xv * wv;
                        }
                    }
                float got = out[((size_t)y * w + x) * co + oc];
                double err = fabs(got - acc);
                if (err > max_err) { max_err = err; }
                if (err > 0.01) {
                    if (err_count == 0)
                        printf("  first err at (y=%u,x=%u,oc=%u): got=%.4f ref=%.4f\n",
                               y, x, oc, got, acc);
                    err_count++;
                }
                sum2 += (double)got * got;
                ref2 += acc * acc;
            }
        }
    }
    size_t n = (size_t)h * w * co;
    printf("  CI=%3u CO=%3u: max_err=%.5f  out_rms=%.4f ref_rms=%.4f err%%=%.3f %s\n", ci,
           co, max_err, sqrt(sum2 / n), sqrt(ref2 / n),
           100.0 * (double)err_count / n,
           err_count ? "MISMATCH" : "OK");
    return 0;
}

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu: %s\n", error);
        return 1;
    }
    printf("CI=96（decode 同款，应 OK）:\n");
    run_case(gpu, 96, 192, 32, 32);
    printf("CI=64（pad 后 conv_in 同款）:\n");
    run_case(gpu, 64, 96, 32, 32);
    printf("CI=4（encode conv_in 同款）:\n");
    run_case(gpu, 4, 96, 32, 32);
    h3_gpu_free(gpu);
    return 0;
}
