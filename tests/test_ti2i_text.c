/* ti2i 文本塔对照测试：同一组视觉 embeds，分别走
 * qw_text_encode_vlm（V1b 已 fixture 验证）与 qw_text_encode_ti2i（新路径），
 * 比较输出 hidden 的 RMS / NaN，定位 ti2i 组装的问题。
 *
 * 诊断开关（2026-10-29 排查固定纯色 bug）：
 *   QW_MINI_ZERO_COND=1  视觉 merged/deepstack 与 cond latents 全部置零，
 *                        用于判定 DiT 输出是否对条件有响应。
 *   pred 全量 dump 到 /tmp/mini_pred.bin（int64 头 [rows,64] + fp32）。 */
#include "qwen_text_encoder.h"
#include "qwen_dit.h"
#include "qwen_pipeline.h"

#include "h3_gpu.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_DIR "/Volumes/Work/models/Qwen-Image-2.1.models"
#define SLOTS 1024 /* 官方对拍规模: media.png 1024x1024 -> merged 32x32 */
#define GRID_H 32
#define GRID_W 32
#define PROMPT "把苹果的颜色改成绿色"

static uint32_t rng = 7;
static float rnd(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / 8388608.0f) - 1.0f;
}

static uint16_t bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static void report(const char *tag, const uint16_t *h, size_t rows) {
    size_t n = rows * 4096;
    double s = 0;
    long nan = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t w = (uint32_t)h[i] << 16;
        float f;
        memcpy(&f, &w, 4);
        if (f != f) {
            nan++;
            continue;
        }
        s += (double)f * f;
    }
    printf("  %-12s rows=%zu nan=%ld rms=%.4g\n", tag, rows, nan,
           sqrt(s / (double)n));
}

static void seg_rms(const char *tag, const float *p, size_t off, size_t rows) {
    double s = 0;
    long nan = 0;
    for (size_t i = off * 64; i < (off + rows) * 64; i++) {
        if (p[i] != p[i]) { nan++; continue; }
        s += (double)p[i] * p[i];
    }
    printf("  pred %-14s off=%zu rows=%zu nan=%ld rms=%.4g\n", tag, off,
           rows, nan, rows ? sqrt(s / (double)(rows * 64)) : 0.0);
}

int main(void) {
    int zero_cond = getenv("QW_MINI_ZERO_COND") != NULL;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu: %s\n", error);
        return 1;
    }
    qw_text *text = qw_text_create(gpu, MODEL_DIR, 36, error, sizeof(error));
    if (!text) {
        printf("text: %s\n", error);
        return 1;
    }

    uint16_t *merged = malloc(sizeof(uint16_t) * (size_t)SLOTS * 4096);
    uint16_t *deep =
        malloc(sizeof(uint16_t) * 3 * (size_t)SLOTS * 4096);
    FILE *f1 = fopen("/tmp/of_merged.bin", "rb");
    FILE *f2 = fopen("/tmp/of_ds.bin", "rb");
    if (f1 && fread(merged, 2, (size_t)SLOTS*4096, f1) == (size_t)SLOTS*4096) {
        printf("loaded real merged\n");
    } else {
        printf("using random merged\n");
        for (size_t i = 0; i < (size_t)SLOTS*4096; i++)
            merged[i] = bf16(rnd() * 0.1f);
    }
    if (f1) fclose(f1);
    if (f2 && fread(deep, 2, 3*(size_t)SLOTS*4096, f2) == 3*(size_t)SLOTS*4096) {
        printf("loaded real deepstack\n");
    } else {
        printf("using random deepstack\n");
        for (size_t i = 0; i < 3*(size_t)SLOTS*4096; i++)
            deep[i] = bf16(rnd() * 0.1f);
    }
    if (f2) fclose(f2);
    if (zero_cond) {
        printf("[ZERO_COND] merged/deepstack zeroed\n");
        memset(merged, 0, sizeof(uint16_t) * (size_t)SLOTS * 4096);
        memset(deep, 0, sizeof(uint16_t) * 3 * (size_t)SLOTS * 4096);
    }
    uint8_t *mask = malloc(4096);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, (size_t)4096 * 4096);
    size_t seq = 0;
    uint16_t *host = malloc(sizeof(uint16_t) * (size_t)4096 * 4096);

    /* 路径 A：V1b 已验证的 encode_vlm（ids 序列由外部给出——这里用 ti2i
     * 模板生成的 ids 以保证两条路径输入一致） */
    printf("ti2i 文本塔对照:\n");
    uint32_t *ids = NULL;
    size_t ids_seq = 0;
    uint32_t pad_id = 0;
    size_t drop = 0;
    if (!qw_text_template_tokens_ti2i(text, PROMPT, 1,
                                       (uint32_t[2]){GRID_H, GRID_W}, &ids,
                                       &ids_seq, &drop, &pad_id, error,
                                       sizeof(error))) {
        printf("template: %s\n", error);
        return 1;
    }
    printf("  template seq=%zu drop=%zu pad_id=%u\n", ids_seq, drop, pad_id);
    { FILE *fi = fopen("/tmp/c_ids.txt", "w");
      if (fi) { for (size_t i = 0; i < ids_seq; i++) fprintf(fi, "%u ", ids[i]);
                fclose(fi); printf("saved /tmp/c_ids.txt\n"); } }
    { uint32_t mx=0; size_t padcnt=0;
      for (size_t i=0;i<ids_seq;i++){ if (ids[i]>mx) mx=ids[i]; if (ids[i]==pad_id) padcnt++; }
      printf("  max_id=%u (vocab 151936) pad_count=%zu\n", mx, padcnt); }
    /* 对照：t2i 已验证的 qw_text_encode（同 ids，一维退化 rope） */
    h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu, (size_t)ids_seq * 4096);
    int ok = x && qw_text_encode(text, ids + drop, (uint32_t)(ids_seq - drop),
                                 x, error, sizeof(error));
    ok = ok && h3_gpu_submit(gpu);
    ok = ok && h3_gpu_tensor_read_bf16(x, host, (size_t)(ids_seq - drop) * 4096);
    if (ok) report("t2i encode", host, ids_seq - drop);

    /* 路径 B：encode_ti2i（tokenize + scatter + encode_core） */
    ok = qw_text_encode_ti2i(text, PROMPT, 1,
                             (uint32_t[2]){GRID_H, GRID_W}, merged, deep,
                             hidden, mask, 4096, &seq, error, sizeof(error));
    if (!ok) {
        printf("encode_ti2i: %s\n", error);
        return 1;
    }
    printf("  encode_ti2i seq=%zu\n", seq);
    if (!h3_gpu_submit(gpu)) printf("submit: %s\n", h3_gpu_error(gpu));
    if (!h3_gpu_tensor_read_bf16(hidden, host, (size_t)seq * 4096))
        printf("read: %s\n", h3_gpu_error(gpu));
    report("ti2i hidden", host, seq);
    { FILE *fh = fopen("/tmp/mini_hidden.bin", "wb");
      if (fh) {
          long hdr[2] = {(long)seq, 4096};
          float *fb = malloc(sizeof(float) * (size_t)seq * 4096);
          if (fb) {
              for (size_t q = 0; q < (size_t)seq * 4096; q++) {
                  uint32_t w = (uint32_t)host[q] << 16;
                  memcpy(&fb[q], &w, 4);
              }
              fwrite(hdr, 8, 2, fh);
              fwrite(fb, 4, (size_t)seq * 4096, fh);
              fclose(fh);
              printf("saved /tmp/mini_hidden.bin\n");
              free(fb);
          } else fclose(fh);
      } }

    if (getenv("QW_IDS_ONLY")) {
        printf("[IDS_ONLY] exit\n");
        return 0;
    }
    /* 路径 C：DiT forward_ti2i（mini 版全链路） */
    qw_dit *dit = qw_dit_create(gpu, MODEL_DIR, 32, error, sizeof(error));
    if (!dit) {
        printf("dit: %s\n", error);
        return 1;
    }
    /* 官方对拍模式：官方 ehs + 官方 latents（hook0 的 hs），布局与官方
     * transformer 完全一致（t0=8 文本、cond 4096、t1=11、target 4096）。 */
    uint32_t t0 = 8;
    uint32_t vlm_slots = SLOTS;
    uint32_t t1 = seq - t0 - vlm_slots;
    uint32_t cond_tokens = vlm_slots * 4;
    uint32_t target_tokens = vlm_slots * 4;
    uint32_t rows = t0 + cond_tokens + t1 + target_tokens;
    uint32_t block_h[2] = {64, 64};
    uint32_t block_w[2] = {64, 64};
    uint32_t text_before[2] = {t0, t1};
    h3_gpu_tensor *rope_cos = NULL, *rope_sin = NULL;
    if (!qw_pipe_rope_tables_blocks(gpu, block_h, block_w, text_before, 2,
                                    &rope_cos, &rope_sin, error,
                                    sizeof(error))) {
        printf("rope: %s\n", error);
        return 1;
    }
    qw_dit_segment segs[4] = {
        {t0, 0, 0},
        {cond_tokens, 1, 0},
        {t1, 0, t0 + SLOTS},
        {target_tokens, 1, cond_tokens},
    };
    h3_gpu_tensor *latents_t =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * 64);
    h3_gpu_tensor *pred =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * 64);
    if (!latents_t || !pred) {
        printf("dit alloc failed\n");
        return 1;
    }
    /* latents: 官方 hook0 的 hs（[8192,64] = cond 4096 + target 4096）转
     * C 端文本语义布局（[t0 零, cond, t1 零, target]）；hidden 换官方 ehs。 */
    {
        uint16_t *lh = calloc((size_t)rows * 64, sizeof(uint16_t));
        float *hs = malloc(sizeof(float) * (size_t)8192 * 64);
        FILE *fh2 = fopen("/tmp/of_hs_0.npy", "rb");
        int got_hs = 0;
        if (fh2) {
            fseek(fh2, 0, SEEK_END);
            long sz = ftell(fh2);
            fseek(fh2, 0, SEEK_SET);
            /* npy 头：跳到数据（官方 hook 用 np.save，头 128 字节） */
            if (sz == (long)(128 + 8192 * 64 * 4)) {
                fseek(fh2, 128, SEEK_SET);
                got_hs = fread(hs, 4, (size_t)8192 * 64, fh2) ==
                         (size_t)8192 * 64;
            }
            fclose(fh2);
        }
        if (got_hs) {
            printf("loaded official hs (cond+target)\n");
            /* packed 布局（与 CLI 管线一致）：cond 行 [0, cond_tokens)、
             * target 行 [cond_tokens, cond_tokens+target_tokens)，文本段零。 */
            for (uint32_t r = 0; r < cond_tokens; r++)
                for (int c = 0; c < 64; c++)
                    lh[(size_t)r * 64 + c] = bf16(hs[(size_t)r * 64 + c]);
            for (uint32_t r = 0; r < target_tokens; r++)
                for (int c = 0; c < 64; c++)
                    lh[(size_t)(cond_tokens + r) * 64 + c] =
                        bf16(hs[((size_t)4096 + r) * 64 + c]);
        } else {
            printf("official hs missing; random latents\n");
            for (size_t i = 0; i < (size_t)rows * 64; i++)
                lh[i] = bf16(rnd());
        }
        free(hs);
        if (zero_cond) {
            printf("[ZERO_COND] cond latents zeroed\n");
            memset(lh + (size_t)t0 * 64, 0,
                   sizeof(uint16_t) * (size_t)cond_tokens * 64);
        }
        /* 官方 ehs 覆盖 hidden（fp32 npy -> bf16） */
        float *ehs = malloc(sizeof(float) * (size_t)seq * 4096);
        FILE *fe = fopen("/tmp/of_ehs.npy", "rb");
        int got_ehs = 0;
        if (fe) {
            fseek(fe, 0, SEEK_END);
            long sz = ftell(fe);
            fseek(fe, 0, SEEK_SET);
            if (sz == (long)(128 + (size_t)seq * 4096 * 4)) {
                fseek(fe, 128, SEEK_SET);
                got_ehs = fread(ehs, 4, (size_t)seq * 4096, fe) ==
                          (size_t)seq * 4096;
            }
            fclose(fe);
        }
        if (got_ehs) {
            printf("loaded official ehs into hidden\n");
            uint16_t *hb = malloc(sizeof(uint16_t) * (size_t)seq * 4096);
            for (size_t q = 0; q < (size_t)seq * 4096; q++)
                hb[q] = bf16(ehs[q]);
            h3_gpu_tensor_write_bf16(hidden, hb, (size_t)seq * 4096);
            free(hb);
        } else {
            printf("official ehs missing; using C encode_ti2i hidden\n");
        }
        free(ehs);
        h3_gpu_tensor_write_bf16(latents_t, lh, (size_t)rows * 64);
        free(lh);
    }
    if (!h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) {
        printf("begin: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    printf("[run] zero_cond=%d no_prefix=%s\n", zero_cond,
           getenv("QW_NO_PREFIX") ? "1" : "0");
    if (!qw_dit_forward_ti2i(dit, hidden, latents_t, segs, 4, 0, rope_cos,
                             rope_sin, 0.9140625f, pred, error, sizeof(error))) {
        printf("forward_ti2i: %s\n", error);
        return 1;
    }
    if (!h3_gpu_submit(gpu)) {
        printf("submit: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    {
        size_t n = (size_t)rows * 64;
        uint16_t *ph = malloc(sizeof(uint16_t) * n);
        float *fb = malloc(sizeof(float) * n);
        long nan = 0;
        double s0 = 0;
        if (ph && h3_gpu_tensor_read_bf16(pred, ph, n)) {
            for (size_t q = 0; q < n; q++) {
                uint32_t w = (uint32_t)ph[q] << 16;
                float f;
                memcpy(&f, &w, 4);
                fb[q] = f;
                if (f != f) {
                    nan++;
                    continue;
                }
                s0 += (double)f * f;
            }
            printf("  pred nan=%ld rms=%.4g\n", nan, sqrt(s0 / (double)n));
            seg_rms("text0", fb, 0, t0);
            seg_rms("cond", fb, t0, cond_tokens);
            seg_rms("text1", fb, t0 + cond_tokens, t1);
            seg_rms("target", fb, t0 + cond_tokens + t1, target_tokens);
            FILE *fp = fopen("/tmp/mini_pred.bin", "wb");
            if (fp) {
                long hdr[2] = {(long)rows, 64};
                fwrite(hdr, 8, 2, fp);
                fwrite(fb, 4, n, fp);
                fclose(fp);
                printf("saved /tmp/mini_pred.bin\n");
            }
        } else {
            printf("pred read failed\n");
        }
        free(ph);
        free(fb);
    }
    (void)block_h;
    (void)block_w;
    (void)text_before;
    (void)rows;
    (void)t0;
    (void)t1;
    (void)cond_tokens;
    (void)target_tokens;
    qw_dit_free(dit);
    (void)hidden;
    (void)mask;
    (void)deep;
    (void)merged;
    h3_gpu_tensor_free(x);
    
    h3_gpu_tensor_free(hidden);
    h3_tokenizer_ids_free(ids);
    free(mask);
    free(host);
    free(merged);
    free(deep);
    qw_text_free(text);
    h3_gpu_free(gpu);
    return 0;
}
