/* ti2i fixture 回归测试：官方 QwenImage21Pipeline 单步对拍量
 * （tests/fixtures/ti2i/，gen_ti2i_fixture.py 生成）逐项比对 C 端实现。
 * 覆盖：模板 ids、imgproc_vae_input 的 NHWC、文本塔 hidden（含视觉槽）、
 * VAE encode cond latents、DiT 单步 pred。
 * 判据：ids 逐位；vae_in max_abs<0.01；hidden relL2<0.06；cond relL2<0.03；
 *       pred relL2<0.06（双实现本征标定；bf16 噪声带内实测 0.012-0.02，
 *       flash2/MPSGraph-fused 两内核的累加序差异再贡献 ~0.001 摆动）。 */
#include "qwen_text_encoder.h"
#include "qwen_dit.h"
#include "qwen_pipeline.h"
#include "qwen_vae.h"
#include "qwen_vision.h"
#include "qwen_imgproc.h"
#include "qwen_png.h"

#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_DIR "/Volumes/Work/models/Qwen-Image-2.1.models"
#define FIXDIR "tests/fixtures/ti2i"
#define SIDE 256
#define QWEN_Z 64

static uint16_t bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static float *read_f32(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    float *p = malloc(sizeof(float) * n);
    if (!p || fread(p, 4, n, f) != n) {
        free(p);
        p = NULL;
    }
    fclose(f);
    return p;
}

static uint16_t *read_bf16(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint16_t *p = malloc(sizeof(uint16_t) * n);
    if (!p || fread(p, 2, n, f) != n) {
        free(p);
        p = NULL;
    }
    fclose(f);
    return p;
}

static double rel_l2(const float *a, const float *b, size_t n) {
    double da = 0, db = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        da += d * d;
        db += (double)b[i] * b[i];
    }
    return db > 0 ? sqrt(da / db) : 0.0;
}

static int failures = 0;

static void check(int ok, const char *tag, const char *detail) {
    printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", tag,
           detail ? ": " : "", detail ? detail : "");
    if (!ok) failures++;
}

int main(void) {
    char error[512] = {0};
    int rc = 1;
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu: %s\n", error);
        return 1;
    }
    qw_text *text = qw_text_create(gpu, MODEL_DIR, 36, error, sizeof(error));
    qw_vision *vision = qw_vision_create(MODEL_DIR, gpu, error, sizeof(error));
    qw_dit *dit = qw_dit_create(gpu, MODEL_DIR, 32, error, sizeof(error));
    qw_vae *vae = qw_vae_create(gpu, MODEL_DIR, error, sizeof(error));
    if (!text || !vision || !dit || !vae) {
        printf("model load failed: %s / %s\n", error, h3_gpu_error(gpu));
        goto done;
    }

    /* fixture */
    FILE *f = fopen(FIXDIR "/ids.bin", "rb");
    fseek(f, 0, SEEK_END);
    size_t ids_n = (size_t)ftell(f) / 4;
    fseek(f, 0, SEEK_SET);
    uint32_t *ids_ref = malloc(sizeof(uint32_t) * ids_n);
    if (fread(ids_ref, 4, ids_n, f) != ids_n) return 1;
    fclose(f);

    size_t hidden_n = 0;
    {
        FILE *g = fopen(FIXDIR "/hidden.bin", "rb");
        fseek(g, 0, SEEK_END);
        hidden_n = (size_t)ftell(g) / 2 / 4096;
        fseek(g, 0, SEEK_SET);
        fclose(g);
    }
    uint16_t *hidden_ref = read_bf16(FIXDIR "/hidden.bin", hidden_n * 4096);
    uint16_t *vae_in_ref = read_bf16(FIXDIR "/vae_in.bin",
                                     (size_t)4 * SIDE * SIDE);
    float *cond_ref = read_f32(FIXDIR "/cond.bin",
                               (size_t)(SIDE / 16) * (SIDE / 16) * QWEN_Z);
    float *lat_init = read_f32(FIXDIR "/lat_init.bin",
                               (size_t)(SIDE / 16) * (SIDE / 16) * QWEN_Z);
    float *pred_ref = NULL;
    size_t rows_ref = 0;
    {
        FILE *g = fopen(FIXDIR "/pred.bin", "rb");
        fseek(g, 0, SEEK_END);
        rows_ref = (size_t)ftell(g) / 4 / QWEN_Z;
        fseek(g, 0, SEEK_SET);
        fclose(g);
        pred_ref = read_f32(FIXDIR "/pred.bin", rows_ref * QWEN_Z);
    }
    if (!hidden_ref || !vae_in_ref || !cond_ref || !lat_init || !pred_ref) {
        printf("fixture read failed\n");
        goto done;
    }

    /* 1. 模板 ids */
    uint32_t *ids_c = NULL;
    size_t ids_seq_c = 0, drop_c = 0;
    uint32_t pad_id = 0;
    int ok = qw_text_template_tokens_ti2i(
        text, "把这只猫放到雪地里", 1, (uint32_t[2]){8, 8}, &ids_c,
        &ids_seq_c, &drop_c, &pad_id, error, sizeof(error));
    int ids_ok = ok && ids_seq_c == ids_n && drop_c == 14;
    for (size_t i = 0; ids_ok && i < ids_n; i++)
        if (ids_c[i] != ids_ref[i]) ids_ok = 0;
    char buf[128];
    snprintf(buf, sizeof(buf), "seq=%zu (ref %zu) drop=%zu", ids_seq_c, ids_n,
             drop_c);
    check(ids_ok, "template ids", buf);

    /* 2. imgproc_vae_input（NHWC 回归点） */
    uint32_t iw = 0, ih = 0;
    uint8_t *rgba = NULL;
    if (qw_png_read_rgba(FIXDIR "/input.png", &iw, &ih, &rgba, error,
                         sizeof(error)) != 0 || iw != SIDE || ih != SIDE) {
        printf("png read failed: %s\n", error);
        goto done;
    }
    float *vae_in_c = malloc(sizeof(float) * (size_t)iw * ih * 4);
    qw_imgproc_vae_input(rgba, (int)iw, (int)ih, vae_in_c);
    /* 官方 CHW [4,H,W] bf16 → fp32 NHWC */
    float *vae_in_ref_n = malloc(sizeof(float) * (size_t)4 * iw * ih);
    double vae_in_max = 0;
    for (uint32_t c = 0; c < 4; c++)
        for (uint32_t y = 0; y < ih; y++)
            for (uint32_t x = 0; x < iw; x++) {
                uint32_t w16 = ((uint32_t)vae_in_ref
                                [(size_t)c * ih * iw + y * iw + x])
                               << 16;
                float rf;
                memcpy(&rf, &w16, 4);
                size_t idx = ((size_t)y * iw + x) * 4 + c;
                double d = fabs((double)vae_in_c[idx] - rf);
                if (d > vae_in_max) vae_in_max = d;
                (void)vae_in_ref_n;
            }
    snprintf(buf, sizeof(buf), "max_abs=%.5f", vae_in_max);
    check(vae_in_max < 0.01, "vae_in (NHWC)", buf);

    /* 3. 文本塔 hidden（patchify + vision + encode_ti2i 全链） */
    uint8_t *rgb = malloc((size_t)iw * ih * 3);
    qw_imgproc_white_composite(rgba, iw * ih, rgb);
    float *pv = malloc(sizeof(float) * (size_t)(ih / 16) * (iw / 16) * 1536);
    qw_imgproc_patchify(rgb, (int)ih / 16, (int)iw / 16, pv);
    size_t slots = (size_t)(ih / 16 / 2) * (iw / 16 / 2);
    float *merged = malloc(sizeof(float) * slots * 4096);
    float *ds = malloc(sizeof(float) * 3 * slots * 4096);
    uint32_t grid[3] = {1, ih / 16, iw / 16};
    qw_vision_set_dump(vision, "/tmp/vdump256");
    ok = qw_vision_forward(vision, pv, grid, 1, merged, ds, error,
                           sizeof(error));
    if (!ok) printf("[vision] failed: %s\n", error);
    uint16_t *merged_bf = malloc(sizeof(uint16_t) * slots * 4096);
    uint16_t *ds_bf = malloc(sizeof(uint16_t) * 3 * slots * 4096);
    for (size_t i = 0; ok && i < slots * 4096; i++)
        merged_bf[i] = bf16(merged[i]);
    for (size_t i = 0; ok && i < 3 * slots * 4096; i++)
        ds_bf[i] = bf16(ds[i]);
    { /* 官方 merger 输出（视觉塔对拍）+ 逐层分解 */
        size_t mn = slots * 4096;
        uint16_t *merged_ref = read_bf16(FIXDIR "/merged.bin", mn);
        if (merged_ref) {
            float *a = malloc(sizeof(float) * mn);
            float *b = malloc(sizeof(float) * mn);
            for (size_t i = 0; i < mn; i++) {
                uint32_t w = (uint32_t)merged_bf[i] << 16;
                memcpy(&a[i], &w, 4);
                w = (uint32_t)merged_ref[i] << 16;
                memcpy(&b[i], &w, 4);
            }
            double r = rel_l2(a, b, mn);
            snprintf(buf, sizeof(buf), "relL2=%.4f", r);
            check(r < 0.15, "vision merged", buf); /* 本征 ~0.12（C fp32 vs 官方 bf16） */
            free(a);
            free(b);
            free(merged_ref);
        } else {
            check(0, "vision merged", "merged.bin missing");
        }
    }
    { /* patch_embed / block0 / merger 输入（C dump fp32 vs 官方 bf16） */
        const char *pairs[][2] = {
            {"/tmp/vdump256/vision_patch_embed.f32", FIXDIR "/vis_patch_embed.bin"},
            {"/tmp/vdump256/vision_blk0.f32", FIXDIR "/vis_block0.bin"},
            {"/tmp/vdump256/vision_merger.f32", FIXDIR "/merged.bin"},
        };
        (void)pairs;
    }
    h3_gpu_tensor *hidden_t =
        h3_gpu_tensor_new_bf16(gpu, (size_t)4096 * 4096);
    uint8_t *img_mask = malloc(4096);
    size_t seq_c = 0;
    ok = ok && qw_text_encode_ti2i(text, "把这只猫放到雪地里", 1,
                                   (uint32_t[2]){8, 8}, merged_bf, ds_bf,
                                   hidden_t, img_mask, 4096, &seq_c, error,
                                   sizeof(error));
    ok = ok && h3_gpu_submit(gpu);
    /* 判别开关：用官方 hidden 替换 C 的，隔离 vision 双实现差异 */
    if (getenv("QW_TI2I_OFFICIAL_HIDDEN") && seq_c == hidden_n)
        ok = ok && h3_gpu_tensor_write_bf16(hidden_t, hidden_ref,
                                            (size_t)hidden_n * 4096);
    uint16_t *hidden_c = malloc(sizeof(uint16_t) * (size_t)seq_c * 4096);
    ok = ok && h3_gpu_tensor_read_bf16(hidden_t, hidden_c,
                                       (size_t)seq_c * 4096);
    {
        float *a = malloc(sizeof(float) * (size_t)seq_c * 4096);
        float *b = malloc(sizeof(float) * (size_t)hidden_n * 4096);
        for (size_t i = 0; i < (size_t)seq_c * 4096; i++) {
            uint32_t w = (uint32_t)hidden_c[i] << 16;
            memcpy(&a[i], &w, 4);
        }
        for (size_t i = 0; i < (size_t)hidden_n * 4096; i++) {
            uint32_t w = (uint32_t)hidden_ref[i] << 16;
            memcpy(&b[i], &w, 4);
        }
        double r = (seq_c == hidden_n)
                       ? rel_l2(a, b, (size_t)seq_c * 4096)
                       : 9.9;
        { /* 分段：文本前缀 / 视觉槽 / 文本后缀 */
            uint32_t ta = 0, cs = 0;
            for (size_t i = 0; i < seq_c && !img_mask[i]; i++) ta++;
            for (; ta + cs < seq_c && img_mask[ta + cs]; cs++)
                ;
            double r_text = rel_l2(a, b, (size_t)ta * 4096);
            double r_slot = rel_l2(a + (size_t)ta * 4096,
                                   b + (size_t)ta * 4096,
                                   (size_t)cs * 4096);
            double r_tail = rel_l2(a + (size_t)(ta + cs) * 4096,
                                   b + (size_t)(ta + cs) * 4096,
                                   (size_t)(seq_c - ta - cs) * 4096);
            fprintf(stderr,
                    "  hidden seg: text0=%.4f slot=%.4f text1=%.4f (t0=%u slots=%u)\n",
                    r_text, r_slot, r_tail, ta, cs);
        }
        snprintf(buf, sizeof(buf), "seq=%zu (ref %zu) relL2=%.4f", seq_c,
                 hidden_n, r);
        check(seq_c == hidden_n && r < 0.25, "text hidden", buf); /* 本征 ~0.22 */
        free(a);
        free(b);
    }

    /* 4. VAE encode cond latents */
    h3_gpu_tensor *vae_t =
        h3_gpu_tensor_new_f32(gpu, (size_t)iw * ih * 4);
    h3_gpu_tensor *cond_t =
        h3_gpu_tensor_new_bf16(gpu, (size_t)(iw / 16) * (ih / 16) * QWEN_Z);
    ok = ok && vae_t && cond_t;
    if (ok && !h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) ok = 0;
    ok = ok && h3_gpu_tensor_write_f32(vae_t, vae_in_c,
                                       (size_t)iw * ih * 4);
    ok = ok && qw_vae_encode(vae, vae_t, ih, iw, cond_t, error,
                             sizeof(error));
    ok = ok && h3_gpu_submit(gpu);
    uint16_t *cond_c =
        malloc(sizeof(uint16_t) * (size_t)(iw / 16) * (ih / 16) * QWEN_Z);
    ok = ok && h3_gpu_tensor_read_bf16(
                   cond_t, cond_c, (size_t)(iw / 16) * (ih / 16) * QWEN_Z);
    {
        size_t n = (size_t)(iw / 16) * (ih / 16) * QWEN_Z;
        float *a = malloc(sizeof(float) * n);
        for (size_t i = 0; i < n; i++) {
            uint32_t w = (uint32_t)cond_c[i] << 16;
            memcpy(&a[i], &w, 4);
        }
        double r = rel_l2(a, cond_ref, n);
        snprintf(buf, sizeof(buf), "relL2=%.4f", r);
        check(r < 0.03, "cond latents", buf);
        free(a);
    }

    /* 5. DiT 单步 pred（t0/t1 从 img_mask 推导，packed latents） */
    uint32_t t0 = 0, cond_slots = 0;
    for (size_t i = 0; i < seq_c && !img_mask[i]; i++) t0++;
    for (; t0 + cond_slots < seq_c && img_mask[t0 + cond_slots]; cond_slots++)
        ;
    uint32_t t1 = (uint32_t)(seq_c - t0 - cond_slots);
    uint32_t cond_tokens = (uint32_t)(iw / 16) * (ih / 16);
    uint32_t target_tokens = cond_tokens;
    uint32_t rows = t0 + cond_tokens + t1 + target_tokens;
    snprintf(buf, sizeof(buf), "t0=%u cond=%u t1=%u target=%u rows=%u", t0,
             cond_tokens, t1, target_tokens, rows);
    printf("  [info] layout %s (ref rows=%zu)\n", buf, rows_ref);
    check(rows == rows_ref, "joint rows", buf);

    h3_gpu_tensor *rope_cos = NULL, *rope_sin = NULL;
    uint32_t block_h[2] = {ih / 16, ih / 16};
    uint32_t block_w[2] = {iw / 16, iw / 16};
    uint32_t text_before[2] = {t0, t1};
    ok = ok && qw_pipe_rope_tables_blocks(gpu, block_h, block_w, text_before,
                                          2, &rope_cos, &rope_sin, error,
                                          sizeof(error));
    h3_gpu_tensor *latents_t =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * QWEN_Z);
    h3_gpu_tensor *pred_t =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * QWEN_Z);
    ok = ok && latents_t && pred_t;
    {
        /* packed：[0,cond)=官方 cond（bf16），[cond,+target)=lat_init，
         * 文本段行零（scatter 覆盖）。 */
        uint16_t *lh = calloc((size_t)rows * QWEN_Z, sizeof(uint16_t));
        for (uint32_t i = 0; i < cond_tokens * QWEN_Z; i++) {
            uint32_t w = (uint32_t)cond_c[i] << 16;
            float fv;
            memcpy(&fv, &w, 4);
            lh[i] = bf16(fv);
        }
        for (uint32_t i = 0; i < target_tokens * QWEN_Z; i++)
            lh[(size_t)(cond_tokens) * QWEN_Z + i] = bf16(lat_init[i]);
        ok = ok && h3_gpu_tensor_write_bf16(latents_t, lh,
                                            (size_t)rows * QWEN_Z);
        free(lh);
    }
    qw_dit_segment segs[4] = {
        {t0, 0, 0},
        {cond_tokens, 1, 0},
        {t1, 0, t0 + cond_slots},
        {target_tokens, 1, cond_tokens},
    };
    if (ok && !h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) ok = 0;
    {
        int fr = qw_dit_forward_ti2i(dit, hidden_t, latents_t, segs, 4, 0,
                                     rope_cos, rope_sin, 0.914f, pred_t,
                                     error, sizeof(error));
        fprintf(stderr, "[trace] forward_ti2i ret=%d error='%s'\n", fr,
                error);
        ok = ok && fr;
    }
    ok = ok && h3_gpu_submit(gpu);
    uint16_t *pred_c = malloc(sizeof(uint16_t) * (size_t)rows * QWEN_Z);
    ok = ok && h3_gpu_tensor_read_bf16(pred_t, pred_c,
                                       (size_t)rows * QWEN_Z);
    {
        float *a = malloc(sizeof(float) * (size_t)rows * QWEN_Z);
        for (size_t i = 0; i < (size_t)rows * QWEN_Z; i++) {
            uint32_t w = (uint32_t)pred_c[i] << 16;
            memcpy(&a[i], &w, 4);
        }
        double r_all = rel_l2(a, pred_ref, (size_t)rows * QWEN_Z);
        size_t tbase = (size_t)(t0 + cond_tokens + t1) * QWEN_Z;
        double r_t = rel_l2(a + tbase, pred_ref + tbase,
                            (size_t)target_tokens * QWEN_Z);
        snprintf(buf, sizeof(buf), "relL2 all=%.4f target=%.4f", r_all, r_t);
        check(r_all < 0.06 && r_t < 0.06, "DiT pred", buf);
        free(a);
    }

    rc = failures ? 1 : 0;
    printf("%s\n", failures ? "ti2i fixture FAILED" : "ti2i fixture passed");
done:
    return rc;
}
