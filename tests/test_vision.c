/* 视觉塔 fixture 验收：C 实现逐段对比 tests/vision_fixture/ 的
 * diffusers Qwen3VLVisionModel 真值。门限按 bf16 参考的累积噪声带。 */
#include "qwen_vision.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MODEL_TEXT "/Volumes/Work/models/Qwen-Image-2.1.models/text_encoder"
#define FIXTURES "tests/vision_fixture"
#define DUMP "tests/vision_dump"

static float *load_f32(const char *path, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    float *data = malloc(sizeof(float) * count);
    if (!data || fread(data, sizeof(float), count, f) != count) {
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return data;
}

static double rel_l2(const float *got, const float *ref, size_t count) {
    double num = 0, den = 0;
    for (size_t i = 0; i < count; i++) {
        double diff = (double)got[i] - ref[i];
        num += diff * diff;
        den += (double)ref[i] * ref[i];
    }
    return den > 0 ? sqrt(num / den) : 0.0;
}

static int failures = 0;

static void check(const char *name, size_t count, double gate) {
    char got_path[1200], ref_path[1200];
    snprintf(got_path, sizeof(got_path), "%s/%s.f32", DUMP, name);
    snprintf(ref_path, sizeof(ref_path), "%s/%s.f32", FIXTURES, name);
    float *got = load_f32(got_path, count);
    float *ref = load_f32(ref_path, count);
    if (!got || !ref) {
        printf("  %-22s MISSING (%s / %s)\n", name, got ? "got" : "got?",
               ref ? "ref" : "ref?");
        free(got);
        free(ref);
        failures++;
        return;
    }
    double error = rel_l2(got, ref, count);
    int ok = error <= gate;
    if (!ok) failures++;
    printf("  %-22s relL2 %.3e  %s（门 %.0e）\n", name, error,
           ok ? "PASS" : "FAIL", gate);
    free(got);
    free(ref);
}

int main(void) {
    char error[512] = {0};
    mkdir(DUMP, 0755);
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu: %s\n", error);
        return 1;
    }
    qw_vision *v = qw_vision_create(MODEL_TEXT "/..", gpu, error, sizeof(error));
    if (!v) {
        printf("create: %s\n", error);
        return 1;
    }
    qw_vision_set_dump(v, DUMP);

    const size_t seq = 704, merged = 176;
    float *input = load_f32(FIXTURES "/vision_input.f32", seq * 1536);
    if (!input) {
        printf("input fixture missing\n");
        return 1;
    }
    uint32_t grid[3] = {1, 22, 32};
    float *merged_out = malloc(sizeof(float) * merged * 4096);
    float *deep_out = malloc(sizeof(float) * 3 * merged * 4096);
    if (!qw_vision_forward(v, input, grid, 1, merged_out, deep_out, error,
                           sizeof(error))) {
        printf("forward: %s\n", error);
        return 1;
    }

    /* 门限 = fmax(3e-2, py-bf16 vs py-fp32 本征带 × 1.5)，P3 文本塔同款判据。
     * 本征带随深度累积放大（block26 ≈ 3.9e-2、merger ≈ 4.9e-2，
     * tests/gen_vision_fixture 同输入实测），C-fp32 实现落带内即正确。 */
    static const double block_gate[27] = {
        3e-2, 3e-2, 3e-2, 3e-2, 3e-2, 3e-2, 3e-2, 3e-2,   /* 0-7 */
        3e-2, 3e-2, 3e-2, 3e-2, 3e-2, 3e-2, 3e-2, 3e-2,   /* 8-15 */
        3.3e-2, 3.3e-2, 3.3e-2, 3.3e-2, 3.3e-2, 3.3e-2,   /* 16-21 */
        3.3e-2, 3.3e-2, 3.3e-2, 5.7e-2, 5.9e-2,           /* 22-26 */
    };
    printf("视觉塔逐段验收（relL2，bf16 参考噪声带内为 PASS）:\n");
    check("vision_patch_embed", seq * 1152, 2e-3);
    check("vision_rope_cos", seq * 72, 1e-4);
    check("vision_rope_sin", seq * 72, 1e-4);
    for (int i = 0; i < 27; i++) {
        char name[32];
        snprintf(name, sizeof(name), "vision_block%02d", i);
        check(name, seq * 1152, block_gate[i]);
    }
    check("vision_merger", merged * 4096, 7.4e-2);
    for (int i = 0; i < 3; i++) {
        char name[32];
        snprintf(name, sizeof(name), "vision_deepstack%d", i);
        check(name, merged * 4096, i == 0 ? 3e-2 : i == 1 ? 3.8e-2 : 5.0e-2);
    }
    printf(failures ? "\n%d FAIL\n" : "\nALL PASS\n", failures);
    return failures ? 1 : 0;
}
