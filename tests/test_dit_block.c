#include "qwen_dit.h"

#include "qwen_gpu_ext.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_DIR "/Volumes/Work/models/Qwen-Image-2.1.models"
#define FIXTURE_DIR "tests/dit_fixture"

static int failures = 0;

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            failures++;                                                     \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                     \
            printf(__VA_ARGS__);                                            \
            printf("\n");                                                   \
        }                                                                   \
    } while (0)

static uint16_t *read_bf16(const char *name, size_t *count) {
    char path[512];
    snprintf(path, sizeof(path), FIXTURE_DIR "/%s", name);
    FILE *file = fopen(path, "rb");
    if (!file) {
        printf("missing fixture %s\n", path);
        failures++;
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long bytes = ftell(file);
    rewind(file);
    *count = (size_t)bytes / 2;
    uint16_t *values = malloc((size_t)bytes);
    if (fread(values, 2, *count, file) != *count) {
        printf("short read %s\n", path);
        failures++;
        free(values);
        values = NULL;
    }
    fclose(file);
    return values;
}

static float *read_f32(const char *name, size_t *count) {
    char path[512];
    snprintf(path, sizeof(path), FIXTURE_DIR "/%s", name);
    FILE *file = fopen(path, "rb");
    if (!file) {
        printf("missing fixture %s\n", path);
        failures++;
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long bytes = ftell(file);
    rewind(file);
    *count = (size_t)bytes / 4;
    float *values = malloc((size_t)bytes);
    if (fread(values, 4, *count, file) != *count) {
        printf("short read %s\n", path);
        failures++;
        free(values);
        values = NULL;
    }
    fclose(file);
    return values;
}

static float bits_to_f32(uint16_t bits) {
    uint32_t expanded = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &expanded, sizeof(value));
    return value;
}

/* Prints rel L2 / max abs between a computed host BF16 buffer and the
 * reference dump; returns the rel L2 value. */
static double compare_stage(const char *label, const uint16_t *computed,
                            const uint16_t *reference, size_t count) {
    double sum = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < count; i++) {
        double a = bits_to_f32(computed[i]);
        double b = bits_to_f32(reference[i]);
        double diff = a - b;
        sum += diff * diff;
        if (fabs(diff) > max_abs) max_abs = fabs(diff);
    }
    double rel = 0.0;
    for (size_t i = 0; i < count; i++) {
        double b = bits_to_f32(reference[i]);
        rel += b * b;
    }
    rel = sqrt(sum / (rel > 1e-12 ? rel : 1e-12));
    printf("  %-14s relL2 %.3g  maxAbs %.3g\n", label, rel, max_abs);
    return rel;
}

static void run_case(h3_gpu *gpu, qw_dit *dit, const char *name,
                     uint32_t text_rows, uint32_t image_rows, char *error,
                     size_t error_size) {
    uint32_t rows = text_rows + image_rows;
    size_t hidden_elements = (size_t)rows * QWEN_HIDDEN;
    printf("case %s (rows=%u text=%u):\n", name, rows, text_rows);

    char file[64];
    size_t count = 0;

    snprintf(file, sizeof(file), "%s_text_in.bin", name);
    uint16_t *text_host = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_latents.bin", name);
    uint16_t *latents_host = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_rope_cos.f32", name);
    float *cos_host = read_f32(file, &count);
    snprintf(file, sizeof(file), "%s_rope_sin.f32", name);
    float *sin_host = read_f32(file, &count);
    snprintf(file, sizeof(file), "%s_joint_in.bin", name);
    uint16_t *joint_ref = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_temb.bin", name);
    uint16_t *temb_ref = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_modulation.bin", name);
    uint16_t *modulation_ref = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_attn_in.bin", name);
    uint16_t *attn_in_ref = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_block_out.bin", name);
    uint16_t *block_ref = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_output.bin", name);
    uint16_t *output_ref = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_attn_merged.bin", name);
    uint16_t *attn_merged_ref = read_bf16(file, &count);
    if (!text_host || !latents_host || !cos_host || !sin_host || !joint_ref ||
        !temb_ref || !modulation_ref || !attn_in_ref || !block_ref ||
        !output_ref || !attn_merged_ref) {
        return;
    }

    h3_gpu_tensor *text_t = text_rows
        ? h3_gpu_tensor_from_bf16(gpu, text_host,
                                  (size_t)text_rows * QWEN_HIDDEN)
        : NULL;
    h3_gpu_tensor *latents_t = h3_gpu_tensor_from_bf16(
        gpu, latents_host, (size_t)image_rows * QWEN_Z_DIM);
    h3_gpu_tensor *joint = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *cos_t = h3_gpu_tensor_from_f32(
        gpu, cos_host, (size_t)rows * (QWEN_HEAD_DIM / 2));
    h3_gpu_tensor *sin_t = h3_gpu_tensor_from_f32(
        gpu, sin_host, (size_t)rows * (QWEN_HEAD_DIM / 2));

    uint32_t *row_map_host = malloc(sizeof(uint32_t) * rows);
    for (uint32_t i = 0; i < rows; i++) {
        row_map_host[i] = i < text_rows ? 1u : 0u;
    }
    h3_gpu_tensor *row_map = h3_gpu_tensor_from_u32(gpu, row_map_host, rows);

    float sigmas[2] = {0.37f, 0.0f};
    float features[512];
    qw_dit_timestep_features(sigmas, 2, features);
    h3_gpu_tensor *features_t = h3_gpu_tensor_from_f32(gpu, features, 512);
    h3_gpu_tensor *temb = h3_gpu_tensor_new_bf16(gpu, 2 * QWEN_HIDDEN);
    h3_gpu_tensor *modulation = h3_gpu_tensor_new_bf16(gpu, 2 * 4 * QWEN_HIDDEN);
    h3_gpu_tensor *norm_scale = h3_gpu_tensor_new_bf16(gpu, 2 * QWEN_HIDDEN);

    if ((text_rows && !text_t) || !latents_t || !joint || !cos_t || !sin_t ||
        !row_map || !features_t || !temb || !modulation || !norm_scale) {
        printf("allocation failure\n");
        failures++;
        return;
    }

    CHECK(h3_gpu_begin(gpu), "begin: %s", h3_gpu_error(gpu));
    CHECK(qw_dit_embed(dit, text_t, text_rows, latents_t, image_rows, joint,
                       error, error_size),
          "embed: %s", error);
    CHECK(qw_dit_conditioning(dit, features_t, 2, temb, modulation,
                              norm_scale, error, error_size),
          "conditioning: %s", error);
    CHECK(h3_gpu_submit(gpu), "submit: %s", h3_gpu_error(gpu));

    uint16_t *host = malloc(sizeof(uint16_t) * 2 * 4 * QWEN_HIDDEN);
    h3_gpu_tensor_read_bf16(joint, host, hidden_elements);
    double rel = compare_stage("embed", host, joint_ref, hidden_elements);
    CHECK(rel < 2e-2, "embed relL2 %.3g", rel);
    h3_gpu_tensor_read_bf16(temb, host, 2 * QWEN_HIDDEN);
    rel = compare_stage("temb", host, temb_ref, 2 * QWEN_HIDDEN);
    CHECK(rel < 2e-2, "temb relL2 %.3g", rel);
    h3_gpu_tensor_read_bf16(modulation, host, 2 * 4 * QWEN_HIDDEN);
    for (int slot = 0; slot < 4; slot++) {
        printf("    slot%d computed:", slot);
        for (int j = 0; j < 4; j++)
            printf(" %.4f", bits_to_f32(host[slot * QWEN_HIDDEN + j]));
        printf("  reference:");
        for (int j = 0; j < 4; j++)
            printf(" %.4f",
                   bits_to_f32(modulation_ref[slot * QWEN_HIDDEN + j]));
        printf("\n");
    }
    rel = compare_stage("modulation", host, modulation_ref,
                        2 * 4 * QWEN_HIDDEN);
    CHECK(rel < 2e-2, "modulation relL2 %.3g", rel);

    /* Norm + modulate + row_map stage against the fixture attn input. */
    h3_gpu_tensor *modulated = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    CHECK(h3_gpu_begin(gpu), "begin: %s", h3_gpu_error(gpu));
    CHECK(qw_gpu_layer_norm_mod_bf16(gpu, modulated, joint, modulation,
                                     row_map, rows, QWEN_HIDDEN, 4, 0,
                                     1e-6f),
          "norm_mod: %s", h3_gpu_error(gpu));
    CHECK(h3_gpu_submit(gpu), "submit: %s", h3_gpu_error(gpu));
    h3_gpu_tensor_read_bf16(modulated, host, hidden_elements);
    rel = compare_stage("norm_mod", host, attn_in_ref, hidden_elements);
    CHECK(rel < 2e-2, "norm_mod relL2 %.3g", rel);
    h3_gpu_tensor_free(modulated);

    /* Full block + final head. */
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * QWEN_Z_DIM);
    h3_gpu_tensor *debug_attn = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    CHECK(h3_gpu_begin(gpu), "begin: %s", h3_gpu_error(gpu));
    CHECK(h3_gpu_copy_bf16(gpu, hidden, 0, joint, 0, hidden_elements),
          "copy");
    CHECK(qw_dit_block(dit, 0, hidden, modulation, cos_t, sin_t, row_map,
                       rows, text_rows, NULL, NULL, 0, NULL, NULL,
                       debug_attn, error, error_size),
          "block: %s", error);
    CHECK(qw_dit_final(dit, hidden, norm_scale, row_map, rows, output, error,
                       error_size),
          "final: %s", error);
    CHECK(h3_gpu_submit(gpu), "submit: %s", h3_gpu_error(gpu));

    h3_gpu_tensor_read_bf16(debug_attn, host, hidden_elements);
    rel = compare_stage("attn_merged", host, attn_merged_ref,
                        hidden_elements);
    CHECK(rel < 3e-2, "attn_merged relL2 %.3g", rel);
    h3_gpu_tensor_read_bf16(hidden, host, hidden_elements);
    rel = compare_stage("block", host, block_ref, hidden_elements);
    CHECK(rel < 3e-2, "block relL2 %.3g", rel);
    h3_gpu_tensor_read_bf16(output, host, (size_t)rows * QWEN_Z_DIM);
    rel = compare_stage("output", host, output_ref,
                        (size_t)rows * QWEN_Z_DIM);
    CHECK(rel < 3e-2, "output relL2 %.3g", rel);

    free(host);
    free(row_map_host);
    free(text_host);
    free(latents_host);
    free(cos_host);
    free(sin_host);
    free(joint_ref);
    free(temb_ref);
    free(modulation_ref);
    free(attn_in_ref);
    free(block_ref);
    free(output_ref);
    free(attn_merged_ref);
    h3_gpu_tensor_free(text_t);
    h3_gpu_tensor_free(latents_t);
    h3_gpu_tensor_free(joint);
    h3_gpu_tensor_free(cos_t);
    h3_gpu_tensor_free(sin_t);
    h3_gpu_tensor_free(row_map);
    h3_gpu_tensor_free(features_t);
    h3_gpu_tensor_free(temb);
    h3_gpu_tensor_free(modulation);
    h3_gpu_tensor_free(norm_scale);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(debug_attn);
}

static void run_full(h3_gpu *gpu, qw_dit *dit, const char *name,
                     uint32_t text_rows, uint32_t image_rows, char *error,
                     size_t error_size) {
    uint32_t rows = text_rows + image_rows;
    size_t hidden_elements = (size_t)rows * QWEN_HIDDEN;
    printf("full %s (rows=%u text=%u):\n", name, rows, text_rows);

    char file[64];
    size_t count = 0;
    snprintf(file, sizeof(file), "%s_text_in.bin", name);
    uint16_t *text_host = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_latents.bin", name);
    uint16_t *latents_host = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_rope_cos.f32", name);
    float *cos_host = read_f32(file, &count);
    snprintf(file, sizeof(file), "%s_rope_sin.f32", name);
    float *sin_host = read_f32(file, &count);
    snprintf(file, sizeof(file), "%s_full_output.bin", name);
    uint16_t *output_ref = read_bf16(file, &count);
    snprintf(file, sizeof(file), "%s_full_output_f32.bin", name);
    uint16_t *output_f32_ref = read_bf16(file, &count);
    if (!text_host || !latents_host || !cos_host || !sin_host || !output_ref ||
        !output_f32_ref) {
        return;
    }

    h3_gpu_tensor *text_t = text_rows
        ? h3_gpu_tensor_from_bf16(gpu, text_host,
                                  (size_t)text_rows * QWEN_HIDDEN)
        : NULL;
    h3_gpu_tensor *latents_t = h3_gpu_tensor_from_bf16(
        gpu, latents_host, (size_t)image_rows * QWEN_Z_DIM);
    h3_gpu_tensor *cos_t = h3_gpu_tensor_from_f32(
        gpu, cos_host, (size_t)rows * (QWEN_HEAD_DIM / 2));
    h3_gpu_tensor *sin_t = h3_gpu_tensor_from_f32(
        gpu, sin_host, (size_t)rows * (QWEN_HEAD_DIM / 2));
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * QWEN_Z_DIM);
    if ((text_rows && !text_t) || !latents_t || !cos_t || !sin_t || !output) {
        printf("allocation failure\n");
        failures++;
        return;
    }

    CHECK(h3_gpu_begin(gpu), "begin: %s", h3_gpu_error(gpu));
    CHECK(qw_dit_forward(dit, text_t, text_rows, latents_t, image_rows, 0.37f,
                         cos_t, sin_t, output, error, error_size),
          "forward: %s", error);
    CHECK(h3_gpu_submit(gpu), "submit: %s", h3_gpu_error(gpu));

    uint16_t *host = malloc(sizeof(uint16_t) * (size_t)rows * QWEN_Z_DIM);
    h3_gpu_tensor_read_bf16(output, host, (size_t)rows * QWEN_Z_DIM);
    double rel = compare_stage("vs bf16 ref", host, output_ref,
                               (size_t)rows * QWEN_Z_DIM);
    compare_stage("vs f32 yardstick", host, output_f32_ref,
                  (size_t)rows * QWEN_Z_DIM);
    double intrinsic = compare_stage("py bf16 vs f32", output_ref,
                                     output_f32_ref,
                                     (size_t)rows * QWEN_Z_DIM);
    /* The 32-layer bf16 path is intrinsically sensitive: the reference's
     * own distance from the fp32 wiring is the yardstick. The Metal replay
     * must simply sit inside that noise band. */
    CHECK(rel < fmax(3e-2, intrinsic),
          "full output relL2 %.3g (intrinsic %.3g)", rel, intrinsic);

    free(host);
    free(text_host);
    free(latents_host);
    free(cos_host);
    free(sin_host);
    free(output_ref);
    free(output_f32_ref);
    h3_gpu_tensor_free(text_t);
    h3_gpu_tensor_free(latents_t);
    h3_gpu_tensor_free(cos_t);
    h3_gpu_tensor_free(sin_t);
    h3_gpu_tensor_free(output);
}

static void run_ti2i(h3_gpu *gpu, qw_dit *dit, char *error,
                     size_t error_size) {
    uint32_t t0 = 20, cond_tokens = 48, t1 = 17, target_tokens = 64;
    uint32_t vlm_rows = t0 + cond_tokens / 4 + t1;
    uint32_t latents_rows = cond_tokens + target_tokens;
    uint32_t rows = t0 + cond_tokens + t1 + target_tokens;
    printf("full ti2i (rows=%u vlm=%u):\n", rows, vlm_rows);

    char file[64];
    size_t count = 0;
    uint16_t *text_host = read_bf16("c_text_in.bin", &count);
    uint16_t *latents_host = read_bf16("c_latents.bin", &count);
    float *cos_host = read_f32("c_rope_cos.f32", &count);
    float *sin_host = read_f32("c_rope_sin.f32", &count);
    uint16_t *output_ref = read_bf16("c_full_output.bin", &count);
    uint16_t *output_f32_ref = read_bf16("c_full_output_f32.bin", &count);
    if (!text_host || !latents_host || !cos_host || !sin_host || !output_ref ||
        !output_f32_ref) {
        printf("ti2i fixture missing\n");
        failures++;
        return;
    }

    h3_gpu_tensor *text_t = h3_gpu_tensor_from_bf16(
        gpu, text_host, (size_t)vlm_rows * QWEN_HIDDEN);
    h3_gpu_tensor *latents_t = h3_gpu_tensor_from_bf16(
        gpu, latents_host, (size_t)latents_rows * QWEN_Z_DIM);
    h3_gpu_tensor *cos_t = h3_gpu_tensor_from_f32(
        gpu, cos_host, (size_t)rows * (QWEN_HEAD_DIM / 2));
    h3_gpu_tensor *sin_t = h3_gpu_tensor_from_f32(
        gpu, sin_host, (size_t)rows * (QWEN_HEAD_DIM / 2));
    h3_gpu_tensor *output =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * QWEN_Z_DIM);
    if (!text_t || !latents_t || !cos_t || !sin_t || !output) {
        printf("allocation failure\n");
        failures++;
        return;
    }

    qw_dit_segment segs[4] = {
        {t0, 0, 0},
        {cond_tokens, 1, 0},
        {t1, 0, t0 + cond_tokens / 4},
        {target_tokens, 1, cond_tokens},
    };
    CHECK(h3_gpu_begin(gpu), "begin: %s", h3_gpu_error(gpu));
    CHECK(qw_dit_forward_ti2i(dit, text_t, latents_t, segs, 4, 0, cos_t,
                              sin_t, 0.37f, output, error, error_size),
          "ti2i forward: %s", error);
    CHECK(h3_gpu_submit(gpu), "submit: %s", h3_gpu_error(gpu));

    uint16_t *host = malloc(sizeof(uint16_t) * (size_t)rows * QWEN_Z_DIM);
    h3_gpu_tensor_read_bf16(output, host, (size_t)rows * QWEN_Z_DIM);
    double rel = compare_stage("vs bf16 ref", host, output_ref,
                               (size_t)rows * QWEN_Z_DIM);
    compare_stage("vs f32 yardstick", host, output_f32_ref,
                  (size_t)rows * QWEN_Z_DIM);
    double intrinsic = compare_stage("py bf16 vs f32", output_ref,
                                     output_f32_ref,
                                     (size_t)rows * QWEN_Z_DIM);
    CHECK(rel < fmax(3e-2, intrinsic),
          "ti2i output relL2 %.3g (intrinsic %.3g)", rel, intrinsic);

    free(host);
    free(text_host);
    free(latents_host);
    free(cos_host);
    free(sin_host);
    free(output_ref);
    free(output_f32_ref);
    h3_gpu_tensor_free(text_t);
    h3_gpu_tensor_free(latents_t);
    h3_gpu_tensor_free(cos_t);
    h3_gpu_tensor_free(sin_t);
    h3_gpu_tensor_free(output);
}

int main(void) {
    char error[1024] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu create failed: %s\n", error);
        return 1;
    }
    qw_dit *dit = qw_dit_create(gpu, MODEL_DIR, 32, error, sizeof(error));
    if (!dit) {
        printf("dit create failed: %s\n", error);
        h3_gpu_free(gpu);
        return 1;
    }

    run_case(gpu, dit, "a", 0, 16, error, sizeof(error));
    run_case(gpu, dit, "b", 8, 12, error, sizeof(error));
    run_full(gpu, dit, "a", 0, 16, error, sizeof(error));
    run_full(gpu, dit, "b", 8, 12, error, sizeof(error));
    run_ti2i(gpu, dit, error, sizeof(error));

    qw_dit_free(dit);
    h3_gpu_free(gpu);
    if (failures == 0) {
        printf("dit parity passed (single block + full 32-layer)\n");
        return 0;
    }
    printf("%d failure(s)\n", failures);
    return 1;
}
