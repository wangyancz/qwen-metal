// VAE decoder parity: staged comparison against the diffusers golden
// fixture (tests/vae_fixture, fp32). The C implementation keeps
// activations channels-last [H, W, C]; the fixture is NCHW [C, H, W]
// (temporal axis folded), so comparisons permute on the fly.

#include "h3_gpu.h"
#include "qwen_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define FIXTURE_DIR "tests/vae_fixture"
#define MODEL_DIR "/Volumes/Work/models/Qwen-Image-2.1.models"

static int failures = 0;

static float *read_f32(const char *path, size_t *count) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long bytes = ftell(file);
    fseek(file, 0, SEEK_SET);
    *count = (size_t)bytes / sizeof(float);
    float *data = malloc((size_t)bytes);
    if (!data || fread(data, sizeof(float), *count, file) != *count) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    return data;
}

/* relL2 and max abs between the channels-last dump and the NCHW fixture. */
static void compare_stage(const char *case_name, const char *fixture_key,
                          const char *dump_name, uint32_t channels,
                          uint32_t height, uint32_t width) {
    char path[512];
    snprintf(path, sizeof(path), FIXTURE_DIR "/%s_%s.bin", case_name,
             fixture_key);
    size_t ref_count = 0;
    float *ref = read_f32(path, &ref_count);
    snprintf(path, sizeof(path), FIXTURE_DIR "/cdump/%s/%s.bin", case_name,
             dump_name);
    size_t got_count = 0;
    float *got = read_f32(path, &got_count);

    size_t expect = (size_t)channels * height * width;
    if (!ref || !got || ref_count != expect || got_count != expect) {
        printf("  %-44s MISSING or size mismatch (ref %zu got %zu expect %zu)\n",
               fixture_key, ref_count, got_count, expect);
        failures++;
        free(ref);
        free(got);
        return;
    }
    double sum = 0.0;
    double ref_sum = 0.0;
    float max_abs = 0.0f;
    for (uint32_t h = 0; h < height; h++) {
        for (uint32_t w = 0; w < width; w++) {
            for (uint32_t c = 0; c < channels; c++) {
                float r = ref[((size_t)c * height + h) * width + w];
                float g = got[((size_t)h * width + w) * channels + c];
                double d = (double)g - (double)r;
                sum += d * d;
                ref_sum += (double)r * (double)r;
                float abs = fabsf(g - r);
                if (abs > max_abs) max_abs = abs;
            }
        }
    }
    double rel = sqrt(sum) / sqrt(ref_sum + 1e-30);
    printf("  %-44s relL2 %-10.3g maxAbs %.4g\n", fixture_key, rel, max_abs);
    if (!(rel < 2e-3)) {
        printf("FAIL tests/test_vae.c: stage %s relL2 %.3g\n", fixture_key,
               rel);
        failures++;
    }
    free(ref);
    free(got);
}

static void compare_exact_layout(const char *case_name, const char *key,
                                 const float *got, size_t count) {
    char path[512];
    snprintf(path, sizeof(path), FIXTURE_DIR "/%s_%s.bin", key, case_name);
    size_t ref_count = 0;
    float *ref = read_f32(path, &ref_count);
    if (!ref || ref_count != count) {
        printf("  %-44s MISSING or size mismatch\n", key);
        failures++;
        free(ref);
        return;
    }
    double sum = 0.0;
    double ref_sum = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; i++) {
        double d = (double)got[i] - (double)ref[i];
        sum += d * d;
        ref_sum += (double)ref[i] * (double)ref[i];
        float abs = fabsf(got[i] - ref[i]);
        if (abs > max_abs) max_abs = abs;
    }
    double rel = sqrt(sum) / sqrt(ref_sum + 1e-30);
    printf("  %-44s relL2 %-10.3g maxAbs %.4g\n", key, rel, max_abs);
    if (!(rel < 2e-3)) {
        printf("FAIL tests/test_vae.c: stage %s relL2 %.3g\n", key, rel);
        failures++;
    }
    free(ref);
}

static void run_case(h3_gpu *gpu, qw_vae *vae, const char *case_name,
                     uint32_t side) {
    char error[1024];
    char path[512];
    char dump_dir[256];
    snprintf(dump_dir, sizeof(dump_dir), FIXTURE_DIR "/cdump/%s", case_name);
    mkdir(FIXTURE_DIR "/cdump", 0755);
    mkdir(dump_dir, 0755);
    setenv("QWEN_VAE_DUMP", dump_dir, 1);

    size_t z_count = (size_t)64 * side * side;
    snprintf(path, sizeof(path), FIXTURE_DIR "/z_%s.bin", case_name);
    size_t z_file_count = 0;
    float *z_host = read_f32(path, &z_file_count);
    if (!z_host || z_file_count != z_count) {
        printf("case %s: missing latent fixture\n", case_name);
        failures++;
        free(z_host);
        return;
    }
    h3_gpu_tensor *z = h3_gpu_tensor_new_f32(gpu, z_count);
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(
        gpu, (size_t)4 * (side * 16) * (side * 16));
    if (!z || !out || !h3_gpu_tensor_write_f32(z, z_host, z_count)) {
        printf("case %s: tensor setup failed\n", case_name);
        failures++;
        h3_gpu_tensor_free(z);
        h3_gpu_tensor_free(out);
        free(z_host);
        return;
    }
    if (!qw_vae_decode(vae, z, side, side, out, error, sizeof(error))) {
        printf("case %s: decode failed: %s\n", case_name, error);
        failures++;
        h3_gpu_tensor_free(z);
        h3_gpu_tensor_free(out);
        free(z_host);
        return;
    }
    if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) {
        printf("case %s: final sync failed: %s\n", case_name,
               h3_gpu_error(gpu));
        failures++;
        h3_gpu_tensor_free(z);
        h3_gpu_tensor_free(out);
        free(z_host);
        return;
    }

    // Final image: both sides are NCHW [4, 16s, 16s] — direct compare.
    size_t out_count = (size_t)4 * (side * 16) * (side * 16);
    float *out_host = malloc(sizeof(float) * out_count);
    if (!out_host ||
        !h3_gpu_tensor_read_f32(out, out_host, out_count)) {
        printf("case %s: output read failed\n", case_name);
        failures++;
    } else {
        compare_exact_layout(case_name, "frame0", out_host, out_count);
    }
    free(out_host);
    h3_gpu_tensor_free(z);
    h3_gpu_tensor_free(out);
    free(z_host);

    printf("  staged parity (case %s):\n", case_name);
    uint32_t os = side * 16;
    compare_stage(case_name, "m900_post_quant_conv", "m900_post_quant_conv",
                  64, side, side);
    compare_stage(case_name, "m001_conv_in", "m001_conv_in", 1152, side,
                  side);
    // Mid block.
    compare_stage(case_name, "m011_mid_block_resnets_0_conv_shortcut",
                  "mid_r0_shortcut", 1152, side, side);
    compare_stage(case_name, "m006_mid_block_resnets_0_norm1",
                  "mid_r0_norm1", 1152, side, side);
    compare_stage(case_name, "m007_mid_block_resnets_0_conv1",
                  "mid_r0_conv1", 1152, side, side);
    compare_stage(case_name, "m010_mid_block_resnets_0_conv2",
                  "mid_r0_conv2", 1152, side, side);
    compare_stage(case_name, "m002_mid_block_attentions_0_norm",
                  "attn_norm", 1152, side, side);
    compare_stage(case_name, "m003_mid_block_attentions_0_to_qkv",
                  "attn_qkv", 3456, side, side);
    compare_stage(case_name, "m004_mid_block_attentions_0_proj",
                  "attn_proj", 1152, side, side);
    compare_stage(case_name, "m017_mid_block_resnets_1_conv2",
                  "mid_r1_conv2", 1152, side, side);
    // Up block 0 (1152 -> 1152, 2x spatial).
    compare_stage(case_name, "m019_up_blocks_0_avg_shortcut", "up0_shortcut",
                  1152, os / 8, os / 8);
    compare_stage(case_name, "m025_up_blocks_0_resnets_0_conv2", "up0_r0_conv2",
                  1152, side, side);
    compare_stage(case_name, "m039_up_blocks_0_resnets_2_conv2", "up0_r2_conv2",
                  1152, side, side);
    compare_stage(case_name, "m041_up_blocks_0_upsampler_resample_0",
                  "up0_nearest", 1152, os / 8, os / 8);
    compare_stage(case_name, "m042_up_blocks_0_upsampler_resample_1",
                  "up0_resample", 1152, os / 8, os / 8);
    // Up block 1 (1152 -> 1152).
    compare_stage(case_name, "m044_up_blocks_1_avg_shortcut", "up1_shortcut",
                  1152, os / 4, os / 4);
    compare_stage(case_name, "m064_up_blocks_1_resnets_2_conv2", "up1_r2_conv2",
                  1152, os / 8, os / 8);
    compare_stage(case_name, "m067_up_blocks_1_upsampler_resample_1",
                  "up1_resample", 1152, os / 4, os / 4);
    // Up block 2 (1152 -> 576).
    compare_stage(case_name, "m069_up_blocks_2_avg_shortcut", "up2_shortcut",
                  576, os / 2, os / 2);
    compare_stage(case_name, "m076_up_blocks_2_resnets_0_conv_shortcut",
                  "up2_r0_shortcut", 576, os / 4, os / 4);
    compare_stage(case_name, "m089_up_blocks_2_resnets_2_conv2", "up2_r2_conv2",
                  576, os / 4, os / 4);
    compare_stage(case_name, "m092_up_blocks_2_upsampler_resample_1",
                  "up2_resample", 576, os / 2, os / 2);
    // Up block 3 (576 -> 288, spatial-only upsampler).
    compare_stage(case_name, "m094_up_blocks_3_avg_shortcut", "up3_shortcut",
                  288, os, os);
    compare_stage(case_name, "m101_up_blocks_3_resnets_0_conv_shortcut",
                  "up3_r0_shortcut", 288, os / 2, os / 2);
    compare_stage(case_name, "m114_up_blocks_3_resnets_2_conv2", "up3_r2_conv2",
                  288, os / 2, os / 2);
    compare_stage(case_name, "m117_up_blocks_3_upsampler_resample_1",
                  "up3_resample", 288, os, os);
    // Up block 4 (288 -> 144, no upsampler).
    compare_stage(case_name, "m124_up_blocks_4_resnets_0_conv_shortcut",
                  "up4_r0_shortcut", 144, os, os);
    compare_stage(case_name, "m137_up_blocks_4_resnets_2_conv2", "up4_r2_conv2",
                  144, os, os);
    compare_stage(case_name, "m139_norm_out", "m139_norm_out", 144, os, os);
    compare_stage(case_name, "m140_conv_out", "m140_conv_out", 4, os, os);
}

/* ti2i encoder parity: fixture input [1, 4, 1, 128, 128] NCHW; the C
 * encoder consumes channels-last. mu rows (64ch, mean/std normalized)
 * compare against enc_latents.bin row-major over [h][w]. */
static void run_encoder(h3_gpu *gpu, qw_vae *vae, char *error,
                        size_t error_size) {
    printf("encoder (128x128 RGBA):\n");
    setenv("QWEN_VAE_DUMP", "tests/vae_cdump", 1);
    size_t count = 0;
    float *input = read_f32("tests/vae_fixture/enc_input.bin", &count);
    float *ref = read_f32("tests/vae_fixture/enc_latents.bin", &count);
    if (!input || !ref || count != (size_t)64 * 8 * 8) {
        printf("encoder fixture missing\n");
        failures++;
        free(input);
        free(ref);
        return;
    }
    uint32_t h = 128, w = 128;
    /* NCHW [1,4,1,32,32] -> HWC [32,32,4] */
    float *hwc = malloc(sizeof(float) * (size_t)h * w * 4);
    for (uint32_t c = 0; c < 4; c++)
        for (uint32_t y = 0; y < h; y++)
            for (uint32_t x = 0; x < w; x++)
                hwc[((size_t)y * w + x) * 4 + c] =
                    input[((size_t)c * h + y) * w + x];
    h3_gpu_tensor *in_t =
        h3_gpu_tensor_new_f32(gpu, (size_t)h * w * 4);
    h3_gpu_tensor *lat =
        h3_gpu_tensor_new_bf16(gpu, (size_t)(h / 16) * (w / 16) * 64);
    if (!in_t || !lat) {
        printf("allocation failure\n");
        failures++;
        free(input);
        free(ref);
        free(hwc);
        h3_gpu_tensor_free(in_t);
        h3_gpu_tensor_free(lat);
        return;
    }
    if (!h3_gpu_tensor_write_f32(in_t, hwc, (size_t)h * w * 4)) {
        printf("input copy failed\n");
        failures++;
    }
    if (!qw_vae_encode(vae, in_t, h, w, lat, error, error_size)) {
        printf("encode: %s\n", error);
        failures++;
    }
    if (!h3_gpu_submit(gpu)) {
        printf("submit: %s\n", h3_gpu_error(gpu));
        failures++;
    }
    uint16_t *host =
        malloc(sizeof(uint16_t) * (size_t)(h / 16) * (w / 16) * 64);
    h3_gpu_tensor_read_bf16(lat, host, (size_t)(h / 16) * (w / 16) * 64);
    /* packed rows -> [h][w][c] f32 */
    uint32_t lh = h / 16, lw = w / 16;
    size_t ln = (size_t)64 * lh * lw;
    float *got = malloc(sizeof(float) * ln);
    for (uint32_t y = 0; y < lh; y++)
        for (uint32_t x = 0; x < lw; x++)
            for (uint32_t c = 0; c < 64; c++) {
                uint32_t bits =
                    (uint32_t)host[((size_t)y * lw + x) * 64 + c] << 16;
                memcpy(&got[((size_t)y * lw + x) * 64 + c], &bits, 4);
            }
    /* fixture ref [1,64,1,lh,lw] NCHW -> [h][w][c] */
    float *ref_hwc = malloc(sizeof(float) * ln);
    for (uint32_t c = 0; c < 64; c++)
        for (uint32_t y = 0; y < lh; y++)
            for (uint32_t x = 0; x < lw; x++)
                ref_hwc[((size_t)y * lw + x) * 64 + c] =
                    ref[((size_t)c * lh + y) * lw + x];
    double num = 0, den = 0;
    for (size_t i = 0; i < (size_t)64 * 2 * 2; i++) {
        double diff = (double)got[i] - ref_hwc[i];
        num += diff * diff;
        den += (double)ref_hwc[i] * ref_hwc[i];
    }
    double rel = den > 0 ? sqrt(num / den) : 0.0;
    int ok = rel <= 2e-3;
    if (!ok) failures++;
    printf("  enc_latents         relL2 %.3e  %s（门 2e-03）\n", rel,
           ok ? "PASS" : "FAIL");
    free(input);
    free(ref);
    free(hwc);
    free(got);
    free(ref_hwc);
    free(host);
    h3_gpu_tensor_free(in_t);
    h3_gpu_tensor_free(lat);
}

int main(void) {
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu create failed: %s\n", error);
        return 1;
    }
    qw_vae *vae = qw_vae_create(gpu, MODEL_DIR, error, sizeof(error));
    if (!vae) {
        printf("vae create failed: %s\n", error);
        h3_gpu_free(gpu);
        return 1;
    }
    if (!h3_gpu_begin(gpu)) {
        printf("begin failed: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    run_case(gpu, vae, "a", 16);
    run_case(gpu, vae, "b", 32);
    run_encoder(gpu, vae, error, sizeof(error));
    qw_vae_free(vae);
    h3_gpu_free(gpu);
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("vae parity passed\n");
    return 0;
}
