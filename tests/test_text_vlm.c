/* ti2i 文本塔验收（V1b）：C 端 encode_vlm（mrope + DeepStack 注入）对比
 * tests/gen_text_vlm_fixture.py 的 diffusers Qwen3VLTextModel 真值。
 * 门限按 P3 判据 fmax(3e-2, 本征带×1.5)；rope 表应逐位一致。 */
#include "qwen_text_encoder.h"

#include "qwen_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MODEL_DIR "/Volumes/Work/models/Qwen-Image-2.1.models"
#define FIXTURE_DIR "tests/vision_fixture"
#define DUMP_DIR "tests/text_vlm_dump"
#define ROPE_PAIRS (QWEN_TEXT_HEAD_DIM / 2)

static int failures = 0;

static void *read_file(const char *path, size_t *bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    void *data = malloc((size_t)size);
    if (!data || fread(data, 1, (size_t)size, file) != (size_t)size) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *bytes = (size_t)size;
    return data;
}

static float bits_to_f32(uint16_t bits) {
    uint32_t expanded = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &expanded, sizeof(value));
    return value;
}

/* computed bf16 vs reference f32 */
static double rel_l2(const uint16_t *computed, const float *reference,
                     size_t count) {
    double num = 0, den = 0;
    for (size_t i = 0; i < count; i++) {
        double diff = (double)bits_to_f32(computed[i]) - reference[i];
        num += diff * diff;
        den += (double)reference[i] * reference[i];
    }
    return den > 0 ? sqrt(num / den) : 0.0;
}

static void check_layer(const char *dump_name, const char *fixture_name,
                        size_t count) {
    char path[512];
    snprintf(path, sizeof(path), DUMP_DIR "/%s.bin", dump_name);
    size_t bytes = 0;
    uint16_t *computed = read_file(path, &bytes);
    snprintf(path, sizeof(path), FIXTURE_DIR "/%s", fixture_name);
    size_t fbytes = 0;
    float *reference = read_file(path, &fbytes);
    if (!computed || !reference || bytes / 2 != count ||
        fbytes / 4 != count) {
        printf("  %-18s MISSING/SIZE (dump %zu ref %zu)\n", dump_name,
               bytes / 2, fbytes / 4);
        if (bytes / 2 != count || fbytes / 4 != count) failures++;
        free(computed);
        free(reference);
        return;
    }
    double rel = rel_l2(computed, reference, count);
    printf("  %-18s relL2 %.3e\n", dump_name, rel);
    free(computed);
    free(reference);
}

int main(void) {
    char error[1024] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu create failed: %s\n", error);
        return 1;
    }
    qw_text *text = qw_text_create(gpu, MODEL_DIR, 36, error, sizeof(error));
    if (!text) {
        printf("text encoder create failed: %s\n", error);
        h3_gpu_free(gpu);
        return 1;
    }

    size_t bytes = 0;
    uint16_t *embeds = read_file(FIXTURE_DIR "/text_vlm_input_embeds.bf16",
                                 &bytes);
    size_t seq = bytes / 2 / QWEN_TEXT_HIDDEN;
    int32_t *pos_ids =
        read_file(FIXTURE_DIR "/text_vlm_pos_ids.i32", &bytes);
    size_t pos_count = bytes / 4;
    uint8_t *mask =
        read_file(FIXTURE_DIR "/text_vlm_visual_mask.u8", &bytes);
    size_t mask_count = bytes;
    uint16_t *deepstack =
        read_file(FIXTURE_DIR "/text_vlm_deepstack.bf16", &bytes);
    size_t n_vis = bytes / 2 / 3 / QWEN_TEXT_HIDDEN;
    if (!embeds || !pos_ids || !mask || !deepstack ||
        pos_count != 3 * seq || mask_count != seq) {
        printf("fixture missing/mismatched (seq %zu pos %zu mask %zu)\n",
               seq, pos_count, mask_count);
        return 1;
    }
    printf("seq=%zu visual slots=%zu\n", seq, n_vis);

    mkdir(DUMP_DIR, 0755);
    setenv("QWEN_TEXT_DUMP", DUMP_DIR, 1);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(
        gpu, (size_t)seq * QWEN_TEXT_HIDDEN);
    if (h3_gpu_begin(gpu) &&
        qw_text_encode_vlm(text, embeds, pos_ids, mask, deepstack,
                           (uint32_t)seq, (uint32_t)n_vis, hidden, error,
                           sizeof(error)) &&
        h3_gpu_submit(gpu)) {
        /* rope 表（C 端 dump f32 [seq, 64] vs fixture [seq, 128] 前 64 槽） */
        size_t rope_bytes = 0;
        float *rope_cos = read_file(DUMP_DIR "/rope_cos.bin", &rope_bytes);
        float *rope_sin;
        size_t sin_bytes = 0;
        rope_sin = read_file(DUMP_DIR "/rope_sin.bin", &sin_bytes);
        float *ref_cos =
            read_file(FIXTURE_DIR "/text_vlm_rope_cos.f32", &bytes);
        float *ref_sin =
            read_file(FIXTURE_DIR "/text_vlm_rope_sin.f32", &bytes);
        if (rope_cos && rope_sin && ref_cos && ref_sin &&
            rope_bytes / 4 == seq * ROPE_PAIRS &&
            bytes / 4 == seq * QWEN_TEXT_HEAD_DIM) {
            double rel_c = 0, rel_s = 0;
            for (size_t s = 0; s < seq; s++) {
                for (int p = 0; p < ROPE_PAIRS; p++) {
                    double dc = (double)rope_cos[s * ROPE_PAIRS + p] -
                                ref_cos[s * QWEN_TEXT_HEAD_DIM + p];
                    double ds = (double)rope_sin[s * ROPE_PAIRS + p] -
                                ref_sin[s * QWEN_TEXT_HEAD_DIM + p];
                    rel_c += dc * dc;
                    rel_s += ds * ds;
                }
            }
            double den_c = 0, den_s = 0;
            for (size_t i = 0; i < seq * ROPE_PAIRS; i++) {
                den_c += (double)ref_cos[i] * ref_cos[i];
                den_s += (double)ref_sin[i] * ref_sin[i];
            }
            rel_c = sqrt(rel_c / (den_c > 0 ? den_c : 1));
            rel_s = sqrt(rel_s / (den_s > 0 ? den_s : 1));
            printf("  rope_cos            relL2 %.3e\n", rel_c);
            printf("  rope_sin            relL2 %.3e\n", rel_s);
            if (rel_c > 1e-4 || rel_s > 1e-4) failures++;
        } else {
            printf("  rope dump/reference missing or size mismatch\n");
            failures++;
        }
        free(rope_cos);
        free(rope_sin);
        free(ref_cos);
        free(ref_sin);

        /* 层输出：0/1/2 为 DeepStack 注入后（fixture _after），深层直比 */
        printf("  layer outputs (relL2):\n");
        check_layer("layer_out_0", "text_vlm_layer0_after.f32",
                    seq * QWEN_TEXT_HIDDEN);
        check_layer("layer_out_1", "text_vlm_layer1_after.f32",
                    seq * QWEN_TEXT_HIDDEN);
        check_layer("layer_out_2", "text_vlm_layer2_after.f32",
                    seq * QWEN_TEXT_HIDDEN);
        check_layer("layer_out_6", "text_vlm_layer6.f32",
                    seq * QWEN_TEXT_HIDDEN);
        check_layer("layer_out_16", "text_vlm_layer16.f32",
                    seq * QWEN_TEXT_HIDDEN);
        check_layer("layer_out_35", "text_vlm_layer35.f32",
                    seq * QWEN_TEXT_HIDDEN);

        /* 终值（pre-final-norm，即层 35 输出） */
        uint16_t *host = malloc(sizeof(uint16_t) * seq * QWEN_TEXT_HIDDEN);
        h3_gpu_begin(gpu);
        h3_gpu_tensor_read_bf16(hidden, host,
                                seq * QWEN_TEXT_HIDDEN);
        h3_gpu_submit(gpu);
        float *ref_final =
            read_file(FIXTURE_DIR "/text_vlm_final.f32", &bytes);
        if (host && ref_final && bytes / 4 == seq * QWEN_TEXT_HIDDEN) {
            double rel = rel_l2(host, ref_final, seq * QWEN_TEXT_HIDDEN);
            printf("  hidden              relL2 %.3e\n", rel);
        } else {
            printf("  final reference missing\n");
            failures++;
        }
        free(host);
        free(ref_final);
    } else {
        printf("encode_vlm failed: %s\n", error);
        failures++;
    }

    free(embeds);
    free(pos_ids);
    free(mask);
    free(deepstack);
    h3_gpu_tensor_free(hidden);
    qw_text_free(text);
    h3_gpu_free(gpu);
    if (failures == 0) {
        printf("text vlm parity passed\n");
        return 0;
    }
    printf("%d failure(s)\n", failures);
    return 1;
}
