#include "qwen_text_encoder.h"

#include "qwen_host.h"

#include "h3_tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MODEL_DIR "/Volumes/Work/models/Qwen-Image-2.1.models"
#define FIXTURE_DIR "tests/text_fixture"
#define PROMPT "A red apple on a wooden table, morning light, 4k photo"

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
        failures++;
        free(values);
        values = NULL;
    }
    fclose(file);
    return values;
}

static int read_i32(const char *name, int32_t **values, size_t *count) {
    char path[512];
    snprintf(path, sizeof(path), FIXTURE_DIR "/%s", name);
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    fseek(file, 0, SEEK_END);
    long bytes = ftell(file);
    rewind(file);
    *count = (size_t)bytes / 4;
    *values = malloc((size_t)bytes);
    if (fread(*values, 4, *count, file) != *count) {
        failures++;
        free(*values);
        *values = NULL;
        fclose(file);
        return 0;
    }
    fclose(file);
    return 1;
}

static float bits_to_f32(uint16_t bits) {
    uint32_t expanded = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &expanded, sizeof(value));
    return value;
}

static double compare_stage(const char *label, const uint16_t *computed,
                            const uint16_t *reference, size_t count) {
    double sum = 0.0, rel = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < count; i++) {
        double a = bits_to_f32(computed[i]);
        double b = bits_to_f32(reference[i]);
        double diff = a - b;
        sum += diff * diff;
        rel += b * b;
        if (fabs(diff) > max_abs) max_abs = fabs(diff);
    }
    rel = sqrt(sum / (rel > 1e-12 ? rel : 1e-12));
    printf("  %-16s relL2 %.3g  maxAbs %.3g\n", label, rel, max_abs);
    return rel;
}

/* Staged layer-0 parity: compare the encoder's debug dump against the
 * hook-captured fixture stages. Embedding must be bitwise exact; other
 * stages tolerate rounding but must stay far below the intrinsic band. */
static void compare_dump(const char *name, int size_class, size_t tokens) {
    size_t stage_count;
    if (size_class == 1) {
        stage_count = (size_t)tokens * QWEN_KV_HEADS_TEXT *
                      QWEN_TEXT_HEAD_DIM;
    } else if (size_class == 2) {
        stage_count = (size_t)tokens * QWEN_TEXT_MLP;
    } else {
        stage_count = (size_t)tokens * QWEN_TEXT_HIDDEN;
    }
    char ref_name[96];
    snprintf(ref_name, sizeof(ref_name), "%s.bin", name);
    /* Structural absences (stages the engine only dumps for layer 0) are
     * skipped quietly; read_bf16 would count them as failures. */
    char probe_path[512];
    snprintf(probe_path, sizeof(probe_path), FIXTURE_DIR "/%s", ref_name);
    FILE *probe = fopen(probe_path, "rb");
    if (!probe) {
        printf("  %-16s (no fixture)\n", name);
        return;
    }
    fclose(probe);
    size_t ref_count = 0;
    uint16_t *ref = read_bf16(ref_name, &ref_count);
    if (!ref) return;
    char dump_path[512];
    snprintf(dump_path, sizeof(dump_path), FIXTURE_DIR "/cdump/%s.bin",
             name);
    FILE *file = fopen(dump_path, "rb");
    if (!file) {
        printf("  %-16s (no dump)\n", name);
        failures++;
        free(ref);
        return;
    }
    uint16_t *computed = malloc(sizeof(uint16_t) * stage_count);
    size_t read = fread(computed, 2, stage_count, file);
    fclose(file);
    if (read != stage_count || ref_count != stage_count) {
        printf("  %-16s size mismatch dump %zu ref %zu\n", name, read,
               ref_count);
        failures++;
        free(ref);
        free(computed);
        return;
    }
    double rel = compare_stage(name, computed, ref, stage_count);
    if (strcmp(name, "l0_embed") == 0) {
        CHECK(rel == 0.0, "embed must be bitwise exact, relL2 %.3g", rel);
    } else {
        CHECK(rel < 0.05, "stage %s relL2 %.3g", name, rel);
    }
    free(ref);
    free(computed);
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

    /* Template tokenization against the released tokenizer. */
    uint32_t *ids = NULL;
    size_t count = 0, drop = 0;
    CHECK(qw_text_template_tokens(text, PROMPT, &ids, &count, &drop, error,
                                  sizeof(error)),
          "template tokens: %s", error);
    int32_t *ref_ids = NULL;
    size_t ref_count = 0;
    CHECK(read_i32("input_ids.bin", &ref_ids, &ref_count), "read ids");
    CHECK(count == ref_count, "token count %zu vs %zu", count, ref_count);
    size_t mismatches = 0;
    for (size_t i = 0; i < count && i < ref_count; i++) {
        if (ids[i] != (uint32_t)ref_ids[i]) mismatches++;
    }
    CHECK(mismatches == 0, "token id mismatches: %zu/%zu", mismatches, count);
    if (mismatches) {
        for (size_t i = 0; i < count && i < ref_count; i++) {
            if (ids[i] != (uint32_t)ref_ids[i]) {
                printf("  first mismatch at %zu: %u vs %u\n", i, ids[i],
                       (uint32_t)ref_ids[i]);
                break;
            }
        }
    }
    CHECK(drop == 14, "drop %zu vs 14", drop);

    /* Full 36-layer encode, with debug dumps for the staged comparison. */
    mkdir(FIXTURE_DIR "/cdump", 0755);
    setenv("QWEN_TEXT_DUMP", FIXTURE_DIR "/cdump", 1);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(
        gpu, (size_t)count * QWEN_TEXT_HIDDEN);
    CHECK(h3_gpu_begin(gpu), "begin: %s", h3_gpu_error(gpu));
    CHECK(qw_text_encode(text, ids, (uint32_t)count, hidden, error,
                         sizeof(error)),
          "encode: %s", error);
    CHECK(h3_gpu_submit(gpu), "submit: %s", h3_gpu_error(gpu));

    size_t host_count = 0;
    uint16_t *ref_hidden = read_bf16("hidden.bin", &host_count);
    uint16_t *ref_f32 = read_bf16("hidden_f32.bin", &host_count);
    size_t out_rows = count - drop;
    uint16_t *host = malloc(sizeof(uint16_t) * (size_t)out_rows *
                            QWEN_TEXT_HIDDEN);
    h3_gpu_tensor *post = h3_gpu_tensor_new_bf16(
        gpu, (size_t)out_rows * QWEN_TEXT_HIDDEN);
    CHECK(h3_gpu_begin(gpu), "begin: %s", h3_gpu_error(gpu));
    CHECK(h3_gpu_copy_bf16(gpu, post, 0, hidden,
                           (size_t)drop * QWEN_TEXT_HIDDEN,
                           (size_t)out_rows * QWEN_TEXT_HIDDEN),
          "drop copy");
    CHECK(h3_gpu_submit(gpu), "submit: %s", h3_gpu_error(gpu));
    h3_gpu_tensor_read_bf16(post, host, (size_t)out_rows * QWEN_TEXT_HIDDEN);

    double rel = compare_stage("vs bf16 ref", host, ref_hidden,
                               (size_t)out_rows * QWEN_TEXT_HIDDEN);
    double intrinsic = compare_stage("py bf16 vs f32", ref_hidden, ref_f32,
                                     (size_t)out_rows * QWEN_TEXT_HIDDEN);
    compare_stage("vs f32 yardstick", host, ref_f32,
                  (size_t)out_rows * QWEN_TEXT_HIDDEN);
    /* Two independent bf16 realizations (ours and py's) differ by about
     * sqrt(2) x the py-bf16-vs-fp32 distance; the remaining freedom is
     * gemm accumulation order. 1.5x covers it while a real wiring bug
     * would show rel >= 1. */
    CHECK(rel < fmax(3e-2, intrinsic * 1.5),
          "hidden relL2 %.3g (intrinsic %.3g)", rel, intrinsic);

    /* Staged comparisons: layer 0 (wiring) and layer 16 (massive
     * activation emergence). */
    struct { const char *suffix; int size_class; } stages[] = {
        {"embed", 0}, {"input_ln", 0}, {"q_proj", 0}, {"k_proj", 1},
        {"v_proj", 1}, {"q_norm", 0}, {"k_norm", 1}, {"q_embed", 0},
        {"k_embed", 1}, {"attn", 0}, {"o_proj", 0}, {"res1", 0},
        {"post_ln", 0}, {"act", 2}, {"mlp", 0}, {"out", 0},
    };
    const char *prefixes[] = {"l0_", "l6_", "l16_"};
    for (size_t p = 0; p < sizeof(prefixes) / sizeof(prefixes[0]); p++) {
        printf("  staged %s parity:\n", prefixes[p]);
        for (size_t s = 0; s < sizeof(stages) / sizeof(stages[0]); s++) {
            char name[64];
            snprintf(name, sizeof(name), "%s%s", prefixes[p],
                     stages[s].suffix);
            compare_dump(name, stages[s].size_class, count);
        }
    }
    printf("  per-layer output drift:\n");
    for (int li = 0; li < 36; li++) {
        char name[32];
        snprintf(name, sizeof(name), "layer_out_%d", li);
        compare_dump(name, 0, count);
    }

    h3_tokenizer_ids_free(ids);
    free(ref_ids);
    free(ref_hidden);
    free(ref_f32);
    free(host);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(post);
    qw_text_free(text);
    h3_gpu_free(gpu);
    if (failures == 0) {
        printf("text encoder parity passed\n");
        return 0;
    }
    printf("%d failure(s)\n", failures);
    return 1;
}
