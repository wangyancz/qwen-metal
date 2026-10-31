/* LoRA fusion regression: apply an adapter, then compare the fused
 * weights against an independent naive fp32 recomputation (plain k-order
 * loops, no Accelerate) with round-to-nearest-even bf16 rounding. The
 * engine's sgemm accumulates in a different order, so elements may
 * differ by ~1 bf16 ulp at rounding boundaries; the gate allows 2 ulps
 * and reports the bitexact rate. Also exercises re-apply with a new
 * scale, the idempotent fast path, the error path, and the clear path
 * (base restore must be bit-exact).
 *
 * Needs the model weights and a LoRA adapter:
 *   ./tests/test_lora [adapter.safetensors]
 * (default: the Qwen-Image-2.1 Q1 LoRA under /Volumes/Work/loras) */

#include "qwen_dit.h"

#include "h3_safetensors.h"
#include "qwen_gpu_ext.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_DIR "/Volumes/Work/models/Qwen-Image-2.1.models"
#define DEFAULT_LORA \
    "/Volumes/Work/loras/Qwen-Image-2.1/" \
    "Qwen-Image-2.1-Q1-lora_c2-st3000.safetensors"
#define SECOND_LORA \
    "/Volumes/Work/loras/Qwen-Image-2.1/" \
    "qwen-image2.1-q-3d_c1-st6000.safetensors"

#define HIDDEN 4096
#define MLP 12288
#define RANK 16
#define LAYERS 32

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

static float bits_to_f32(uint16_t bits) {
    uint32_t expanded = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &expanded, sizeof(value));
    return value;
}

static uint16_t f32_to_bits_rne(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static uint16_t *read_tensor_bf16(const h3_gpu_tensor *tensor, size_t count) {
    uint16_t *values = malloc(sizeof(uint16_t) * count);
    if (!values || !h3_gpu_tensor_read_bf16(tensor, values, count)) {
        free(values);
        return NULL;
    }
    return values;
}

/* Load one LoRA pair from the adapter file as fp32 host buffers:
 * a [rank, in], b [out, rank]. Returns 0 on failure (reported). */
static int load_pair_f32(const char *path, int block, const char *module,
                         float **a, float **b, uint32_t *out, uint32_t *in) {
    char error[512];
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, sizeof(error))) {
        printf("cannot open adapter: %s\n", error);
        return 0;
    }
    char key[192];
    snprintf(key, sizeof(key),
             "transformer_blocks.%d.%s.lora_A.default.weight", block,
             module);
    const h3_st_tensor *ta = h3_st_find(&header, key);
    snprintf(key, sizeof(key),
             "transformer_blocks.%d.%s.lora_B.default.weight", block,
             module);
    const h3_st_tensor *tb = h3_st_find(&header, key);
    int ok = ta && tb && ta->ndim == 2 && tb->ndim == 2 &&
             ta->dtype == H3_DTYPE_BF16 && tb->dtype == H3_DTYPE_BF16 &&
             ta->shape[0] == RANK && tb->shape[1] == RANK;
    if (!ok) {
        printf("adapter missing/mismatched pair for %s / block %d\n",
               module, block);
        h3_st_free_header(&header);
        return 0;
    }
    *in = (uint32_t)ta->shape[1];
    *out = (uint32_t)tb->shape[0];
    size_t na = (size_t)RANK * *in, nb = (size_t)*out * RANK;
    uint16_t *ab = malloc(sizeof(uint16_t) * na);
    uint16_t *bb = malloc(sizeof(uint16_t) * nb);
    *a = malloc(sizeof(float) * na);
    *b = malloc(sizeof(float) * nb);
    ok = ab && bb && *a && *b &&
         h3_st_read_data(&header, ta, ab, sizeof(uint16_t) * na, error,
                         sizeof(error)) &&
         h3_st_read_data(&header, tb, bb, sizeof(uint16_t) * nb, error,
                         sizeof(error));
    if (ok) {
        for (size_t i = 0; i < na; i++) (*a)[i] = bits_to_f32(ab[i]);
        for (size_t i = 0; i < nb; i++) (*b)[i] = bits_to_f32(bb[i]);
    } else {
        printf("adapter read failed: %s\n", error);
    }
    free(ab);
    free(bb);
    h3_st_free_header(&header);
    return ok;
}

/* Compare fused bf16 against the naive expectation element-by-element,
 * with a stack of adapters summed in fp32 (base + Σ scale_a · B_a@A_a).
 * rows_limit > 0 checks only the first rows (long segments). */
static void check_fused(const char *label, const uint16_t *fused,
                        const uint16_t *base, const float **b_arr,
                        const float **a_arr, const float *scale_arr, int n,
                        uint32_t rows, uint32_t cols, uint32_t rows_limit) {
    uint32_t checked_rows = rows_limit ? (rows_limit < rows ? rows_limit : rows)
                                       : rows;
    size_t count = (size_t)checked_rows * cols;
    size_t bitexact = 0;
    double worst = 0.0;
    size_t beyond = 0;
    for (size_t i = 0; i < count; i++) {
        uint32_t row = (uint32_t)(i / cols);
        uint32_t col = (uint32_t)(i % cols);
        float dw = 0.0f;
        for (int k = 0; k < n; k++) {
            const float *b = b_arr[k];
            const float *a = a_arr[k];
            for (uint32_t r = 0; r < RANK; r++)
                dw += scale_arr[k] * b[(size_t)row * RANK + r] *
                      a[(size_t)r * cols + col];
        }
        uint16_t expect = f32_to_bits_rne(bits_to_f32(base[i]) + dw);
        if (fused[i] == expect) {
            bitexact++;
            continue;
        }
        float e = bits_to_f32(expect);
        float f = bits_to_f32(fused[i]);
        double ulps = fabs((double)f - (double)e) /
                      (fabs((double)e) * 0x1p-7 + 1e-30);
        if (ulps > worst) worst = ulps;
        if (ulps > 2.0) {
            if (beyond < 3)
                printf("    sample[%zu] base=%.6g fused=%.6g expect=%.6g\n",
                       i, (double)bits_to_f32(base[i]), (double)f, (double)e);
            beyond++;
        }
    }
    printf("  %-26s %zu elements, bitexact %.4f%%, worst %.2f ulp%s\n",
           label, count, 100.0 * (double)bitexact / (double)count, worst,
           beyond ? "  <-- FAIL" : "");
    CHECK(beyond == 0, "%s: %zu elements deviate beyond 2 ulp", label,
          beyond);
}

/* Naive-expectation helper: load every (path, block, module) pair of the
 * adapter stack and run check_fused against the snapshot. Returns 0 on a
 * load failure (reported; counted). */
static float *a_hold[QW_DIT_MAX_LORA], *b_hold[QW_DIT_MAX_LORA];
typedef struct {
    const char *path;
    int block;
    const char *module;
    float scale;
} expect_seg;

static int check_against(const char *label, const uint16_t *fused,
                         const uint16_t *base, const expect_seg *segs, int n,
                         uint32_t rows, uint32_t cols, uint32_t rows_limit) {
    const float *b_arr[QW_DIT_MAX_LORA];
    const float *a_arr[QW_DIT_MAX_LORA];
    float scale_arr[QW_DIT_MAX_LORA];
    for (int i = 0; i < n; i++) {
        uint32_t out = 0, in = 0;
        if (!load_pair_f32(segs[i].path, segs[i].block, segs[i].module,
                           &a_hold[i], &b_hold[i], &out, &in)) {
            failures++;
            return 0;
        }
        b_arr[i] = b_hold[i];
        a_arr[i] = a_hold[i];
        scale_arr[i] = segs[i].scale;
    }
    check_fused(label, fused, base, b_arr, a_arr, scale_arr, n, rows, cols,
                rows_limit);
    for (int i = 0; i < n; i++) {
        free(a_hold[i]);
        free(b_hold[i]);
    }
    return 1;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *lora_path = argc > 1 ? argv[1] : DEFAULT_LORA;
    char error[512];

    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("FAIL gpu: %s\n", error);
        return 1;
    }
    if (!h3_gpu_begin(gpu)) {
        printf("FAIL begin: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    qw_dit *dit = qw_dit_create(gpu, MODEL_DIR, LAYERS, error, sizeof(error));
    if (!dit) {
        printf("FAIL dit create: %s\n", error);
        return 1;
    }
    const h3_gpu_tensor *t_q0 = qw_dit_block_tensor(dit, 0, 0);
    const h3_gpu_tensor *t_fc1_0 = qw_dit_block_tensor(dit, 0, 4);
    const h3_gpu_tensor *t_fc2_0 = qw_dit_block_tensor(dit, 0, 5);
    const h3_gpu_tensor *t_v17 = qw_dit_block_tensor(dit, 17, 2);
    CHECK(t_q0 && t_fc1_0 && t_fc2_0 && t_v17, "block tensor accessor");

    /* Base snapshots before any fuse. */
    uint16_t *base_q0 = read_tensor_bf16(t_q0, (size_t)HIDDEN * HIDDEN);
    uint16_t *base_fc1 = read_tensor_bf16(t_fc1_0,
                                          (size_t)2 * MLP * HIDDEN);
    uint16_t *base_fc2 = read_tensor_bf16(t_fc2_0,
                                          (size_t)HIDDEN * MLP);
    uint16_t *base_v17 = read_tensor_bf16(t_v17, (size_t)HIDDEN * HIDDEN);
    CHECK(base_q0 && base_fc1 && base_fc2 && base_v17, "base snapshots");
    if (!base_q0 || !base_fc1 || !base_fc2 || !base_v17) return 1;

    /* Adapter files: the first from argv[1] (default Q1), a second one
     * for the stacking scenarios (default q-3d). */
    const char *lora_a = lora_path;
    const char *lora_b = argc > 2 ? argv[2] : SECOND_LORA;
    printf("lora regression: %s + %s\n", lora_a, lora_b);

    /* 1) single adapter at 0.7 — full naive verification. */
    CHECK(qw_dit_apply_lora(dit, lora_a, 0.7f, error, sizeof(error)),
          "apply A 0.7: %s", error);
    CHECK(qw_dit_lora_count(dit) == 1, "count 1");
    CHECK(qw_dit_lora_path_at(dit, 0) &&
              strcmp(qw_dit_lora_path_at(dit, 0), lora_a) == 0,
          "path_at(0)");
    CHECK(qw_dit_lora_scale_at(dit, 0) == 0.7f, "scale_at(0)");

    uint16_t *fused = read_tensor_bf16(qw_dit_block_tensor(dit, 0, 0),
                                       (size_t)HIDDEN * HIDDEN);
    CHECK(fused != NULL, "fused to_q read");
    check_against("to_q L0 A@0.7", fused, base_q0,
                  (expect_seg[]){{lora_a, 0, "attn.to_q", 0.7f}}, 1, HIDDEN,
                  HIDDEN, 0);
    free(fused);

    fused = read_tensor_bf16(qw_dit_block_tensor(dit, 0, 4),
                             (size_t)2 * MLP * HIDDEN);
    CHECK(fused != NULL, "fused fc1 read");
    check_against("fc1 gate L0 A@0.7", fused, base_fc1,
                  (expect_seg[]){{lora_a, 0, "img_mlp.gate_layer", 0.7f}}, 1,
                  MLP, HIDDEN, 1024);
    check_against("fc1 proj L0 A@0.7", fused + (size_t)MLP * HIDDEN,
                  base_fc1 + (size_t)MLP * HIDDEN,
                  (expect_seg[]){{lora_a, 0, "img_mlp.proj", 0.7f}}, 1, MLP,
                  HIDDEN, 1024);
    free(fused);

    fused = read_tensor_bf16(qw_dit_block_tensor(dit, 0, 5),
                             (size_t)HIDDEN * MLP);
    CHECK(fused != NULL, "fused fc2 read");
    check_against("fc2 L0 A@0.7", fused, base_fc2,
                  (expect_seg[]){{lora_a, 0, "img_mlp.out", 0.7f}}, 1, HIDDEN,
                  MLP, 0);
    free(fused);

    fused = read_tensor_bf16(qw_dit_block_tensor(dit, 17, 2),
                             (size_t)HIDDEN * HIDDEN);
    CHECK(fused != NULL, "fused to_v read");
    check_against("to_v L17 A@0.7", fused, base_v17,
                  (expect_seg[]){{lora_a, 17, "attn.to_v", 0.7f}}, 1, HIDDEN,
                  HIDDEN, 0);
    free(fused);

    /* 2) stack a second adapter: W' = base + 0.7·A + 0.5·B. */
    CHECK(qw_dit_apply_lora(dit, lora_b, 0.5f, error, sizeof(error)),
          "apply B 0.5: %s", error);
    CHECK(qw_dit_lora_count(dit) == 2, "count 2");
    CHECK(qw_dit_lora_scale_at(dit, 0) == 0.7f &&
              qw_dit_lora_scale_at(dit, 1) == 0.5f,
          "stack order preserved");
    fused = read_tensor_bf16(qw_dit_block_tensor(dit, 0, 0),
                             (size_t)HIDDEN * HIDDEN);
    CHECK(fused != NULL, "fused to_q read (stack)");
    check_against("to_q L0 A@0.7+B@0.5", fused, base_q0,
                  (expect_seg[]){{lora_a, 0, "attn.to_q", 0.7f},
                                 {lora_b, 0, "attn.to_q", 0.5f}},
                  2, HIDDEN, HIDDEN, 0);
    free(fused);

    /* 3) update the first adapter's scale in place. */
    CHECK(qw_dit_apply_lora(dit, lora_a, 1.0f, error, sizeof(error)),
          "apply A 1.0: %s", error);
    CHECK(qw_dit_lora_count(dit) == 2 &&
              qw_dit_lora_scale_at(dit, 0) == 1.0f &&
              qw_dit_lora_scale_at(dit, 1) == 0.5f,
          "scale updated in place");
    fused = read_tensor_bf16(qw_dit_block_tensor(dit, 0, 0),
                             (size_t)HIDDEN * HIDDEN);
    CHECK(fused != NULL, "fused to_q read (scale update)");
    check_against("to_q L0 A@1.0+B@0.5", fused, base_q0,
                  (expect_seg[]){{lora_a, 0, "attn.to_q", 1.0f},
                                 {lora_b, 0, "attn.to_q", 0.5f}},
                  2, HIDDEN, HIDDEN, 0);
    free(fused);

    /* 4) remove the first adapter from the middle of the stack. */
    CHECK(qw_dit_remove_lora(dit, lora_a, error, sizeof(error)),
          "remove A: %s", error);
    CHECK(qw_dit_lora_count(dit) == 1 &&
              strcmp(qw_dit_lora_path_at(dit, 0), lora_b) == 0,
          "stack after remove");
    fused = read_tensor_bf16(qw_dit_block_tensor(dit, 0, 0),
                             (size_t)HIDDEN * HIDDEN);
    CHECK(fused != NULL, "fused to_q read (removed)");
    check_against("to_q L0 B@0.5", fused, base_q0,
                  (expect_seg[]){{lora_b, 0, "attn.to_q", 0.5f}}, 1, HIDDEN,
                  HIDDEN, 0);
    free(fused);

    /* 5) remove the last: the base restore must be bit-exact. */
    CHECK(qw_dit_remove_lora(dit, lora_b, error, sizeof(error)),
          "remove B: %s", error);
    CHECK(qw_dit_lora_count(dit) == 0, "stack empty");
    CHECK(qw_dit_remove_lora(dit, lora_b, error, sizeof(error)),
          "remove is a no-op when absent");
    CHECK(qw_dit_clear_lora(dit, error, sizeof(error)),
          "clear is idempotent");
    fused = read_tensor_bf16(qw_dit_block_tensor(dit, 0, 0),
                             (size_t)HIDDEN * HIDDEN);
    CHECK(fused && memcmp(fused, base_q0,
                          sizeof(uint16_t) * HIDDEN * HIDDEN) == 0,
          "to_q restored bit-exact");
    free(fused);
    fused = read_tensor_bf16(qw_dit_block_tensor(dit, 0, 4),
                             (size_t)2 * MLP * HIDDEN);
    CHECK(fused && memcmp(fused, base_fc1,
                          sizeof(uint16_t) * 2 * MLP * HIDDEN) == 0,
          "fc1 restored bit-exact");
    free(fused);

    /* 6) error path: a missing file must fail and leave state alone. */
    CHECK(qw_dit_apply_lora(dit, lora_a, 0.5f, error, sizeof(error)),
          "fresh cycle apply: %s", error);
    CHECK(!qw_dit_apply_lora(dit, "/nonexistent-xyz.safetensors", 1.0f,
                             error, sizeof(error)),
          "missing file must fail");
    CHECK(qw_dit_lora_count(dit) == 1 &&
              qw_dit_lora_scale_at(dit, 0) == 0.5f,
          "state intact after error");

    free(base_q0);
    free(base_fc1);
    free(base_fc2);
    free(base_v17);
    qw_dit_free(dit);
    printf(failures ? "lora regression: %d FAILURES\n"
                    : "lora regression: all green\n",
           failures);
    return failures ? 1 : 0;
}