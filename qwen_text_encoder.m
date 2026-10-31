#include "qwen_text_encoder.h"

#include <time.h>

#include "h3_tokenizer.h"
#include "h3_weights.h"
#include "qwen_gpu_ext.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_TEXT_ROPE_PAIRS (QWEN_TEXT_HEAD_DIM / 2)
#define QWEN_TEXT_THETA 5000000.0

static double pipe_now_text(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Round-to-nearest-even quantization to bf16 precision, matching the
 * transformers cast of the cos/sin tables to the compute dtype. */
static float bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

typedef struct {
    h3_gpu_tensor *input_norm;
    h3_gpu_tensor *q_proj;
    h3_gpu_tensor *k_proj;
    h3_gpu_tensor *v_proj;
    h3_gpu_tensor *o_proj;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    /* Fused [2 * mlp, hidden]: gate_proj rows then up_proj rows. */
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *down;
    h3_gpu_tensor *post_norm;
} qw_text_layer;

struct qw_text {
    h3_gpu *gpu;
    h3_weight_store *store;
    h3_tokenizer *tokenizer;
    int layers;
    qw_text_layer *layer;
    h3_gpu_tensor *embed;
    char tokenizer_path[4096];
    uint32_t image_pad_id; /* 惰性 tokenize <|image_pad|> */
};

static int set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error && error_size) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(error, error_size, fmt, args);
        va_end(args);
    }
    return 0;
}

qw_text *qw_text_create(h3_gpu *gpu, const char *model_dir, int layers,
                        char *error, size_t error_size) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/text_encoder", model_dir);
    h3_weight_store *store = h3_weight_store_open(path, error, error_size);
    if (!store) return NULL;

    qw_text *text = calloc(1, sizeof(qw_text));
    if (!text) {
        h3_weight_store_free(store);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    text->gpu = gpu;
    text->store = store;
    text->layers = layers;
    snprintf(text->tokenizer_path, sizeof(text->tokenizer_path),
             "%s/processor/tokenizer.json", model_dir);

    const uint64_t hidden2[2] = {QWEN_TEXT_HIDDEN, QWEN_TEXT_HIDDEN};
    const uint64_t kv_hidden[2] = {QWEN_KV_HEADS_TEXT * QWEN_TEXT_HEAD_DIM,
                                   QWEN_TEXT_HIDDEN};
    const uint64_t mlp_hidden[2] = {QWEN_TEXT_MLP, QWEN_TEXT_HIDDEN};
    const uint64_t hidden_mlp[2] = {QWEN_TEXT_HIDDEN, QWEN_TEXT_MLP};
    const uint64_t scalar[1] = {QWEN_TEXT_HIDDEN};
    const uint64_t scalar_dim[1] = {QWEN_TEXT_HEAD_DIM};
    const uint64_t vocab[2] = {151936, QWEN_TEXT_HIDDEN};

    text->embed = h3_weight_load_bf16(
        store, gpu, "model.language_model.embed_tokens.weight", 2, vocab,
        error, error_size);
    if (!text->embed) {
        qw_text_free(text);
        return NULL;
    }

    text->layer = calloc((size_t)layers, sizeof(qw_text_layer));
    if (!text->layer) {
        qw_text_free(text);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    for (int i = 0; i < layers; i++) {
        char name[192];
        qw_text_layer *layer = &text->layer[i];
#define QW_LOAD(field, suffix, ndim, shape)                              \
        do {                                                             \
            snprintf(name, sizeof(name),                                 \
                     "model.language_model.layers.%d." suffix, i);       \
            layer->field = h3_weight_load_bf16(store, gpu, name, ndim,   \
                                               shape, error, error_size);\
            if (!layer->field) {                                         \
                qw_text_free(text);                                      \
                return NULL;                                             \
            }                                                            \
        } while (0)
        QW_LOAD(input_norm, "input_layernorm.weight", 1, scalar);
        QW_LOAD(q_proj, "self_attn.q_proj.weight", 2, hidden2);
        QW_LOAD(k_proj, "self_attn.k_proj.weight", 2, kv_hidden);
        QW_LOAD(v_proj, "self_attn.v_proj.weight", 2, kv_hidden);
        QW_LOAD(o_proj, "self_attn.o_proj.weight", 2, hidden2);
        QW_LOAD(q_norm, "self_attn.q_norm.weight", 1, scalar_dim);
        QW_LOAD(k_norm, "self_attn.k_norm.weight", 1, scalar_dim);
        QW_LOAD(post_norm, "post_attention_layernorm.weight", 1, scalar);
        QW_LOAD(down, "mlp.down_proj.weight", 2, hidden_mlp);
#undef QW_LOAD

        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.mlp.gate_proj.weight", i);
        h3_gpu_tensor *gate = h3_weight_load_bf16(store, gpu, name, 2,
                                                  mlp_hidden, error,
                                                  error_size);
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.mlp.up_proj.weight", i);
        h3_gpu_tensor *up = h3_weight_load_bf16(store, gpu, name, 2,
                                                mlp_hidden, error, error_size);
        if (!gate || !up) {
            h3_gpu_tensor_free(gate);
            h3_gpu_tensor_free(up);
            qw_text_free(text);
            return NULL;
        }
        layer->fc1 = h3_gpu_tensor_new_bf16(
            gpu, (size_t)2 * QWEN_TEXT_MLP * QWEN_TEXT_HIDDEN);
        if (!layer->fc1) {
            h3_gpu_tensor_free(gate);
            h3_gpu_tensor_free(up);
            qw_text_free(text);
            set_error(error, error_size, "fc1 allocation failed");
            return NULL;
        }
        size_t half = (size_t)QWEN_TEXT_MLP * QWEN_TEXT_HIDDEN;
        uint16_t *host = malloc(sizeof(uint16_t) * half);
        int ok = host && h3_gpu_tensor_read_bf16(gate, host, half) &&
                 h3_gpu_tensor_write_bf16_range(layer->fc1, 0, host, half) &&
                 h3_gpu_tensor_read_bf16(up, host, half) &&
                 h3_gpu_tensor_write_bf16_range(layer->fc1, half, host, half);
        free(host);
        h3_gpu_tensor_free(gate);
        h3_gpu_tensor_free(up);
        if (!ok) {
            qw_text_free(text);
            set_error(error, error_size, "fc1 concat failed at layer %d", i);
            return NULL;
        }
    }
    return text;
}

void qw_text_free(qw_text *text) {
    if (!text) return;
    if (text->layer) {
        for (int i = 0; i < text->layers; i++) {
            h3_gpu_tensor_free(text->layer[i].input_norm);
            h3_gpu_tensor_free(text->layer[i].q_proj);
            h3_gpu_tensor_free(text->layer[i].k_proj);
            h3_gpu_tensor_free(text->layer[i].v_proj);
            h3_gpu_tensor_free(text->layer[i].o_proj);
            h3_gpu_tensor_free(text->layer[i].q_norm);
            h3_gpu_tensor_free(text->layer[i].k_norm);
            h3_gpu_tensor_free(text->layer[i].fc1);
            h3_gpu_tensor_free(text->layer[i].down);
            h3_gpu_tensor_free(text->layer[i].post_norm);
        }
        free(text->layer);
    }
    h3_gpu_tensor_free(text->embed);
    h3_weight_store_free(text->store);
    h3_tokenizer_free(text->tokenizer);
    free(text);
}

static int run_linear(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                      char *error, size_t error_size, const char *label) {
    if (!h3_gpu_linear_bf16(gpu, output, input, weight, NULL, rows, input_dim,
                            output_dim)) {
        return set_error(error, error_size, "%s: linear failed: %s", label,
                         h3_gpu_error(gpu));
    }
    return 1;
}

/* Debug helper: write a stage tensor as raw bf16 for the staged parity
 * harness. Host reads must drain the command buffer first. */
static int dump_tensor(h3_gpu *gpu, const char *dir, const char *name,
                       const h3_gpu_tensor *tensor, size_t count) {
    if (!dir || !dir[0]) return 1;
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    uint16_t *host = malloc(sizeof(uint16_t) * count);
    if (!host) return 0;
    h3_gpu_submit(gpu);
    h3_gpu_begin(gpu);
    int ok = h3_gpu_tensor_read_bf16(tensor, host, count);
    if (ok) {
        FILE *file = fopen(path, "wb");
        ok = file && fwrite(host, sizeof(uint16_t), count, file) == count;
        if (file) fclose(file);
    }
    free(host);
    return ok;
}

/* Stage-gated variant used inside the forward chain: layer 0 proves the
 * wiring; layers 6 and 16 are massive-activation emergence points. */
static int dump_stage(h3_gpu *gpu, const char *dir, int layer_index,
                      const char *suffix, const h3_gpu_tensor *tensor,
                      size_t count) {
    if (layer_index != 0 && layer_index != 6 && layer_index != 8 &&
        layer_index != 10 && layer_index != 12 && layer_index != 14 &&
        layer_index != 16) return 1;
    char name[64];
    snprintf(name, sizeof(name), "l%d_%s", layer_index, suffix);
    return dump_tensor(gpu, dir, name, tensor, count);
}

int qw_text_template_tokens(qw_text *text, const char *prompt, uint32_t **ids,
                            size_t *count, size_t *drop, char *error,
                            size_t error_size) {
    if (!text->tokenizer) {
        text->tokenizer = h3_tokenizer_load(text->tokenizer_path, error,
                                            error_size);
        if (!text->tokenizer) {
            return set_error(error, error_size, "tokenizer load failed");
        }
    }
    char system_message[512];
    snprintf(system_message, sizeof(system_message),
             "<|im_start|>system\n%s<|im_end|>\n", QWEN_TEXT_SYS_PROMPT);
    uint32_t *system_ids = NULL;
    size_t system_count = 0;
    if (!h3_tokenizer_encode(text->tokenizer, system_message, 0, &system_ids,
                             &system_count, error, error_size)) {
        return set_error(error, error_size, "system tokenize failed");
    }
    h3_tokenizer_ids_free(system_ids);
    *drop = system_count;

    size_t total = strlen(system_message) + strlen(prompt) + 64;
    char *full = malloc(total);
    if (!full) return set_error(error, error_size, "out of memory");
    snprintf(full, total,
             "%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
             system_message, prompt);
    int ok = h3_tokenizer_encode(text->tokenizer, full, 0, ids, count, error,
                                 error_size);
    free(full);
    return ok;
}

/* Shared forward: mrope table + decoder layers + DeepStack injection.
 * x carries the already-uploaded embeddings and is freed by the caller.
 * pos_ids NULL degenerates to the pure-text 1-D rope (every axis shares
 * position s); visual_mask/deepstack NULL skip the DeepStack adds. */
static int encode_core(qw_text *text, h3_gpu_tensor *x, uint32_t seq,
                       const int32_t *pos_ids, const uint8_t *visual_mask,
                       const uint16_t *deepstack_bf16, uint32_t n_vis,
                       h3_gpu_tensor *hidden, char *error, size_t error_size) {
    const char *dump_dir = getenv("QWEN_TEXT_DUMP");
    /* Interleaved mrope: of the 64 rotation pairs the first 60 alternate
     * T/H/W (pair p takes axis p % 3) and pairs 60-63 fall back to T; the
     * 128 rotation slots duplicate those angles for the half-split
     * pairing. Pure text degenerates to the 1-D rope: every axis shares
     * position s, so pair p carries theta^(-2p/128) regardless of the
     * 24/20/20 section split. The text tower rotates half-split (element
     * i pairs with i + head_dim / 2), unlike the DiT's interleaved
     * adjacent-pair rotation. */
    float *rope_host = malloc(sizeof(float) * 2 * (size_t)seq *
                              QWEN_TEXT_ROPE_PAIRS);
    if (!rope_host) return set_error(error, error_size, "out of memory");
    int text_timing_early = getenv("QWEN_TEXT_TIMING") != NULL;
    double text_t0 = text_timing_early ? pipe_now_text() : 0.0;
    float *cos_host = rope_host;
    float *sin_host = rope_host + (size_t)seq * QWEN_TEXT_ROPE_PAIRS;
    for (uint32_t pos = 0; pos < seq; pos++) {
        for (int p = 0; p < QWEN_TEXT_ROPE_PAIRS; p++) {
            /* transformers stores inv_freq as a module buffer, so
             * model.to(bfloat16) quantizes the FREQUENCIES themselves;
             * the phase is then computed in fp32 from the quantized
             * values, and the cos/sin tables are quantized to bf16 once
             * more (rotary_emb returns .to(x.dtype)). Both quantizations
             * are systematic and must be replicated. */
            float inv = bf16_round((float)pow(QWEN_TEXT_THETA,
                                              -2.0 * p / QWEN_TEXT_HEAD_DIM));
            int axis = (p < 60) ? (p % 3) : 0;
            int32_t pos_val = pos_ids ? pos_ids[(size_t)axis * seq + pos]
                                      : (int32_t)pos;
            float argument = (float)pos_val * inv;
            cos_host[(size_t)pos * QWEN_TEXT_ROPE_PAIRS + p] =
                bf16_round((float)cos((double)argument));
            sin_host[(size_t)pos * QWEN_TEXT_ROPE_PAIRS + p] =
                bf16_round((float)sin((double)argument));
        }
    }
    h3_gpu_tensor *rope_cos_t = h3_gpu_tensor_from_f32(
        text->gpu, cos_host, (size_t)seq * QWEN_TEXT_ROPE_PAIRS);
    h3_gpu_tensor *rope_sin_t = h3_gpu_tensor_from_f32(
        text->gpu, sin_host, (size_t)seq * QWEN_TEXT_ROPE_PAIRS);
    if (text_timing_early)
        fprintf(stderr, "[text] rope build %.3fs\n",
                pipe_now_text() - text_t0);
    text_t0 = pipe_now_text();
    if (dump_dir && dump_dir[0]) {
        /* rope 表是 f32 tensor，dump_tensor 的 bf16 读法不适用；host
         * 数组与上传数据同源，直接落盘 f32（free 前写）。 */
        char path[4096];
        snprintf(path, sizeof(path), "%s/rope_cos.bin", dump_dir);
        FILE *file = fopen(path, "wb");
        if (file) {
            fwrite(cos_host, sizeof(float),
                   (size_t)seq * QWEN_TEXT_ROPE_PAIRS, file);
            fclose(file);
        }
        snprintf(path, sizeof(path), "%s/rope_sin.bin", dump_dir);
        file = fopen(path, "wb");
        if (file) {
            fwrite(sin_host, sizeof(float),
                   (size_t)seq * QWEN_TEXT_ROPE_PAIRS, file);
            fclose(file);
        }
    }
    free(rope_host);

    size_t elements = (size_t)seq * QWEN_TEXT_HIDDEN;
    size_t kv_elements = (size_t)seq * QWEN_KV_HEADS_TEXT * QWEN_TEXT_HEAD_DIM;
    size_t mlp_elements = (size_t)seq * QWEN_TEXT_MLP;
    size_t fc1_elements = (size_t)seq * 2 * QWEN_TEXT_MLP;
    /* The flash attention kernel reads whole 64-row query tiles — the
     * query buffer must be padded past seq (output writes are guarded).
     * Causal limits are shared by all layers: row r attends keys [0, r]. */
    uint32_t q_pad_rows = (seq + 63u) & ~63u;
    size_t q_elements = (size_t)q_pad_rows * QWEN_TEXT_HIDDEN;
    uint32_t *limit_host = malloc(sizeof(uint32_t) * (seq ? seq : 1));
    h3_gpu_tensor *normed = h3_gpu_tensor_new_bf16(text->gpu, elements);
    h3_gpu_tensor *q = h3_gpu_tensor_new_bf16(text->gpu, q_elements);
    h3_gpu_tensor *k = h3_gpu_tensor_new_bf16(text->gpu, kv_elements);
    h3_gpu_tensor *v = h3_gpu_tensor_new_bf16(text->gpu, kv_elements);
    h3_gpu_tensor *attn = h3_gpu_tensor_new_bf16(text->gpu, elements);
    h3_gpu_tensor *branch = h3_gpu_tensor_new_bf16(text->gpu, elements);
    h3_gpu_tensor *fc1out = h3_gpu_tensor_new_bf16(text->gpu, fc1_elements);
    h3_gpu_tensor *mlp_mid = h3_gpu_tensor_new_bf16(text->gpu, mlp_elements);
    h3_gpu_tensor *causal_limit = NULL;
    if (limit_host) {
        for (uint32_t r = 0; r < seq; r++) limit_host[r] = r + 1;
        causal_limit = h3_gpu_tensor_from_u32(text->gpu, limit_host, seq);
        free(limit_host);
    }
    int ok = rope_cos_t && rope_sin_t && normed && q && k && v && attn &&
             branch && fc1out && mlp_mid && causal_limit;
    if (ok && dump_dir && dump_dir[0]) {
        ok = dump_stage(text->gpu, dump_dir, 0, "embed", x, elements);
    }
    if (text_timing_early)
        fprintf(stderr, "[text] workspace alloc %.3fs\n",
                pipe_now_text() - text_t0);
    int text_timing = text_timing_early;
    /* Submit between op groups: queuing all 36 layers into ONE command
     * buffer makes the 216 MPS graph executions hold every intermediate
     * until the final commit (same retain mechanism as the VAE encode
     * fix) and measured ~5x slower than the same work with staged
     * submits (8.15s -> 1.7s for a 4136-row ti2i encode). QWEN_TEXT_TIMING
     * still prints per-layer wall times; QWEN_TEXT_PROFILE aggregates
     * per-group times. */
    int text_profile = getenv("QWEN_TEXT_PROFILE") != NULL;
    double prof[5] = {0, 0, 0, 0, 0};
    for (int i = 0; ok && i < text->layers; i++) {
        double layer_t0 = text_timing ? pipe_now_text() : 0.0;
        const qw_text_layer *layer = &text->layer[i];
        double g0 = text_profile ? pipe_now_text() : 0.0;
        ok = qw_gpu_rms_norm_bf16(text->gpu, normed, x, layer->input_norm,
                                  seq, QWEN_TEXT_HIDDEN, 1e-6f) &&
             dump_stage(text->gpu, dump_dir, i, "input_ln", normed,
                        elements) &&
             run_linear(text->gpu, q, normed, layer->q_proj, seq,
                        QWEN_TEXT_HIDDEN, QWEN_TEXT_HIDDEN, error, error_size,
                        "q_proj") &&
             dump_stage(text->gpu, dump_dir, i, "q_proj", q, elements) &&
             run_linear(text->gpu, k, normed, layer->k_proj, seq,
                        QWEN_TEXT_HIDDEN,
                        QWEN_KV_HEADS_TEXT * QWEN_TEXT_HEAD_DIM, error,
                        error_size, "k_proj") &&
             dump_stage(text->gpu, dump_dir, i, "k_proj", k, kv_elements) &&
             run_linear(text->gpu, v, normed, layer->v_proj, seq,
                        QWEN_TEXT_HIDDEN,
                        QWEN_KV_HEADS_TEXT * QWEN_TEXT_HEAD_DIM, error,
                        error_size, "v_proj") &&
             dump_stage(text->gpu, dump_dir, i, "v_proj", v, kv_elements) &&
             qw_gpu_head_rms_norm_bf16(text->gpu, q, layer->q_norm, seq,
                                       QWEN_TEXT_HEADS, QWEN_TEXT_HEAD_DIM,
                                       1e-6f) &&
             dump_stage(text->gpu, dump_dir, i, "q_norm", q, elements) &&
             qw_gpu_head_rms_norm_bf16(text->gpu, k, layer->k_norm, seq,
                                       QWEN_KV_HEADS_TEXT,
                                       QWEN_TEXT_HEAD_DIM, 1e-6f) &&
             dump_stage(text->gpu, dump_dir, i, "k_norm", k, kv_elements);
        ok = ok && h3_gpu_submit(text->gpu) && h3_gpu_begin(text->gpu);
        if (text_profile) {
            prof[0] += pipe_now_text() - g0;
            g0 = pipe_now_text();
        }
        ok = ok && qw_gpu_qk_rope_half_bf16(text->gpu, q, q, q, q, rope_cos_t,
                                            rope_sin_t, seq, QWEN_TEXT_HEADS,
                                            QWEN_TEXT_HEAD_DIM) &&
             dump_stage(text->gpu, dump_dir, i, "q_embed", q, elements) &&
             qw_gpu_qk_rope_half_bf16(text->gpu, k, k, k, k, rope_cos_t,
                                      rope_sin_t, seq, QWEN_KV_HEADS_TEXT,
                                      QWEN_TEXT_HEAD_DIM) &&
             dump_stage(text->gpu, dump_dir, i, "k_embed", k, kv_elements);
        ok = ok && h3_gpu_submit(text->gpu) && h3_gpu_begin(text->gpu);
        if (text_profile) {
            prof[1] += pipe_now_text() - g0;
            g0 = pipe_now_text();
        }
        ok = ok && qw_gpu_flash_sdpa_prefix_bf16(text->gpu, attn, q, k, v,
                                                 causal_limit, seq, seq,
                                                 QWEN_TEXT_HEADS,
                                                 QWEN_KV_HEADS_TEXT,
                                                 QWEN_TEXT_HEAD_DIM,
                                                 1.0f / sqrtf(
                                                     (float)QWEN_TEXT_HEAD_DIM)) &&
             dump_stage(text->gpu, dump_dir, i, "attn", attn, elements);
        ok = ok && h3_gpu_submit(text->gpu) && h3_gpu_begin(text->gpu);
        if (text_profile) {
            prof[2] += pipe_now_text() - g0;
            g0 = pipe_now_text();
        }
        ok = ok && run_linear(text->gpu, branch, attn, layer->o_proj, seq,
                              QWEN_TEXT_HIDDEN, QWEN_TEXT_HIDDEN, error,
                              error_size, "o_proj") &&
             dump_stage(text->gpu, dump_dir, i, "o_proj", branch, elements) &&
             h3_gpu_add_bf16(text->gpu, x, x, branch, (uint32_t)elements) &&
             dump_stage(text->gpu, dump_dir, i, "res1", x, elements);
        ok = ok && h3_gpu_submit(text->gpu) && h3_gpu_begin(text->gpu);
        if (text_profile) {
            prof[3] += pipe_now_text() - g0;
            g0 = pipe_now_text();
        }
        ok = ok && qw_gpu_rms_norm_bf16(text->gpu, normed, x, layer->post_norm, seq,
                                        QWEN_TEXT_HIDDEN, 1e-6f) &&
             dump_stage(text->gpu, dump_dir, i, "post_ln", normed,
                        elements) &&
             run_linear(text->gpu, fc1out, normed, layer->fc1, seq,
                        QWEN_TEXT_HIDDEN, 2 * QWEN_TEXT_MLP, error,
                        error_size, "fc1") &&
             dump_stage(text->gpu, dump_dir, i, "fc1out", fc1out,
                        fc1_elements) &&
             qw_gpu_swiglu_bf16(text->gpu, mlp_mid, fc1out,
                                (uint32_t)mlp_elements,
                                QWEN_TEXT_MLP) &&
             dump_stage(text->gpu, dump_dir, i, "act", mlp_mid,
                        mlp_elements) &&
             run_linear(text->gpu, branch, mlp_mid, layer->down, seq,
                        QWEN_TEXT_MLP, QWEN_TEXT_HIDDEN, error, error_size,
                        "down") &&
             dump_stage(text->gpu, dump_dir, i, "mlp", branch, elements) &&
             h3_gpu_add_bf16(text->gpu, x, x, branch, (uint32_t)elements) &&
             dump_stage(text->gpu, dump_dir, i, "out", x, elements);
        ok = ok && h3_gpu_submit(text->gpu) && h3_gpu_begin(text->gpu);
        if (text_profile) {
            prof[4] += pipe_now_text() - g0;
        }
        if (text_timing)
            fprintf(stderr, "[text] layer %d %.3fs\n", i,
                    pipe_now_text() - layer_t0);
        /* DeepStack: after decoder layers 0/1/2 the vision tower's
         * per-level merged features are added at the visual positions
         * (_deepstack_process semantics: hidden[visual] += embeds). */
        if (ok && deepstack_bf16 && i < 3) {
            uint16_t *delta = malloc(sizeof(uint16_t) * elements);
            ok = delta != NULL;
            if (ok) {
                uint32_t vis = 0;
                for (uint32_t s = 0; s < seq; s++) {
                    uint16_t *row = delta + (size_t)s * QWEN_TEXT_HIDDEN;
                    if (visual_mask && visual_mask[s]) {
                        memcpy(row,
                               deepstack_bf16 +
                                   ((size_t)i * n_vis + vis++) *
                                       QWEN_TEXT_HIDDEN,
                               sizeof(uint16_t) * QWEN_TEXT_HIDDEN);
                    } else {
                        memset(row, 0, sizeof(uint16_t) * QWEN_TEXT_HIDDEN);
                    }
                }
                h3_gpu_tensor *delta_t = h3_gpu_tensor_from_bf16(
                    text->gpu, delta, elements);
                ok = delta_t && h3_gpu_add_bf16(text->gpu, x, x, delta_t,
                                                (uint32_t)elements);
                h3_gpu_tensor_free(delta_t);
                free(delta);
            }
            if (!ok) break;
        }
        if (ok && dump_dir && dump_dir[0]) {
            char name[32];
            snprintf(name, sizeof(name), "layer_out_%d", i);
            ok = dump_tensor(text->gpu, dump_dir, name, x, elements);
        }
    }
    if (text_timing)
        fprintf(stderr, "[text] layers+deepstack %.3fs\n",
                pipe_now_text() - text_t0);
    if (text_profile)
        fprintf(stderr,
                "[text] profile: qkv+ln %.2fs rope %.2fs attn %.2fs "
                "o+res %.2fs mlp %.2fs\n",
                prof[0], prof[1], prof[2], prof[3], prof[4]);
    h3_gpu_tensor_free(rope_cos_t);
    h3_gpu_tensor_free(rope_sin_t);
    h3_gpu_tensor_free(causal_limit);
    h3_gpu_tensor_free(normed);
    h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(v);
    h3_gpu_tensor_free(attn);
    h3_gpu_tensor_free(branch);
    h3_gpu_tensor_free(fc1out);
    h3_gpu_tensor_free(mlp_mid);
    if (!ok) return set_error(error, error_size, "encode failed: %s",
                              h3_gpu_error(text->gpu));
    if (!h3_gpu_copy_bf16(text->gpu, hidden, 0, x, 0, elements)) {
        return set_error(error, error_size, "hidden copy failed");
    }
    return 1;
}

int qw_text_encode(qw_text *text, const uint32_t *token_ids, uint32_t seq,
                   h3_gpu_tensor *hidden, char *error, size_t error_size) {
    h3_gpu_tensor *token_t =
        h3_gpu_tensor_from_u32(text->gpu, token_ids, seq);
    if (!token_t) return set_error(error, error_size, "token upload failed");
    h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(
        text->gpu, (size_t)seq * QWEN_TEXT_HIDDEN);
    int ok = x && h3_gpu_embedding_bf16(text->gpu, x, text->embed, token_t,
                                        seq, 151936, QWEN_TEXT_HIDDEN);
    h3_gpu_tensor_free(token_t);
    if (!ok) {
        h3_gpu_tensor_free(x);
        return set_error(error, error_size, "embedding failed: %s",
                         h3_gpu_error(text->gpu));
    }
    int rc = encode_core(text, x, seq, NULL, NULL, NULL, 0, hidden, error,
                         error_size);
    h3_gpu_tensor_free(x);
    return rc;
}

int qw_text_encode_vlm(qw_text *text, const uint16_t *embeds_bf16,
                       const int32_t *pos_ids, const uint8_t *visual_mask,
                       const uint16_t *deepstack_bf16, uint32_t seq,
                       uint32_t n_vis, h3_gpu_tensor *hidden, char *error,
                       size_t error_size) {
    h3_gpu_tensor *x = h3_gpu_tensor_from_bf16(
        text->gpu, embeds_bf16, (size_t)seq * QWEN_TEXT_HIDDEN);
    if (!x) return set_error(error, error_size, "embeds upload failed");
    int rc = encode_core(text, x, seq, pos_ids, visual_mask, deepstack_bf16,
                         n_vis, hidden, error, error_size);
    h3_gpu_tensor_free(x);
    return rc;
}
/* ------------------------------------------------------------------ */
/* ti2i：模板组装 + get_rope_index + 视觉 embeds scatter                */

static uint32_t qw_text_image_pad_id(qw_text *text) {
    if (!text->image_pad_id) {
        uint32_t *ids = NULL;
        size_t count = 0;
        char error[512];
        if (h3_tokenizer_encode(text->tokenizer, "<|image_pad|>", 0, &ids,
                                &count, error, sizeof(error)) &&
            count == 1)
            text->image_pad_id = ids[0];
        if (ids) h3_tokenizer_ids_free(ids);
    }
    return text->image_pad_id;
}

/* Append one text piece's ids to a growable array. Every piece boundary
 * sits between special tokens, so the tokenizer's special-split produces
 * the same BPE chunks as encoding the whole template in one string. */
static int ti2i_append_text(h3_tokenizer *tokenizer, const char *piece,
                            uint32_t **out, size_t *count, size_t *cap,
                            char *error, size_t error_size) {
    uint32_t *piece_ids = NULL;
    size_t piece_count = 0;
    if (!h3_tokenizer_encode(tokenizer, piece, 0, &piece_ids, &piece_count,
                             error, error_size))
        return 0;
    if (*count + piece_count > *cap) {
        size_t need = *count + piece_count;
        uint32_t *grown = realloc(*out, sizeof(uint32_t) * need);
        if (!grown) {
            h3_tokenizer_ids_free(piece_ids);
            return set_error(error, error_size, "out of memory");
        }
        *out = grown;
        *cap = need;
    }
    memcpy(*out + *count, piece_ids, sizeof(uint32_t) * piece_count);
    *count += piece_count;
    h3_tokenizer_ids_free(piece_ids);
    return 1;
}

static int ti2i_append_pads(uint32_t pad_id, size_t n, uint32_t **out,
                            size_t *count, size_t *cap, char *error,
                            size_t error_size) {
    if (*count + n > *cap) {
        uint32_t *grown = realloc(*out, sizeof(uint32_t) * (*count + n));
        if (!grown)
            return set_error(error, error_size, "out of memory");
        *out = grown;
        *cap = *count + n;
    }
    for (size_t i = 0; i < n; i++) (*out)[(*count)++] = pad_id;
    return 1;
}

int qw_text_template_tokens_ti2i(qw_text *text, const char *prompt,
                                 uint32_t n_images, const uint32_t *grid_hw,
                                 uint32_t **ids, size_t *count, size_t *drop,
                                 uint32_t *image_pad_id, char *error,
                                 size_t error_size) {
    if (!text->tokenizer) {
        text->tokenizer = h3_tokenizer_load(text->tokenizer_path, error,
                                            error_size);
        if (!text->tokenizer) {
            return set_error(error, error_size, "tokenizer load failed");
        }
    }
    uint32_t pad_id = qw_text_image_pad_id(text);
    if (image_pad_id) *image_pad_id = pad_id;

    /* user 段（对齐官方 qwenimage21 多图 replace 语义）：每组
     * <imageN><|vision_start|><|image_pad|>×slots<|vision_end|>，组间空格；
     * prompt 在全部图像组之后。pad 长串绝不进 BPE——上千个重复特殊词面
     * 逐个过 tokenizer 曾占 ti2i 编码的 ~7s——文本分片按边界切开后逐段
     * tokenize（边界均为特殊词面，BPE chunk 与整串编码一致），pad_id 按
     * 槽位数直接展开拼进 id 序列。 */
    char system_message[512];
    snprintf(system_message, sizeof(system_message),
             "<|im_start|>system\n%s<|im_end|>\n", QWEN_TEXT_SYS_PROMPT);

    size_t cap = 64 + (size_t)n_images * 32 + strlen(prompt) + 64;
    for (uint32_t i = 0; i < n_images; i++)
        cap += (size_t)grid_hw[i * 2] * grid_hw[i * 2 + 1];
    uint32_t *out = malloc(sizeof(uint32_t) * cap);
    if (!out) return set_error(error, error_size, "out of memory");
    size_t out_count = 0;

    if (!ti2i_append_text(text->tokenizer, system_message, &out, &out_count,
                          &cap, error, error_size)) {
        free(out);
        return 0;
    }
    *drop = out_count; /* system 段词数（官方 _drop_idx 语义） */

    char head[96];
    snprintf(head, sizeof(head), "<|im_start|>user\n<image%u><|vision_start|>",
             1);
    char between[96];
    int piece_ok = 1;
    for (uint32_t i = 0; piece_ok && i < n_images; i++) {
        if (i) {
            snprintf(between, sizeof(between),
                     "<|vision_end|> <image%u><|vision_start|>", i + 1);
            piece_ok = ti2i_append_text(text->tokenizer, between, &out,
                                        &out_count, &cap, error, error_size);
        } else {
            piece_ok = ti2i_append_text(text->tokenizer, head, &out,
                                        &out_count, &cap, error, error_size);
        }
        if (piece_ok)
            piece_ok = ti2i_append_pads(
                pad_id, (size_t)grid_hw[i * 2] * grid_hw[i * 2 + 1], &out,
                &out_count, &cap, error, error_size);
    }
    if (piece_ok) {
        size_t tail_len = strlen(prompt) + 64;
        char *tail = malloc(tail_len);
        if (!tail) {
            free(out);
            return set_error(error, error_size, "out of memory");
        }
        snprintf(tail, tail_len, "<|vision_end|>%s<|im_end|>\n"
                                 "<|im_start|>assistant\n",
                 prompt);
        piece_ok = ti2i_append_text(text->tokenizer, tail, &out, &out_count,
                                    &cap, error, error_size);
        free(tail);
    }
    if (!piece_ok) {
        free(out);
        return 0;
    }
    *ids = out;
    *count = out_count;
    return 1;
}

void qw_text_pos_ids(const uint32_t *ids, size_t count, uint32_t image_pad_id,
                     uint32_t n_images, const uint32_t *grid_hw,
                     int32_t *pos_out) {
    /* 三行 [T, H, W]，每行 count 个。文本段三轴共用递增位置；image 段
     * T 恒定、H/W 为 merged 网格行主序，位置推进 max(gh, gw)。 */
    uint32_t current = 0;
    size_t s = 0;
    uint32_t img = 0;
    while (s < count) {
        if (img < n_images && ids[s] == image_pad_id) {
            uint32_t gh = grid_hw[img * 2], gw = grid_hw[img * 2 + 1];
            size_t run_start = s;
            while (s < count && ids[s] == image_pad_id) s++;
            for (uint32_t r = 0; r < gh; r++)
                for (uint32_t c = 0; c < gw; c++) {
                    pos_out[(size_t)0 * count + run_start + r * gw + c] =
                        (int32_t)current;
                    pos_out[(size_t)1 * count + run_start + r * gw + c] =
                        (int32_t)(current + r);
                    pos_out[(size_t)2 * count + run_start + r * gw + c] =
                        (int32_t)(current + c);
                }
            current += (gh > gw ? gh : gw);
            img++;
            continue;
        }
        pos_out[(size_t)0 * count + s] = (int32_t)current;
        pos_out[(size_t)1 * count + s] = (int32_t)current;
        pos_out[(size_t)2 * count + s] = (int32_t)current;
        current++;
        s++;
    }
}

int qw_text_encode_ti2i(qw_text *text, const char *prompt, uint32_t n_images,
                        const uint32_t *grid_hw, const uint16_t *merged_bf16,
                        const uint16_t *deepstack_bf16, h3_gpu_tensor *hidden,
                        uint8_t *img_mask_out, uint32_t out_rows,
                        size_t *seq_out, char *error, size_t error_size) {
    uint32_t pad_id = 0;
    uint32_t *ids = NULL;
    size_t count = 0, drop = 0;
    if (!qw_text_template_tokens_ti2i(text, prompt, n_images, grid_hw, &ids,
                                      &count, &drop, &pad_id, error,
                                      error_size)) {
        return 0;
    }
    if (count <= drop) {
        h3_tokenizer_ids_free(ids);
        return set_error(error, error_size, "ti2i: empty sequence");
    }
    /* 容量硬校验先于任何写入：hidden 的 blit 与 img_mask_out 的逐字节
     * 写入都以此为界（后者是无容量信息的裸指针，必须在这里拦住）。 */
    if (count - drop > out_rows) {
        h3_tokenizer_ids_free(ids);
        return set_error(error, error_size,
                         "ti2i: sequence %zu rows exceeds capacity %u",
                         count - drop, out_rows);
    }
    uint32_t slots_total = 0;
    for (uint32_t i = 0; i < n_images; i++)
        slots_total += grid_hw[i * 2] * grid_hw[i * 2 + 1];

    /* 官方语义：全长（含 system）过 LLM，输出侧才裁 system 段——mrope
     * 位置必须覆盖 system（drop 后再算会整体偏移 14，rope 相位全错）。 */
    size_t seq = count;
    size_t pads = 0;
    uint8_t *mask = malloc(seq);
    int32_t *pos = malloc(sizeof(int32_t) * 3 * seq);
    if (!pos || !mask) {
        free(pos);
        free(mask);
        h3_tokenizer_ids_free(ids);
        return set_error(error, error_size, "out of memory");
    }
    /* pads==slots 校验必须先于 pos_ids/mask 的写入：qw_text_pos_ids 对
     * 检测到的 pad run 无条件写 gh*gw 个位置，run 长度与槽位数不符时
     * 会越过 run 末尾——先校验才能保证后面的写入安全。 */
    for (size_t s = 0; s < seq; s++)
        if (ids[s] == pad_id) pads++;
    if (pads != slots_total) {
        free(pos);
        free(mask);
        h3_tokenizer_ids_free(ids);
        return set_error(error, error_size,
                         "ti2i: pad count %zu != slots %u", pads,
                         slots_total);
    }
    qw_text_pos_ids(ids, seq, pad_id, n_images, grid_hw, pos);
    if (getenv("QW_DUMP_POS")) {
        FILE *pf = fopen("/tmp/c_pos2.bin", "wb");
        if (pf) {
            fwrite(pos, 4, 3 * seq, pf);
            fclose(pf);
        }
    }
    for (size_t s = 0; s < seq; s++) {
        mask[s] = ids[s] == pad_id ? 1 : 0;
        /* img_mask_out 与 hidden 同布局（drop 后） */
        if (img_mask_out && s >= drop) img_mask_out[s - drop] = mask[s];
    }

    /* embed_tokens 查表 + 视觉槽 scatter（各段连续，段内行序 = merged 序，
     * masked_scatter 语义：按段序将 merged 行写入 image_pad 位置）。
     * 单出口清理：x/token_t/merged_t/full 在任何失败路径都释放。 */
    int ti2i_timing = getenv("QWEN_TEXT_TIMING") != NULL;
    double ti2i_t0 = ti2i_timing ? pipe_now_text() : 0.0;
    h3_gpu_tensor *x = NULL, *token_t = NULL, *merged_t = NULL, *full = NULL;
    int rc = 0;
    x = h3_gpu_tensor_new_bf16(text->gpu, (size_t)seq * QWEN_TEXT_HIDDEN);
    token_t = h3_gpu_tensor_from_u32(text->gpu, ids, seq);
    if (!x || !token_t) {
        set_error(error, error_size, "ti2i: %s alloc failed",
                  x ? "token" : "x");
        goto cleanup;
    }
    if (!hidden) {
        set_error(error, error_size, "ti2i: hidden tensor missing");
        goto cleanup;
    }
    if (!h3_gpu_begin(text->gpu) && !h3_gpu_continue(text->gpu)) {
        set_error(error, error_size, "ti2i: begin failed");
        goto cleanup;
    }
    {
        int ok = h3_gpu_embedding_bf16(text->gpu, x, text->embed, token_t,
                                       (uint32_t)seq, 151936,
                                       QWEN_TEXT_HIDDEN);
        if (ok) {
            ok = h3_gpu_submit(text->gpu) && h3_gpu_begin(text->gpu);
            if (!ok)
                set_error(error, error_size, "ti2i: submit/begin failed: %s",
                          h3_gpu_error(text->gpu));
        } else {
            set_error(error, error_size, "ti2i: embedding failed: %s",
                      h3_gpu_error(text->gpu));
        }
        if (ok) {
            merged_t = h3_gpu_tensor_from_bf16(
                text->gpu, merged_bf16,
                (size_t)slots_total * QWEN_TEXT_HIDDEN);
            if (!merged_t)
                set_error(error, error_size,
                          "ti2i: merged alloc failed (slots=%u)",
                          slots_total);
            ok = merged_t != NULL;
        }
        size_t merged_off = 0, scan = 0;
        for (uint32_t i = 0; ok && i < n_images; i++) {
            uint32_t gh = grid_hw[i * 2], gw = grid_hw[i * 2 + 1];
            size_t slots = (size_t)gh * gw;
            while (scan < seq && !mask[scan]) scan++;
            if (scan >= seq || scan + slots > seq) {
                ok = 0;
                set_error(error, error_size, "ti2i: mask layout mismatch");
                break;
            }
            ok = h3_gpu_copy_bf16(text->gpu, x, scan * QWEN_TEXT_HIDDEN,
                                  merged_t, merged_off * QWEN_TEXT_HIDDEN,
                                  slots * QWEN_TEXT_HIDDEN);
            scan += slots;
            merged_off += slots;
        }
        if (!ok) goto cleanup;
    }
    if (ti2i_timing)
        fprintf(stderr, "[text] tokenize+embed+scatter %.3fs\n",
                pipe_now_text() - ti2i_t0);
    ti2i_t0 = pipe_now_text();
    /* 全长编码 → 输出侧裁掉 system 段（官方 _drop_idx 语义） */
    full = h3_gpu_tensor_new_bf16(text->gpu, (size_t)seq * QWEN_TEXT_HIDDEN);
    if (!full) {
        set_error(error, error_size, "ti2i: full alloc failed");
        goto cleanup;
    }
    rc = encode_core(text, x, (uint32_t)seq, pos, mask, deepstack_bf16,
                     slots_total, full, error, error_size);
    if (ti2i_timing)
        fprintf(stderr, "[text] encode_core+copy %.3fs\n",
                pipe_now_text() - ti2i_t0);
    if (rc)
        rc = h3_gpu_copy_bf16(text->gpu, hidden, 0, full,
                              drop * QWEN_TEXT_HIDDEN,
                              (seq - drop) * QWEN_TEXT_HIDDEN);

cleanup:
    /* pos/mask 是 encode_core 的输入（rope 表构建读它），调用后释放 */
    free(pos);
    free(mask);
    h3_tokenizer_ids_free(ids);
    h3_gpu_tensor_free(token_t);
    h3_gpu_tensor_free(merged_t);
    h3_gpu_tensor_free(full);
    h3_gpu_tensor_free(x);
    if (rc && seq_out) *seq_out = seq - drop;
    return rc;
}
