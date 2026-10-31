#include "qwen_dit.h"

#include "h3_weights.h"
#include "qwen_gpu_ext.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QWEN_DIT_MLP (QWEN_HIDDEN * QWEN_MLP_RATIO)
#define QWEN_DIT_SLOTS 4
#define QWEN_DIT_ATTENTION_SCALE (1.0f / sqrtf((float)QWEN_HEAD_DIM))

/* Block-stage profiler (QWEN_BLOCK_PROFILE=1): wall-clock each stage and
 * force a submit+begin after it. Distorts pipelining — diagnostics only. */
enum {
    QW_BP_INPUT = 0, /* LN0 + qkv GEMMs + head norms + rope */
    QW_BP_ATTN,      /* attention (+ query staging copies) */
    QW_BP_OUT,       /* to_out GEMM + gate */
    QW_BP_LN,        /* second LN+mod */
    QW_BP_MLP,       /* MLP GEMMs + swiglu + gate */
    QW_BP_SLOTS
};
static double qw_bp_acc[QW_BP_SLOTS];
static uint32_t qw_bp_blocks;
static double qw_bp_last;

static double qw_bp_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Returns 0 on GPU failure; otherwise marks the stage boundary: submit
 * and wait first, so the recorded span covers this stage's encode plus
 * its full GPU execution. */
static int qw_bp_mark(h3_gpu *gpu, int slot, int ok) {
    if (!getenv("QWEN_BLOCK_PROFILE")) return ok;
    if (!ok) return 0;
    if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) return 0;
    double now = qw_bp_now();
    qw_bp_acc[slot] += now - qw_bp_last;
    qw_bp_last = now;
    return 1;
}

static void qw_bp_report(void) {
    if (!getenv("QWEN_BLOCK_PROFILE") || !qw_bp_blocks) return;
    static const char *names[QW_BP_SLOTS] = {
        "input(LN0+qkv+rope)", "attention", "to_out+gate", "LN1", "mlp+gate"};
    fprintf(stderr, "[block-profile] per layer/step, %u blocks:\n",
            qw_bp_blocks);
    double total = 0;
    for (int i = 0; i < QW_BP_SLOTS; i++) {
        fprintf(stderr, "  %-20s %7.2f ms\n", names[i],
                qw_bp_acc[i] * 1e3 / qw_bp_blocks);
        total += qw_bp_acc[i];
    }
    fprintf(stderr, "  %-20s %7.2f ms\n", "total(sync)", total * 1e3 / qw_bp_blocks);
    memset(qw_bp_acc, 0, sizeof(qw_bp_acc));
    qw_bp_blocks = 0;
}

typedef struct {
    h3_gpu_tensor *to_q;
    h3_gpu_tensor *to_k;
    h3_gpu_tensor *to_v;
    h3_gpu_tensor *to_out;
    h3_gpu_tensor *norm_q;
    h3_gpu_tensor *norm_k;
    /* Fused [2 * mlp, hidden]: gate_layer rows then proj rows, matching
     * the h3 SwiGLU graph split (silu applied to the first half). */
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
} qw_dit_layer;

/* One stacked LoRA adapter: path, strength, and the safetensors header
 * kept open across rebuilds (the rank is probed once at apply time). */
typedef struct {
    char *path;
    float scale;
    uint32_t rank;
    h3_st_header header;
} qw_lora_adapter;

struct qw_dit {
    h3_gpu *gpu;
    h3_weight_store *store;
    int layers;
    qw_dit_layer *layer;
    h3_gpu_tensor *img_in;
    h3_gpu_tensor *txt_norm;
    h3_gpu_tensor *txt_in1;
    h3_gpu_tensor *txt_in2;
    h3_gpu_tensor *time_l1;
    h3_gpu_tensor *time_l2;
    h3_gpu_tensor *mod_w;
    h3_gpu_tensor *norm_out_w;
    h3_gpu_tensor *proj_out;
    /* Per-forward activation workspace: six [rows, hidden] activations
     * allocated once for the first block call and reused by every layer
     * (re-allocated when rows changes between calls, e.g. t2i vs ti2i
     * in one REPL session). */
    h3_gpu_tensor *ws[6];
    uint32_t ws_rows;
    /* ti2i step cache: cond/text rows never change across sampling
     * steps, so after one full pass their per-layer K/V stay valid;
     * later steps recompute only the target rows. Two slots — slot 0
     * for the guided (positive prompt) branch, slot 1 for the CFG
     * negative branch, which has its own run layout. Invalidated when
     * the run layout changes. */
    struct qw_dit_ti2i_cache {
        h3_gpu_tensor **k;
        h3_gpu_tensor **v;
        h3_gpu_tensor *th;      /* [target_rows, hidden] target workspace */
        h3_gpu_tensor *mod;
        h3_gpu_tensor *branch;
        h3_gpu_tensor *q;
        h3_gpu_tensor *kt;
        h3_gpu_tensor *vt;
        h3_gpu_tensor *attn;
        h3_gpu_tensor *t_lat;   /* [target_rows, z] */
        h3_gpu_tensor *row0;    /* [target_rows] u32, all-zero row map */
        h3_gpu_tensor *rope_cos; /* [target_rows, dim/2] f32 slices */
        h3_gpu_tensor *rope_sin;
        qw_dit_segment *segs;
        uint32_t seg_count;
        uint32_t rows;
        uint32_t target_rows;
    } ti2i_cache[2];
    /* LoRA state: while adapters are applied, layer[] holds the fused
     * private copies and base_layer[] a shadow of the base weights (the
     * mmap tensors simply stay referenced, so rebuilds/clears need no
     * weight I/O). base_layer is NULL when no adapter is applied. The
     * adapter stack keeps its safetensors header open across rebuilds. */
    qw_dit_layer *base_layer;
    qw_lora_adapter *lora;
    int n_lora;
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

static float qw_bits_to_f32(uint16_t bits) {
    uint32_t expanded = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &expanded, sizeof(value));
    return value;
}

static uint16_t qw_f32_to_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

qw_dit *qw_dit_create(h3_gpu *gpu, const char *model_dir, int layers,
                      char *error, size_t error_size) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/transformer", model_dir);
    h3_weight_store *store = h3_weight_store_open(path, error, error_size);
    if (!store) return NULL;

    qw_dit *dit = calloc(1, sizeof(qw_dit));
    if (!dit) {
        h3_weight_store_free(store);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    dit->gpu = gpu;
    dit->store = store;
    dit->layers = layers;

    const uint64_t hidden2[2] = {QWEN_HIDDEN, QWEN_HIDDEN};
    const uint64_t hidden1[2] = {QWEN_HIDDEN, 256};
    const uint64_t hidden_z[2] = {QWEN_HIDDEN, QWEN_Z_DIM};
    const uint64_t z_hidden[2] = {QWEN_Z_DIM, QWEN_HIDDEN};
    const uint64_t mod_hidden[2] = {4 * QWEN_HIDDEN, QWEN_HIDDEN};
    const uint64_t mlp_hidden[2] = {QWEN_DIT_MLP, QWEN_HIDDEN};
    const uint64_t hidden_mlp[2] = {QWEN_HIDDEN, QWEN_DIT_MLP};
    const uint64_t scalar[1] = {QWEN_HIDDEN};
    const uint64_t scalar_dim[1] = {QWEN_HEAD_DIM};

    dit->img_in = h3_weight_load_bf16(store, gpu, "img_in.weight", 2,
                                      hidden_z, error, error_size);
    dit->txt_norm = h3_weight_load_bf16(store, gpu, "txt_in.text_norm.weight",
                                        1, scalar, error, error_size);
    dit->txt_in1 = h3_weight_load_bf16(store, gpu, "txt_in.in_layer.weight", 2,
                                       hidden2, error, error_size);
    dit->txt_in2 = h3_weight_load_bf16(store, gpu, "txt_in.out_layer.weight",
                                       2, hidden2, error, error_size);
    dit->time_l1 = h3_weight_load_bf16(
        store, gpu, "time_text_embed.timestep_embedder.linear_1.weight", 2,
        hidden1, error, error_size);
    dit->time_l2 = h3_weight_load_bf16(
        store, gpu, "time_text_embed.timestep_embedder.linear_2.weight", 2,
        hidden2, error, error_size);
    dit->mod_w = h3_weight_load_bf16(store, gpu, "modulation.1.weight", 2,
                                     mod_hidden, error, error_size);
    dit->norm_out_w = h3_weight_load_bf16(store, gpu, "norm_out.linear.weight",
                                          2, hidden2, error, error_size);
    dit->proj_out = h3_weight_load_bf16(store, gpu, "proj_out.weight", 2,
                                        z_hidden, error, error_size);
    if (!dit->img_in || !dit->txt_norm || !dit->txt_in1 || !dit->txt_in2 ||
        !dit->time_l1 || !dit->time_l2 || !dit->mod_w || !dit->norm_out_w ||
        !dit->proj_out) {
        qw_dit_free(dit);
        return NULL;
    }

    dit->layer = calloc((size_t)layers, sizeof(qw_dit_layer));
    if (!dit->layer) {
        qw_dit_free(dit);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    for (int i = 0; i < layers; i++) {
        char name[160];
        qw_dit_layer *layer = &dit->layer[i];
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_q.weight",
                 i);
        layer->to_q = h3_weight_load_bf16(store, gpu, name, 2, hidden2, error,
                                          error_size);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_k.weight",
                 i);
        layer->to_k = h3_weight_load_bf16(store, gpu, name, 2, hidden2, error,
                                          error_size);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_v.weight",
                 i);
        layer->to_v = h3_weight_load_bf16(store, gpu, name, 2, hidden2, error,
                                          error_size);
        snprintf(name, sizeof(name),
                 "transformer_blocks.%d.attn.to_out.0.weight", i);
        layer->to_out = h3_weight_load_bf16(store, gpu, name, 2, hidden2,
                                            error, error_size);
        snprintf(name, sizeof(name),
                 "transformer_blocks.%d.attn.norm_q.weight", i);
        layer->norm_q = h3_weight_load_bf16(store, gpu, name, 1, scalar_dim,
                                            error, error_size);
        snprintf(name, sizeof(name),
                 "transformer_blocks.%d.attn.norm_k.weight", i);
        layer->norm_k = h3_weight_load_bf16(store, gpu, name, 1, scalar_dim,
                                            error, error_size);
        snprintf(name, sizeof(name),
                 "transformer_blocks.%d.img_mlp.gate_layer.weight", i);
        h3_gpu_tensor *gate = h3_weight_load_bf16(store, gpu, name, 2,
                                                  mlp_hidden, error,
                                                  error_size);
        snprintf(name, sizeof(name),
                 "transformer_blocks.%d.img_mlp.proj.weight", i);
        h3_gpu_tensor *proj = h3_weight_load_bf16(store, gpu, name, 2,
                                                  mlp_hidden, error,
                                                  error_size);
        snprintf(name, sizeof(name),
                 "transformer_blocks.%d.img_mlp.out.weight", i);
        layer->fc2 = h3_weight_load_bf16(store, gpu, name, 2, hidden_mlp,
                                         error, error_size);
        if (!layer->to_q || !layer->to_k || !layer->to_v || !layer->to_out ||
            !layer->norm_q || !layer->norm_k || !gate || !proj ||
            !layer->fc2) {
            h3_gpu_tensor_free(gate);
            h3_gpu_tensor_free(proj);
            qw_dit_free(dit);
            return NULL;
        }

        /* Concatenate [gate; proj] rows into the fused fc1. */
        layer->fc1 = h3_gpu_tensor_new_bf16(
            gpu, (size_t)2 * QWEN_DIT_MLP * QWEN_HIDDEN);
        if (!layer->fc1) {
            h3_gpu_tensor_free(gate);
            h3_gpu_tensor_free(proj);
            qw_dit_free(dit);
            set_error(error, error_size, "fc1 allocation failed");
            return NULL;
        }
        size_t half = (size_t)QWEN_DIT_MLP * QWEN_HIDDEN;
        uint16_t *host = malloc(sizeof(uint16_t) * half);
        int ok = host && h3_gpu_tensor_read_bf16(gate, host, half) &&
                 h3_gpu_tensor_write_bf16_range(layer->fc1, 0, host, half) &&
                 h3_gpu_tensor_read_bf16(proj, host, half) &&
                 h3_gpu_tensor_write_bf16_range(layer->fc1, half, host, half);
        free(host);
        h3_gpu_tensor_free(gate);
        h3_gpu_tensor_free(proj);
        if (!ok) {
            qw_dit_free(dit);
            set_error(error, error_size, "fc1 concat failed at layer %d", i);
            return NULL;
        }
    }
    return dit;
}

static void ti2i_cache_free(qw_dit *dit, uint32_t slot) {
    struct qw_dit_ti2i_cache *c = &dit->ti2i_cache[slot];
    /* Either array can be NULL on its own: ti2i_cache_alloc does two
     * consecutive callocs, and an OOM between them leaves k set and v
     * NULL (or vice versa via the failure path below). */
    if (c->k || c->v) {
        for (int i = 0; i < dit->layers; i++) {
            if (c->k) h3_gpu_tensor_free(c->k[i]);
            if (c->v) h3_gpu_tensor_free(c->v[i]);
        }
        free(c->k);
        free(c->v);
        c->k = NULL;
        c->v = NULL;
    }
    h3_gpu_tensor_free(c->th);
    h3_gpu_tensor_free(c->mod);
    h3_gpu_tensor_free(c->branch);
    h3_gpu_tensor_free(c->q);
    h3_gpu_tensor_free(c->kt);
    h3_gpu_tensor_free(c->vt);
    h3_gpu_tensor_free(c->attn);
    h3_gpu_tensor_free(c->t_lat);
    h3_gpu_tensor_free(c->row0);
    h3_gpu_tensor_free(c->rope_cos);
    h3_gpu_tensor_free(c->rope_sin);
    c->th = NULL;
    c->mod = NULL;
    c->branch = NULL;
    c->q = NULL;
    c->kt = NULL;
    c->vt = NULL;
    c->attn = NULL;
    c->t_lat = NULL;
    c->row0 = NULL;
    c->rope_cos = NULL;
    c->rope_sin = NULL;
    free(c->segs);
    c->segs = NULL;
    c->seg_count = 0;
    c->rows = 0;
    c->target_rows = 0;
}

static int ti2i_segs_match(const qw_dit_segment *a, const qw_dit_segment *b,
                           uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (a[i].rows != b[i].rows || a[i].is_image != b[i].is_image)
            return 0;
    }
    return 1;
}

void qw_dit_ti2i_cache_clear(qw_dit *dit) {
    if (!dit) return;
    ti2i_cache_free(dit, 0);
    ti2i_cache_free(dit, 1);
}

/* Allocate the cross-step cache for one ti2i run layout. */
static int ti2i_cache_alloc(qw_dit *dit, uint32_t slot, uint32_t rows,
                            uint32_t target_rows,
                            const qw_dit_segment *segs, uint32_t seg_count,
                            char *error, size_t error_size) {
    ti2i_cache_free(dit, slot);
    struct qw_dit_ti2i_cache *c = &dit->ti2i_cache[slot];
    c->k = calloc((size_t)dit->layers, sizeof(h3_gpu_tensor *));
    c->v = calloc((size_t)dit->layers, sizeof(h3_gpu_tensor *));
    int ok = c->k && c->v;
    for (int i = 0; ok && i < dit->layers; i++) {
        c->k[i] = h3_gpu_tensor_new_bf16(dit->gpu,
                                         (size_t)rows * QWEN_HIDDEN);
        c->v[i] = h3_gpu_tensor_new_bf16(dit->gpu,
                                         (size_t)rows * QWEN_HIDDEN);
        ok = c->k[i] && c->v[i];
    }
    h3_gpu_tensor *row0 = NULL;
    if (ok) {
        uint32_t *zeros = calloc(target_rows, sizeof(uint32_t));
        row0 = zeros ? h3_gpu_tensor_from_u32(dit->gpu, zeros, target_rows)
                     : NULL;
        free(zeros);
        ok = row0 != NULL;
    }
    c->th = ok ? h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)target_rows * QWEN_HIDDEN) : NULL;
    c->mod = ok ? h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)target_rows * QWEN_HIDDEN) : NULL;
    c->branch = ok ? h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)target_rows * QWEN_HIDDEN) : NULL;
    c->q = ok ? h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)target_rows * QWEN_HIDDEN) : NULL;
    c->kt = ok ? h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)target_rows * QWEN_HIDDEN) : NULL;
    c->vt = ok ? h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)target_rows * QWEN_HIDDEN) : NULL;
    c->attn = ok ? h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)target_rows * QWEN_HIDDEN) : NULL;
    c->t_lat = ok ? h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)target_rows * QWEN_Z_DIM) : NULL;
    c->rope_cos = ok ? h3_gpu_tensor_new_f32(
        dit->gpu, (size_t)target_rows * (QWEN_HEAD_DIM / 2)) : NULL;
    c->rope_sin = ok ? h3_gpu_tensor_new_f32(
        dit->gpu, (size_t)target_rows * (QWEN_HEAD_DIM / 2)) : NULL;
    c->segs = ok ? malloc(sizeof(qw_dit_segment) * seg_count) : NULL;
    ok = ok && c->th && c->mod && c->branch && c->q && c->kt && c->vt &&
         c->attn && c->t_lat && c->rope_cos && c->rope_sin && c->segs;
    if (!ok) {
        h3_gpu_tensor_free(row0);
        ti2i_cache_free(dit, slot);
        return set_error(error, error_size, "ti2i cache alloc failed");
    }
    c->row0 = row0;
    memcpy(c->segs, segs, sizeof(qw_dit_segment) * seg_count);
    c->seg_count = seg_count;
    c->rows = rows;
    c->target_rows = target_rows;
    return 1;
}

void qw_dit_free(qw_dit *dit) {
    if (!dit) return;
    ti2i_cache_free(dit, 0);
    ti2i_cache_free(dit, 1);
    /* A pending LoRA shadows the base weights in base_layer[] (fields the
     * adapter never touches, e.g. norm_q/norm_k, alias between the two
     * arrays) — restore first so every tensor is freed exactly once. */
    if (dit->base_layer) qw_dit_clear_lora(dit, NULL, 0);
    if (dit->layer) {
        for (int i = 0; i < dit->layers; i++) {
            h3_gpu_tensor_free(dit->layer[i].to_q);
            h3_gpu_tensor_free(dit->layer[i].to_k);
            h3_gpu_tensor_free(dit->layer[i].to_v);
            h3_gpu_tensor_free(dit->layer[i].to_out);
            h3_gpu_tensor_free(dit->layer[i].norm_q);
            h3_gpu_tensor_free(dit->layer[i].norm_k);
            h3_gpu_tensor_free(dit->layer[i].fc1);
            h3_gpu_tensor_free(dit->layer[i].fc2);
        }
        free(dit->layer);
    }
    if (dit->base_layer) {
        /* Unreachable with the restore above (clear clears the shadow);
         * kept as a safety net for future state changes. */
        free(dit->base_layer);
        dit->base_layer = NULL;
    }
    h3_gpu_tensor_free(dit->img_in);
    h3_gpu_tensor_free(dit->txt_norm);
    h3_gpu_tensor_free(dit->txt_in1);
    h3_gpu_tensor_free(dit->txt_in2);
    h3_gpu_tensor_free(dit->time_l1);
    h3_gpu_tensor_free(dit->time_l2);
    h3_gpu_tensor_free(dit->mod_w);
    h3_gpu_tensor_free(dit->norm_out_w);
    h3_gpu_tensor_free(dit->proj_out);
    for (int i = 0; i < 6; i++) h3_gpu_tensor_free(dit->ws[i]);
    h3_weight_store_free(dit->store);
    free(dit);
}

/* ------------------------------ LoRA ------------------------------ */

static uint16_t qw_f32_to_bits_rne(float value) {
    /* round-to-nearest-even, matching PyTorch's f32 -> bf16 cast */
    uint32_t bits;
    memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* Per-block segments a LoRA adapter can target. Each fuses scale * (B @ A)
 * into rows x cols elements at `offset` of one layer field. Fields:
 * 0=to_q 1=to_k 2=to_v 3=to_out 4=fc1 (gate rows, then proj rows) 5=fc2.
 * The LoRA keys are the checkpoint keys with the trailing ".weight"
 * replaced by ".lora_A/.lora_B.default.weight". */
typedef struct {
    const char *module;
    int field;
    uint32_t rows, cols;
    size_t offset;
} lora_seg;

static const lora_seg LORA_SEGS[] = {
    {"attn.to_q",          0, QWEN_HIDDEN,  QWEN_HIDDEN,  0},
    {"attn.to_k",          1, QWEN_HIDDEN,  QWEN_HIDDEN,  0},
    {"attn.to_v",          2, QWEN_HIDDEN,  QWEN_HIDDEN,  0},
    {"attn.to_out.0",      3, QWEN_HIDDEN,  QWEN_HIDDEN,  0},
    {"img_mlp.gate_layer", 4, QWEN_DIT_MLP, QWEN_HIDDEN,  0},
    {"img_mlp.proj",       4, QWEN_DIT_MLP, QWEN_HIDDEN,
     (size_t)QWEN_DIT_MLP * QWEN_HIDDEN},
    {"img_mlp.out",        5, QWEN_HIDDEN,  QWEN_DIT_MLP, 0},
};
#define LORA_SEG_COUNT 7
#define LORA_FIELD_COUNT 6

static h3_gpu_tensor **lora_field(qw_dit_layer *layer, int field) {
    switch (field) {
    case 0: return &layer->to_q;
    case 1: return &layer->to_k;
    case 2: return &layer->to_v;
    case 3: return &layer->to_out;
    case 4: return &layer->fc1;
    case 5: return &layer->fc2;
    }
    return NULL;
}

/* Read one LoRA matrix (bf16, 2-D) from the adapter header. Returns 1 and
 * sets *bits, 0 when the key is absent (partial adapters are allowed),
 * -1 on a broken entry. */
static int lora_read_matrix(const h3_st_header *header, const char *key,
                            uint64_t rows, uint64_t cols, uint16_t **bits,
                            char *error, size_t error_size) {
    *bits = NULL;
    const h3_st_tensor *tensor = h3_st_find(header, key);
    if (!tensor) return 0;
    if (tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != cols) {
        set_error(error, error_size,
                  "lora tensor %s: expected bf16 [%llu x %llu]",
                  key, (unsigned long long)rows, (unsigned long long)cols);
        return -1;
    }
    uint64_t elements = h3_st_tensor_elements(tensor);
    uint16_t *data = malloc(sizeof(uint16_t) * elements);
    if (!data) {
        set_error(error, error_size, "out of memory");
        return -1;
    }
    if (!h3_st_read_data(header, tensor, data, sizeof(uint16_t) * elements,
                         error, error_size)) {
        free(data);
        return -1;
    }
    *bits = data;
    return 1;
}

/* Accumulate scale * (B @ A) into an fp32 buffer [out, in].
 * B is [out, rank], A is [rank, in], both raw bf16 bits. The product
 * accumulates in fp32 (Accelerate sgemm); the caller rounds the final
 * sum to bf16 with round-to-nearest-even once, after every adapter has
 * contributed. */
static int lora_delta_into_host(float *dw, uint32_t out, uint32_t in,
                                const uint16_t *a_bits,
                                const uint16_t *b_bits, uint32_t rank,
                                float scale) {
    size_t a_count = (size_t)rank * in;
    size_t b_count = (size_t)out * rank;
    float *a = malloc(sizeof(float) * a_count);
    float *b = malloc(sizeof(float) * b_count);
    if (!a || !b) {
        free(a);
        free(b);
        return 0;
    }
    for (size_t i = 0; i < a_count; i++) a[i] = qw_bits_to_f32(a_bits[i]);
    for (size_t i = 0; i < b_count; i++) b[i] = qw_bits_to_f32(b_bits[i]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)out, (int)in, (int)rank, scale,
                b, (int)rank, a, (int)in, 0.0f, dw, (int)in);
    free(a);
    free(b);
    return 1;
}

/* Rank of an adapter: the first bf16 2-D lora_A matrix in the file. */
static uint32_t lora_probe_rank(const h3_st_header *header) {
    for (size_t i = 0; i < header->tensor_count; i++) {
        const h3_st_tensor *t = &header->tensors[i];
        if (t->dtype == H3_DTYPE_BF16 && t->ndim == 2 &&
            strstr(t->name, ".lora_A.default.weight"))
            return (uint32_t)t->shape[0];
    }
    return 0;
}

/* Drop every adapter and restore the base weights (freeing the fused
 * copies). No error reporting — also used by qw_dit_free and as the
 * rebuild-failure fallback. */
static void lora_reset_state(qw_dit *dit) {
    if (dit->base_layer) {
        for (int i = 0; i < dit->layers; i++) {
            for (int f = 0; f < LORA_FIELD_COUNT; f++) {
                h3_gpu_tensor **cur = lora_field(&dit->layer[i], f);
                h3_gpu_tensor **shadow = lora_field(&dit->base_layer[i], f);
                if (*cur != *shadow) {
                    h3_gpu_tensor_free(*cur);
                    *cur = *shadow;
                }
            }
        }
        free(dit->base_layer);
        dit->base_layer = NULL;
    }
    for (int i = 0; i < dit->n_lora; i++) {
        free(dit->lora[i].path);
        h3_st_free_header(&dit->lora[i].header);
    }
    free(dit->lora);
    dit->lora = NULL;
    dit->n_lora = 0;
    ti2i_cache_free(dit, 0);
    ti2i_cache_free(dit, 1);
}

/* Rebuild every fused copy from the shadow base in one pass:
 * W' = RNE_bf16(fp32(W) + Σ_a scale_a · B_a@A_a). Order-independent and
 * free of rounding accumulation. On failure the whole adapter state is
 * reset to none (the error carries the cause).
 *
 * Synchronization contract: the rebuild writes fused weights through
 * h3_gpu_tensor_write_bf16 — a raw host memcpy into shared storage with
 * no GPU sync. It must therefore run only while NO encoded command
 * references the tensors, i.e. at a request boundary after the previous
 * generation's submit drained (every current caller: apply/switch/remove
 * via main.c before any forward of the request). In-place writes only
 * ever hit the private fused copies (`*slot != base_t`), never the
 * mmap'd/shadow base weights. */
static int lora_rebuild(qw_dit *dit, char *error, size_t error_size) {
    int ok = 1;
    unsigned skipped = 0;
    unsigned in_place = 0, fresh = 0;
    for (int i = 0; ok && i < dit->layers; i++) {
        for (int f = 0; ok && f < LORA_FIELD_COUNT; f++) {
            const lora_seg *segs[2];
            int n_seg = 0;
            for (int s = 0; s < LORA_SEG_COUNT; s++)
                if (LORA_SEGS[s].field == f) segs[n_seg++] = &LORA_SEGS[s];
            h3_gpu_tensor *base_t = *lora_field(&dit->base_layer[i], f);
            h3_gpu_tensor **slot = lora_field(&dit->layer[i], f);
            size_t count = h3_gpu_tensor_elements(base_t);
            size_t dw_max = 0;
            for (int s = 0; s < n_seg; s++) {
                size_t n = (size_t)segs[s]->rows * segs[s]->cols;
                if (n > dw_max) dw_max = n;
            }
            float *acc = malloc(sizeof(float) * count);
            uint16_t *out16 = malloc(sizeof(uint16_t) * count);
            float *dw = malloc(sizeof(float) * dw_max);
            if (!acc || !out16 || !dw ||
                !h3_gpu_tensor_read_bf16(base_t, out16, count)) {
                set_error(error, error_size, "lora rebuild: out of memory");
                ok = 0;
            }
            if (ok)
                for (size_t c = 0; c < count; c++)
                    acc[c] = qw_bits_to_f32(out16[c]);
            for (int a = 0; ok && a < dit->n_lora; a++) {
                const qw_lora_adapter *adapter = &dit->lora[a];
                for (int s = 0; ok && s < n_seg; s++) {
                    const lora_seg *seg = segs[s];
                    char key_a[192], key_b[192];
                    snprintf(key_a, sizeof(key_a),
                             "transformer_blocks.%d.%s.lora_A.default.weight",
                             i, seg->module);
                    snprintf(key_b, sizeof(key_b),
                             "transformer_blocks.%d.%s.lora_B.default.weight",
                             i, seg->module);
                    uint16_t *ab = NULL, *bb = NULL;
                    int ra = lora_read_matrix(&adapter->header, key_a,
                                              adapter->rank, seg->cols, &ab,
                                              error, error_size);
                    if (ra < 0) {
                        ok = 0;
                        break;
                    }
                    if (!ra) {
                        skipped++;
                        continue;
                    }
                    int rb = lora_read_matrix(&adapter->header, key_b,
                                              seg->rows, adapter->rank, &bb,
                                              error, error_size);
                    if (rb < 0) {
                        free(ab);
                        ok = 0;
                        break;
                    }
                    if (!rb) {
                        free(ab);
                        skipped++;
                        continue;
                    }
                    if (!lora_delta_into_host(dw, seg->rows, seg->cols, ab,
                                              bb, adapter->rank,
                                              adapter->scale)) {
                        set_error(error, error_size,
                                  "lora rebuild: out of memory");
                        free(ab);
                        free(bb);
                        ok = 0;
                        break;
                    }
                    free(ab);
                    free(bb);
                    size_t n = (size_t)seg->rows * seg->cols;
                    for (size_t c = 0; c < n; c++)
                        acc[c + seg->offset] += dw[c];
                }
            }
            if (ok) {
                for (size_t c = 0; c < count; c++)
                    out16[c] = qw_f32_to_bits_rne(acc[c]);
                if (*slot != base_t) {
                    /* Rebuild in place: fused tensors have fixed shapes,
                     * so scale changes / adapter switches need no fresh
                     * allocation. Allocating a new copy per field (freeing
                     * the old only afterwards) made the Metal pool ratchet
                     * by a full fused set (~14 GB) on every rebuild — the
                     * new field could never reuse the old field's block. */
                    if (!h3_gpu_tensor_write_bf16(*slot, out16, count)) {
                        set_error(error, error_size,
                                  "lora rebuild: cannot store fused weights");
                        ok = 0;
                    } else {
                        in_place++;
                    }
                } else {
                    /* First apply: layer slots alias the mmap base
                     * tensors (read-only) — allocate the private fused
                     * copy and swap it in; the shadow keeps the base. */
                    h3_gpu_tensor *fused =
                        h3_gpu_tensor_new_bf16(dit->gpu, count);
                    if (!fused ||
                        !h3_gpu_tensor_write_bf16(fused, out16, count)) {
                        h3_gpu_tensor_free(fused);
                        set_error(error, error_size,
                                  "lora rebuild: cannot store fused weights");
                        ok = 0;
                    } else {
                        *slot = fused;
                        fresh++;
                    }
                }
            }
            free(acc);
            free(out16);
            free(dw);
        }
    }
    if (!ok) {
        lora_reset_state(dit);
        return 0;
    }
    if (skipped)
        fprintf(stderr, "[lora] %u target slots absent (partial adapter)\n",
                skipped);
    fprintf(stderr, "[lora] rebuild: %u in-place, %u fresh alloc\n", in_place,
            fresh);
    ti2i_cache_free(dit, 0);
    ti2i_cache_free(dit, 1);
    return 1;
}

int qw_dit_apply_lora(qw_dit *dit, const char *path, float scale,
                      char *error, size_t error_size) {
    if (!dit || !path || !*path)
        return set_error(error, error_size, "lora path is required");

    /* Existing adapter: scale update + rebuild (identical → no-op). */
    for (int i = 0; i < dit->n_lora; i++) {
        if (strcmp(dit->lora[i].path, path) == 0) {
            if (dit->lora[i].scale == scale) return 1;
            dit->lora[i].scale = scale;
            return lora_rebuild(dit, error, error_size);
        }
    }

    if (dit->n_lora >= QW_DIT_MAX_LORA)
        return set_error(error, error_size, "at most %d LoRA adapters",
                         QW_DIT_MAX_LORA);

    /* Read the adapter first: a broken file must leave the stack
     * untouched. */
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return 0;
    uint32_t rank = lora_probe_rank(&header);
    if (!rank) {
        h3_st_free_header(&header);
        return set_error(error, error_size,
                         "lora file carries no bf16 lora_A matrices: %s",
                         path);
    }
    /* Match probe: reject adapters that touch no checkpoint tensor
     * instead of burning a rebuild on them. */
    unsigned hits = 0;
    for (int i = 0; i < dit->layers && !hits; i++)
        for (int s = 0; s < LORA_SEG_COUNT; s++) {
            char key[192];
            snprintf(key, sizeof(key),
                     "transformer_blocks.%d.%s.lora_A.default.weight", i,
                     LORA_SEGS[s].module);
            if (h3_st_find(&header, key)) {
                hits = 1;
                break;
            }
        }
    if (!hits) {
        h3_st_free_header(&header);
        return set_error(error, error_size,
                         "lora matched no checkpoint tensors: %s", path);
    }

    char *path_copy = strdup(path);
    qw_lora_adapter *grown =
        realloc(dit->lora,
                sizeof(qw_lora_adapter) * (size_t)(dit->n_lora + 1));
    if (!path_copy || !grown) {
        free(path_copy);
        h3_st_free_header(&header);
        return set_error(error, error_size, "out of memory");
    }
    dit->lora = grown;
    dit->lora[dit->n_lora].path = path_copy;
    dit->lora[dit->n_lora].scale = scale;
    dit->lora[dit->n_lora].rank = rank;
    dit->lora[dit->n_lora].header = header;
    dit->n_lora++;

    /* First adapter: snapshot the base weights into the shadow. */
    if (dit->n_lora == 1) {
        dit->base_layer = malloc(sizeof(qw_dit_layer) * (size_t)dit->layers);
        if (!dit->base_layer) {
            lora_reset_state(dit);
            return set_error(error, error_size, "out of memory");
        }
        memcpy(dit->base_layer, dit->layer,
               sizeof(qw_dit_layer) * (size_t)dit->layers);
    }
    return lora_rebuild(dit, error, error_size);
}

/* Single-adapter switch: swap the stack entry in place. The base shadow
 * and the fused tensors survive, so the rebuild reuses the existing
 * buffers — a clear+apply round-trip allocated a full fresh fused set
 * (~14 GB of new pool pages) on every switch. Validation runs first: a
 * broken file leaves the old adapter active. */
int qw_dit_switch_lora(qw_dit *dit, const char *path, float scale,
                       char *error, size_t error_size) {
    if (!dit || !path || !*path)
        return set_error(error, error_size, "lora path is required");
    if (dit->n_lora != 1)
        return set_error(error, error_size,
                         "switch requires exactly one active adapter");
    if (strcmp(dit->lora[0].path, path) == 0)
        return qw_dit_apply_lora(dit, path, scale, error, error_size);

    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return 0;
    uint32_t rank = lora_probe_rank(&header);
    if (!rank) {
        h3_st_free_header(&header);
        return set_error(error, error_size,
                         "lora file carries no bf16 lora_A matrices: %s",
                         path);
    }
    unsigned hits = 0;
    for (int i = 0; i < dit->layers && !hits; i++)
        for (int s = 0; s < LORA_SEG_COUNT; s++) {
            char key[192];
            snprintf(key, sizeof(key),
                     "transformer_blocks.%d.%s.lora_A.default.weight", i,
                     LORA_SEGS[s].module);
            if (h3_st_find(&header, key)) {
                hits = 1;
                break;
            }
        }
    if (!hits) {
        h3_st_free_header(&header);
        return set_error(error, error_size,
                         "lora matched no checkpoint tensors: %s", path);
    }
    /* strdup first: on failure nothing has been mutated, so the error
     * return leaves the old adapter fully intact (freeing the old path
     * before the copy would leave a dangling entry for lora_reset_state
     * to double-free). */
    char *path_copy = strdup(path);
    if (!path_copy) return set_error(error, error_size, "out of memory");
    free(dit->lora[0].path);
    h3_st_free_header(&dit->lora[0].header);
    dit->lora[0].path = path_copy;
    dit->lora[0].scale = scale;
    dit->lora[0].rank = rank;
    dit->lora[0].header = header;
    return lora_rebuild(dit, error, error_size);
}

int qw_dit_remove_lora(qw_dit *dit, const char *path,
                       char *error, size_t error_size) {
    if (!dit) return set_error(error, error_size, "no dit");
    if (!path || !*path) return qw_dit_clear_lora(dit, error, error_size);
    int index = -1;
    for (int i = 0; i < dit->n_lora; i++) {
        if (strcmp(dit->lora[i].path, path) == 0) {
            index = i;
            break;
        }
    }
    if (index < 0) return 1; /* unknown path: nothing to remove */
    free(dit->lora[index].path);
    h3_st_free_header(&dit->lora[index].header);
    memmove(&dit->lora[index], &dit->lora[index + 1],
            sizeof(qw_lora_adapter) * (size_t)(dit->n_lora - index - 1));
    dit->n_lora--;
    if (!dit->n_lora) {
        lora_reset_state(dit);
        return 1;
    }
    return lora_rebuild(dit, error, error_size);
}

int qw_dit_clear_lora(qw_dit *dit, char *error, size_t error_size) {
    if (!dit) return set_error(error, error_size, "no dit");
    if (!dit->n_lora) return 1;
    lora_reset_state(dit);
    return 1;
}

int qw_dit_lora_count(const qw_dit *dit) {
    return dit ? dit->n_lora : 0;
}

const char *qw_dit_lora_path_at(const qw_dit *dit, int index) {
    if (!dit || index < 0 || index >= dit->n_lora) return NULL;
    return dit->lora[index].path;
}

float qw_dit_lora_scale_at(const qw_dit *dit, int index) {
    if (!dit || index < 0 || index >= dit->n_lora) return 0.0f;
    return dit->lora[index].scale;
}

const h3_gpu_tensor *qw_dit_block_tensor(const qw_dit *dit, int block,
                                         int field) {
    if (!dit || !dit->layer || block < 0 || block >= dit->layers) return NULL;
    h3_gpu_tensor **slot = lora_field(&dit->layer[block], field);
    return slot ? *slot : NULL;
}

static int run_linear(qw_dit *dit, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                      char *error, size_t error_size, const char *label) {
    if (!h3_gpu_linear_bf16(dit->gpu, output, input, weight, NULL, rows,
                            input_dim, output_dim)) {
        return set_error(error, error_size, "%s: linear failed: %s", label,
                         h3_gpu_error(dit->gpu));
    }
    return 1;
}

int qw_dit_embed(qw_dit *dit, const h3_gpu_tensor *text_in, uint32_t text_rows,
                 const h3_gpu_tensor *latents, uint32_t image_rows,
                 h3_gpu_tensor *joint, char *error, size_t error_size) {
    uint32_t rows = text_rows + image_rows;
    if (h3_gpu_tensor_elements(joint) < (size_t)rows * QWEN_HIDDEN) {
        return set_error(error, error_size, "joint tensor too small");
    }
    if (text_rows) {
        h3_gpu_tensor *normed = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)text_rows * QWEN_HIDDEN);
        h3_gpu_tensor *wide = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)text_rows * QWEN_HIDDEN);
        h3_gpu_tensor *embedded = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)text_rows * QWEN_HIDDEN);
        int ok = normed && wide && embedded;
        if (ok) {
            ok = qw_gpu_rms_norm_one_bf16(dit->gpu, normed, text_in,
                                          dit->txt_norm, text_rows,
                                          QWEN_HIDDEN, 1e-6f) &&
                 run_linear(dit, wide, normed, dit->txt_in1, text_rows,
                            QWEN_HIDDEN, QWEN_HIDDEN, error, error_size,
                            "txt_in1") &&
                 h3_gpu_gelu_bf16(
                     dit->gpu, wide, wide,
                     (uint32_t)((size_t)text_rows * QWEN_HIDDEN), 1) &&
                 run_linear(dit, embedded, wide, dit->txt_in2, text_rows,
                            QWEN_HIDDEN, QWEN_HIDDEN, error, error_size,
                            "txt_in2") &&
                 h3_gpu_copy_bf16(dit->gpu, joint, 0, embedded, 0,
                                  (size_t)text_rows * QWEN_HIDDEN);
        }
        h3_gpu_tensor_free(normed);
        h3_gpu_tensor_free(wide);
        h3_gpu_tensor_free(embedded);
        if (!ok) return set_error(error, error_size, "txt_in failed: %s",
                                  h3_gpu_error(dit->gpu));
    }
    if (image_rows) {
        h3_gpu_tensor *embedded = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)image_rows * QWEN_HIDDEN);
        int ok = embedded &&
                 run_linear(dit, embedded, latents, dit->img_in, image_rows,
                            QWEN_Z_DIM, QWEN_HIDDEN, error, error_size,
                            "img_in") &&
                 h3_gpu_copy_bf16(dit->gpu, joint,
                                  (size_t)text_rows * QWEN_HIDDEN, embedded,
                                  0, (size_t)image_rows * QWEN_HIDDEN);
        h3_gpu_tensor_free(embedded);
        if (!ok) return set_error(error, error_size, "img_in failed: %s",
                                  h3_gpu_error(dit->gpu));
    }
    return 1;
}

void qw_dit_timestep_features(const float *row_sigmas, size_t count,
                              float *features) {
    for (size_t r = 0; r < count; r++) {
        double raw = (double)row_sigmas[r] * 1000.0;
        for (int i = 0; i < 128; i++) {
            double frequency = pow(10000.0, -(double)i / 128.0);
            double argument = raw * frequency;
            features[r * 256 + i] = (float)cos(argument);
            features[r * 256 + 128 + i] = (float)sin(argument);
        }
    }
}

int qw_dit_conditioning(qw_dit *dit, const h3_gpu_tensor *features,
                        uint32_t mod_rows, h3_gpu_tensor *temb,
                        h3_gpu_tensor *modulation,
                        h3_gpu_tensor *norm_out_scale, char *error,
                        size_t error_size) {
    h3_gpu_tensor *wide = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)mod_rows * QWEN_HIDDEN);
    h3_gpu_tensor *silu_temb = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)mod_rows * QWEN_HIDDEN);
    /* The released TimestepEmbedder casts the F32 sinusoidal features to
     * BF16 before the first projection. */
    h3_gpu_tensor *features_bf16 = h3_gpu_tensor_new_bf16(dit->gpu,
                                                          (size_t)mod_rows * 256);
    int ok = wide && silu_temb && features_bf16;
    if (ok) {
        ok = h3_gpu_cast_f32_to_bf16(dit->gpu, features_bf16, features,
                                     (uint32_t)((size_t)mod_rows * 256)) &&
             run_linear(dit, wide, features_bf16, dit->time_l1, mod_rows, 256,
                        QWEN_HIDDEN, error, error_size, "temb linear1") &&
             h3_gpu_silu_bf16(dit->gpu, wide, wide,
                              (uint32_t)((size_t)mod_rows * QWEN_HIDDEN)) &&
             run_linear(dit, temb, wide, dit->time_l2, mod_rows, QWEN_HIDDEN,
                        QWEN_HIDDEN, error, error_size, "temb linear2") &&
             h3_gpu_copy_bf16(dit->gpu, silu_temb, 0, temb, 0,
                              (size_t)mod_rows * QWEN_HIDDEN) &&
             h3_gpu_silu_bf16(dit->gpu, silu_temb, silu_temb,
                              (uint32_t)((size_t)mod_rows * QWEN_HIDDEN)) &&
             run_linear(dit, norm_out_scale, silu_temb, dit->norm_out_w,
                        mod_rows, QWEN_HIDDEN, QWEN_HIDDEN, error, error_size,
                        "norm_out") &&
             run_linear(dit, modulation, silu_temb, dit->mod_w, mod_rows,
                        QWEN_HIDDEN, 4 * QWEN_HIDDEN, error, error_size,
                        "modulation");
    }
    if (!ok) {
        h3_gpu_tensor_free(wide);
        h3_gpu_tensor_free(silu_temb);
        h3_gpu_tensor_free(features_bf16);
        return set_error(error, error_size, "conditioning failed: %s",
                         h3_gpu_error(dit->gpu));
    }

    /* The modulation reorder reads the projection result on the host, so
     * the queued work must complete first. */
    if (!h3_gpu_submit(dit->gpu) || !h3_gpu_begin(dit->gpu)) {
        h3_gpu_tensor_free(wide);
        h3_gpu_tensor_free(silu_temb);
        h3_gpu_tensor_free(features_bf16);
        return set_error(error, error_size, "conditioning sync failed: %s",
                         h3_gpu_error(dit->gpu));
    }
    /* Free after the sync: keeping the temporaries alive across the
     * submit avoids touching buffer lifetimes around the host roundtrip
     * (an intermittent MTLResourceListAddResource crash surfaced here). */
    h3_gpu_tensor_free(wide);
    h3_gpu_tensor_free(silu_temb);
    h3_gpu_tensor_free(features_bf16);

    /* Reorder modulation columns into the slot layout with the tanh gate
     * pre-pass. Modulation rows are tiny, so a host roundtrip keeps the
     * transform exact at negligible cost. */
    size_t row_elements = 4 * (size_t)QWEN_HIDDEN;
    size_t total = row_elements * mod_rows;
    uint16_t *raw = malloc(sizeof(uint16_t) * total);
    uint16_t *ordered = malloc(sizeof(uint16_t) * total);
    if (!raw || !ordered) {
        free(raw);
        free(ordered);
        return set_error(error, error_size, "out of memory");
    }
    int ok2 = h3_gpu_tensor_read_bf16(modulation, raw, total);
    if (ok2) {
        for (uint32_t r = 0; r < mod_rows; r++) {
            const uint16_t *src = raw + (size_t)r * row_elements;
            uint16_t *dst = ordered + (size_t)r * row_elements;
            /* Released column order: [mod1.scale|mod1.gate, mod2.scale|
             * mod2.gate]. Slot layout: [mod1.scale, mod2.scale,
             * tanh(mod1.gate), tanh(mod2.gate)]. */
            memcpy(dst, src, sizeof(uint16_t) * (size_t)QWEN_HIDDEN);
            memcpy(dst + QWEN_HIDDEN, src + 2 * QWEN_HIDDEN,
                   sizeof(uint16_t) * (size_t)QWEN_HIDDEN);
            for (uint32_t c = 0; c < QWEN_HIDDEN; c++) {
                dst[2 * QWEN_HIDDEN + c] = qw_f32_to_bits(
                    tanhf(qw_bits_to_f32(src[QWEN_HIDDEN + c])));
                dst[3 * QWEN_HIDDEN + c] = qw_f32_to_bits(
                    tanhf(qw_bits_to_f32(src[3 * QWEN_HIDDEN + c])));
            }
        }
        ok2 = h3_gpu_tensor_write_bf16(modulation, ordered, total);
    }
    free(raw);
    free(ordered);
    if (!ok2) return set_error(error, error_size, "modulation reorder failed");
    return 1;
}

int qw_dit_block(qw_dit *dit, int layer_index, h3_gpu_tensor *hidden,
                 const h3_gpu_tensor *modulation,
                 const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
                 const h3_gpu_tensor *row_map, uint32_t rows,
                 uint32_t text_rows, const h3_gpu_tensor *kv_limit,
                 const qw_dit_segment *segs, uint32_t seg_count,
                 h3_gpu_tensor *k_out, h3_gpu_tensor *v_out,
                 h3_gpu_tensor *debug_attn, char *error, size_t error_size) {
    if (layer_index < 0 || layer_index >= dit->layers) {
        return set_error(error, error_size, "layer %d out of range",
                         layer_index);
    }
    const qw_dit_layer *layer = &dit->layer[layer_index];
    uint32_t image_rows = rows - text_rows;
    size_t elements = (size_t)rows * QWEN_HIDDEN;
    if (dit->ws_rows != rows) {
        int ws_ok = 1;
        for (int i = 0; i < 6; i++) {
            h3_gpu_tensor_free(dit->ws[i]);
            dit->ws[i] = h3_gpu_tensor_new_bf16(dit->gpu, elements);
            ws_ok = ws_ok && dit->ws[i];
        }
        dit->ws_rows = ws_ok ? rows : 0;
    }
    h3_gpu_tensor *modulated = dit->ws[0];
    h3_gpu_tensor *q = dit->ws[1];
    /* Caller-provided K/V buffers (ti2i step cache) take precedence. */
    h3_gpu_tensor *k = k_out ? k_out : dit->ws[2];
    h3_gpu_tensor *v = v_out ? v_out : dit->ws[3];
    h3_gpu_tensor *attn = dit->ws[4];
    h3_gpu_tensor *branch = dit->ws[5];
    int ok = modulated && q && k && v && attn && branch;
    if (!ok) {
        goto block_fail;
    }
    if (getenv("QWEN_BLOCK_PROFILE")) qw_bp_last = qw_bp_now();

    ok = qw_gpu_layer_norm_mod_bf16(
             dit->gpu, modulated, hidden, modulation, row_map, rows,
             QWEN_HIDDEN, QWEN_DIT_SLOTS, 0, 1e-6f) &&
         (getenv("H3_DISABLE_MPS_LINEAR")
              ? run_linear(dit, q, modulated, layer->to_q, rows,
                           QWEN_HIDDEN, QWEN_HIDDEN, error, error_size,
                           "to_q") &&
                   run_linear(dit, k, modulated, layer->to_k, rows,
                              QWEN_HIDDEN, QWEN_HIDDEN, error, error_size,
                              "to_k") &&
                   run_linear(dit, v, modulated, layer->to_v, rows,
                              QWEN_HIDDEN, QWEN_HIDDEN, error, error_size,
                              "to_v")
              : qw_gpu_qkv_bf16(dit->gpu, q, k, v, modulated, layer->to_q,
                                layer->to_k, layer->to_v, rows,
                                QWEN_HIDDEN)) &&
         h3_gpu_head_rms_norm_bf16(dit->gpu, q, layer->norm_q, rows,
                                   QWEN_HEADS, QWEN_HEAD_DIM, 1e-6f) &&
         h3_gpu_head_rms_norm_bf16(dit->gpu, k, layer->norm_k, rows,
                                   QWEN_HEADS, QWEN_HEAD_DIM, 1e-6f) &&
         qw_gpu_qk_rope_bf16(dit->gpu, q, k, q, k, rope_cos, rope_sin, rows,
                             QWEN_HEADS, QWEN_HEAD_DIM);
    ok = qw_bp_mark(dit->gpu, QW_BP_INPUT, ok);
    if (!ok) {
        goto block_fail;
    }

    if (kv_limit) {
        /* ti2i: every row attends through the prefix-visible table
         * (text runs causal, image runs to their block end). With the
         * run layout available, each image run is exactly the rectangle
         * [0, run_end) and goes through the MPSGraph fused SDPA; text
         * runs keep the flash prefix kernel with the in-run causal
         * table. QWEN_FLASH_ATTN=1 or a missing layout falls back to
         * the single prefix kernel over all rows; QWEN_NO_PREFIX keeps
         * the full-attention probe. */
        if (!getenv("QWEN_FLASH_ATTN") && !getenv("QWEN_NO_PREFIX") &&
            segs && seg_count) {
            uint32_t seg_start = 0;
            for (uint32_t s = 0; ok && s < seg_count; s++) {
                uint32_t seg_rows = segs[s].rows;
                uint32_t seg_end = seg_start + seg_rows;
                /* The fused rect consumes exact row counts; only the
                 * legacy prefix kernel reads whole 64-row tiles. */
                uint32_t q_alloc_rows = segs[s].is_image
                    ? seg_rows
                    : (seg_rows + 63u) & ~63u;
                h3_gpu_tensor *q_seg = h3_gpu_tensor_new_bf16(
                    dit->gpu, (size_t)q_alloc_rows * QWEN_HIDDEN);
                h3_gpu_tensor *attn_seg = h3_gpu_tensor_new_bf16(
                    dit->gpu, (size_t)seg_rows * QWEN_HIDDEN);
                ok = q_seg && attn_seg;
                if (ok) {
                    ok = h3_gpu_copy_bf16(
                             dit->gpu, q_seg, 0, q,
                             (size_t)seg_start * QWEN_HIDDEN,
                             (size_t)seg_rows * QWEN_HIDDEN);
                }
                if (ok && segs[s].is_image) {
                    ok = qw_gpu_sdpa_rect_bf16(
                        dit->gpu, attn_seg, q_seg, k, v, seg_rows, seg_end,
                        QWEN_HEADS, QWEN_HEAD_DIM,
                        QWEN_DIT_ATTENTION_SCALE);
                } else if (ok) {
                    /* Text run: in-run causal limits over the full
                     * [0, run_end) key prefix (absolute counts). */
                    uint32_t *seg_limits =
                        malloc(sizeof(uint32_t) * seg_rows);
                    h3_gpu_tensor *seg_limits_t = NULL;
                    if (seg_limits) {
                        for (uint32_t r = 0; r < seg_rows; r++)
                            seg_limits[r] = seg_start + r + 1;
                        seg_limits_t = h3_gpu_tensor_from_u32(
                            dit->gpu, seg_limits, seg_rows);
                    }
                    ok = seg_limits_t &&
                         qw_gpu_flash_sdpa_prefix_bf16(
                             dit->gpu, attn_seg, q_seg, k, v, seg_limits_t,
                             seg_rows, seg_end, QWEN_HEADS, QWEN_HEADS,
                             QWEN_HEAD_DIM,
                             QWEN_DIT_ATTENTION_SCALE);
                    free(seg_limits);
                    h3_gpu_tensor_free(seg_limits_t);
                }
                if (ok) {
                    ok = h3_gpu_copy_bf16(
                        dit->gpu, attn, (size_t)seg_start * QWEN_HIDDEN,
                        attn_seg, 0, (size_t)seg_rows * QWEN_HIDDEN);
                }
                h3_gpu_tensor_free(q_seg);
                h3_gpu_tensor_free(attn_seg);
                seg_start = seg_end;
            }
            if (!ok) {
                goto block_fail;
            }
        } else {
            uint32_t q_pad_rows = (rows + 63u) & ~63u;
            h3_gpu_tensor *q_pad = h3_gpu_tensor_new_bf16(
                dit->gpu, (size_t)q_pad_rows * QWEN_HIDDEN);
            ok = q_pad &&
                 h3_gpu_copy_bf16(dit->gpu, q_pad, 0, q, 0, elements) &&
                 (getenv("QWEN_NO_PREFIX")
                      ? qw_gpu_flash_sdpa_bf16(dit->gpu, attn, q_pad, k, v, rows,
                                               rows, QWEN_HEADS, QWEN_HEAD_DIM,
                                               QWEN_DIT_ATTENTION_SCALE)
                      : qw_gpu_flash_sdpa_prefix_bf16(dit->gpu, attn, q_pad, k, v,
                                                      kv_limit, rows, rows,
                                                      QWEN_HEADS, QWEN_HEADS,
                                                      QWEN_HEAD_DIM,
                                                      QWEN_DIT_ATTENTION_SCALE));
            h3_gpu_tensor_free(q_pad);
            if (!ok) {
                goto block_fail;
            }
        }
    } else if (text_rows) {
        h3_gpu_tensor *q_text = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)text_rows * QWEN_HIDDEN);
        h3_gpu_tensor *k_text = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)text_rows * QWEN_HIDDEN);
        h3_gpu_tensor *v_text = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)text_rows * QWEN_HIDDEN);
        h3_gpu_tensor *attn_text = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)text_rows * QWEN_HIDDEN);
        ok = q_text && k_text && v_text && attn_text;
        if (ok) {
            ok = h3_gpu_copy_bf16(dit->gpu, q_text, 0, q, 0,
                                  (size_t)text_rows * QWEN_HIDDEN) &&
                 h3_gpu_copy_bf16(dit->gpu, k_text, 0, k, 0,
                                  (size_t)text_rows * QWEN_HIDDEN) &&
                 h3_gpu_copy_bf16(dit->gpu, v_text, 0, v, 0,
                                  (size_t)text_rows * QWEN_HIDDEN) &&
                 h3_gpu_gqa_causal_bf16(dit->gpu, attn_text, q_text, k_text,
                                        v_text, text_rows, QWEN_HEADS,
                                        QWEN_HEADS, QWEN_HEAD_DIM,
                                        QWEN_DIT_ATTENTION_SCALE) &&
                 h3_gpu_copy_bf16(dit->gpu, attn, 0, attn_text, 0,
                                  (size_t)text_rows * QWEN_HIDDEN);
        }
        h3_gpu_tensor_free(q_text);
        h3_gpu_tensor_free(k_text);
        h3_gpu_tensor_free(v_text);
        h3_gpu_tensor_free(attn_text);
        if (!ok) {
            goto block_fail;
        }
    }
    if (!kv_limit && image_rows) {
        /* Image rows attend every key. The MPSGraph fused SDPA operator
         * consumes exact row counts; the legacy flash2 kernel
         * (QWEN_FLASH_ATTN=1) reads whole 64-row query tiles and needs
         * the padded allocation. */
        uint32_t q_alloc_rows = getenv("QWEN_FLASH_ATTN")
            ? (image_rows + 63u) & ~63u
            : image_rows;
        h3_gpu_tensor *q_img = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)q_alloc_rows * QWEN_HIDDEN);
        h3_gpu_tensor *attn_img = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)image_rows * QWEN_HIDDEN);
        ok = q_img && attn_img;
        if (ok) {
            ok = h3_gpu_copy_bf16(dit->gpu, q_img, 0, q,
                                  (size_t)text_rows * QWEN_HIDDEN,
                                  (size_t)image_rows * QWEN_HIDDEN) &&
                 (getenv("QWEN_FLASH_ATTN")
                      ? qw_gpu_flash_sdpa_bf16(dit->gpu, attn_img, q_img, k,
                                               v, image_rows, rows,
                                               QWEN_HEADS, QWEN_HEAD_DIM,
                                               QWEN_DIT_ATTENTION_SCALE)
                      : qw_gpu_sdpa_rect_bf16(dit->gpu, attn_img, q_img, k,
                                              v, image_rows, rows,
                                              QWEN_HEADS, QWEN_HEAD_DIM,
                                              QWEN_DIT_ATTENTION_SCALE)) &&
                 h3_gpu_copy_bf16(dit->gpu, attn,
                                  (size_t)text_rows * QWEN_HIDDEN, attn_img,
                                  0, (size_t)image_rows * QWEN_HIDDEN);
        }
        h3_gpu_tensor_free(q_img);
        h3_gpu_tensor_free(attn_img);
        if (!ok) {
            goto block_fail;
        }
    }
    ok = qw_bp_mark(dit->gpu, QW_BP_ATTN, ok);
    if (!ok) {
        goto block_fail;
    }

    ok = run_linear(dit, branch, attn, layer->to_out, rows, QWEN_HIDDEN,
                    QWEN_HIDDEN, error, error_size, "to_out") &&
         (!debug_attn ||
          h3_gpu_copy_bf16(dit->gpu, debug_attn, 0, attn, 0, elements)) &&
         h3_gpu_gate_bf16(dit->gpu, hidden, hidden, branch, modulation,
                          row_map, rows, QWEN_HIDDEN, QWEN_DIT_SLOTS, 2);
    ok = qw_bp_mark(dit->gpu, QW_BP_OUT, ok);
    if (!ok) {
        goto block_fail;
    }
    ok = qw_gpu_layer_norm_mod_bf16(dit->gpu, modulated, hidden, modulation,
                                    row_map, rows, QWEN_HIDDEN,
                                    QWEN_DIT_SLOTS, 1, 1e-6f);
    ok = qw_bp_mark(dit->gpu, QW_BP_LN, ok);
    if (!ok) {
        goto block_fail;
    }
    ok = h3_gpu_mlp_bf16(dit->gpu, branch, modulated, layer->fc1, layer->fc2,
                         rows, QWEN_HIDDEN, QWEN_DIT_MLP, QWEN_HIDDEN) &&
         h3_gpu_gate_bf16(dit->gpu, hidden, hidden, branch, modulation,
                          row_map, rows, QWEN_HIDDEN, QWEN_DIT_SLOTS, 3);
    ok = qw_bp_mark(dit->gpu, QW_BP_MLP, ok);
    if (!ok) {
        goto block_fail;
    }
    if (getenv("QWEN_BLOCK_PROFILE")) qw_bp_blocks++;
    return 1;

block_fail:
    return set_error(error, error_size, "block %d kernel failed: %s",
                     layer_index, h3_gpu_error(dit->gpu));
}

int qw_dit_final(qw_dit *dit, h3_gpu_tensor *hidden,
                 const h3_gpu_tensor *norm_out_scale,
                 const h3_gpu_tensor *row_map, uint32_t rows,
                 h3_gpu_tensor *output, char *error, size_t error_size) {
    h3_gpu_tensor *modulated = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)rows * QWEN_HIDDEN);
    int ok = modulated != NULL;
    if (ok) {
        ok = qw_gpu_layer_norm_mod_bf16(dit->gpu, modulated, hidden,
                                        norm_out_scale, row_map, rows,
                                        QWEN_HIDDEN, 1, 0, 1e-6f) &&
             run_linear(dit, output, modulated, dit->proj_out, rows,
                        QWEN_HIDDEN, QWEN_Z_DIM, error, error_size,
                        "proj_out");
    }
    h3_gpu_tensor_free(modulated);
    if (!ok) return set_error(error, error_size, "final projection failed");
    return 1;
}

int qw_dit_forward(qw_dit *dit, const h3_gpu_tensor *text_in,
                   uint32_t text_rows, const h3_gpu_tensor *latents,
                   uint32_t image_rows, float sigma,
                   const h3_gpu_tensor *rope_cos,
                   const h3_gpu_tensor *rope_sin, h3_gpu_tensor *output,
                   char *error, size_t error_size) {
    uint32_t rows = text_rows + image_rows;
    size_t elements = (size_t)rows * QWEN_HIDDEN;

    h3_gpu_tensor *joint = h3_gpu_tensor_new_bf16(dit->gpu, elements);
    /* row_map: the causal_condition text prefix rides the trailing t=0
     * modulation row, target image rows ride the timestep row. */
    uint32_t *row_map_host = malloc(sizeof(uint32_t) * rows);
    int ok = joint && row_map_host;
    h3_gpu_tensor *row_map_t = NULL;
    if (ok) {
        for (uint32_t i = 0; i < rows; i++) {
            row_map_host[i] = i < text_rows ? 1u : 0u;
        }
        row_map_t = h3_gpu_tensor_from_u32(dit->gpu, row_map_host, rows);
        ok = row_map_t != NULL;
    }
    free(row_map_host);
    if (!ok) {
        h3_gpu_tensor_free(joint);
        h3_gpu_tensor_free(row_map_t);
        return set_error(error, error_size, "forward allocation failed");
    }

    ok = qw_dit_embed(dit, text_in, text_rows, latents, image_rows, joint,
                      error, error_size);
    if (ok) {
        float sigmas[2] = {sigma, 0.0f};
        float features[512];
        qw_dit_timestep_features(sigmas, 2, features);
        h3_gpu_tensor *features_t =
            h3_gpu_tensor_from_f32(dit->gpu, features, 512);
        h3_gpu_tensor *temb = h3_gpu_tensor_new_bf16(dit->gpu,
                                                     2 * QWEN_HIDDEN);
        h3_gpu_tensor *modulation = h3_gpu_tensor_new_bf16(
            dit->gpu, 2 * 4 * QWEN_HIDDEN);
        h3_gpu_tensor *norm_scale = h3_gpu_tensor_new_bf16(
            dit->gpu, 2 * QWEN_HIDDEN);
        ok = features_t && temb && modulation && norm_scale &&
             qw_dit_conditioning(dit, features_t, 2, temb, modulation,
                                 norm_scale, error, error_size);
        h3_gpu_tensor_free(features_t);
        h3_gpu_tensor_free(temb);
        if (ok) {
            ok = h3_gpu_submit(dit->gpu) && h3_gpu_begin(dit->gpu);
            for (int i = 0; ok && i < dit->layers; i++) {
                ok = qw_dit_block(dit, i, joint, modulation, rope_cos,
                                  rope_sin, row_map_t, rows, text_rows, NULL,
                                  NULL, 0, NULL, NULL, NULL, error,
                                  error_size);
            }
            qw_bp_report();
            if (ok) {
                ok = qw_dit_final(dit, joint, norm_scale, row_map_t, rows,
                                  output, error, error_size);
            }
        }
        h3_gpu_tensor_free(modulation);
        h3_gpu_tensor_free(norm_scale);
    }
    h3_gpu_tensor_free(joint);
    h3_gpu_tensor_free(row_map_t);
    if (!ok) return set_error(error, error_size, "forward failed");
    return 1;
}

/* ti2i joint embed: project the full text and latent inputs, then copy
 * the runs into the interleaved joint layout. */
int qw_dit_embed_segments(qw_dit *dit, const h3_gpu_tensor *text_in,
                          const h3_gpu_tensor *latents,
                          const qw_dit_segment *segs, uint32_t seg_count,
                          h3_gpu_tensor *joint, char *error,
                          size_t error_size) {
    /* text_in carries the full VLM sequence (image-slot rows included;
     * they are projected and then skipped by src_offset), latents the
     * packed condition + target rows. */
    uint32_t text_rows =
        (uint32_t)(h3_gpu_tensor_elements(text_in) / QWEN_HIDDEN);
    uint32_t image_rows =
        (uint32_t)(h3_gpu_tensor_elements(latents) / QWEN_Z_DIM);
    uint32_t rows = 0;
    for (uint32_t s = 0; s < seg_count; s++) rows += segs[s].rows;
    if (h3_gpu_tensor_elements(joint) < (size_t)rows * QWEN_HIDDEN) {
        return set_error(error, error_size, "joint tensor too small");
    }
    h3_gpu_tensor *txt_out = NULL, *img_out = NULL;
    int ok = 1;
    if (text_rows) {
        h3_gpu_tensor *normed = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)text_rows * QWEN_HIDDEN);
        h3_gpu_tensor *wide = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)text_rows * QWEN_HIDDEN);
        txt_out = h3_gpu_tensor_new_bf16(dit->gpu,
                                         (size_t)text_rows * QWEN_HIDDEN);
        ok = normed && wide && txt_out &&
             qw_gpu_rms_norm_one_bf16(dit->gpu, normed, text_in,
                                      dit->txt_norm, text_rows, QWEN_HIDDEN,
                                      1e-6f) &&
             run_linear(dit, wide, normed, dit->txt_in1, text_rows,
                        QWEN_HIDDEN, QWEN_HIDDEN, error, error_size,
                        "txt_in1") &&
             h3_gpu_gelu_bf16(dit->gpu, wide, wide,
                              (uint32_t)((size_t)text_rows * QWEN_HIDDEN),
                              1) &&
             run_linear(dit, txt_out, wide, dit->txt_in2, text_rows,
                        QWEN_HIDDEN, QWEN_HIDDEN, error, error_size,
                        "txt_in2");
        h3_gpu_tensor_free(normed);
        h3_gpu_tensor_free(wide);
        if (!ok) {
            h3_gpu_tensor_free(txt_out);
            return set_error(error, error_size, "txt_in failed: %s",
                             h3_gpu_error(dit->gpu));
        }
    }
    if (image_rows) {
        img_out = h3_gpu_tensor_new_bf16(dit->gpu,
                                         (size_t)image_rows * QWEN_HIDDEN);
        ok = img_out &&
             run_linear(dit, img_out, latents, dit->img_in, image_rows,
                        QWEN_Z_DIM, QWEN_HIDDEN, error, error_size,
                        "img_in");
        if (!ok) {
            h3_gpu_tensor_free(txt_out);
            h3_gpu_tensor_free(img_out);
            return set_error(error, error_size, "img_in failed: %s",
                             h3_gpu_error(dit->gpu));
        }
    }
    /* Scatter runs into the joint layout. */
    uint32_t joint_off = 0;
    for (uint32_t s = 0; ok && s < seg_count; s++) {
        size_t elements = (size_t)segs[s].rows * QWEN_HIDDEN;
        const h3_gpu_tensor *src = segs[s].is_image ? img_out : txt_out;
        ok = h3_gpu_copy_bf16(dit->gpu, joint,
                              (size_t)joint_off * QWEN_HIDDEN, src,
                              (size_t)segs[s].src_offset * QWEN_HIDDEN,
                              elements);
        joint_off += segs[s].rows;
    }
    h3_gpu_tensor_free(txt_out);
    h3_gpu_tensor_free(img_out);
    if (!ok) return set_error(error, error_size, "joint scatter failed: %s",
                              h3_gpu_error(dit->gpu));
    return 1;
}

void qw_dit_prefix_limits(const qw_dit_segment *segs, uint32_t seg_count,
                          uint32_t *limits) {
    uint32_t start = 0;
    for (uint32_t s = 0; s < seg_count; s++) {
        if (segs[s].is_image) {
            /* Block-causal image rows see up to the block end. */
            for (uint32_t r = 0; r < segs[s].rows; r++)
                limits[start + r] = start + segs[s].rows;
        } else {
            for (uint32_t r = 0; r < segs[s].rows; r++)
                limits[start + r] = start + r + 1;
        }
        start += segs[s].rows;
    }
}

int qw_dit_forward_ti2i(qw_dit *dit, const h3_gpu_tensor *text_in,
                        const h3_gpu_tensor *latents,
                        const qw_dit_segment *segs, uint32_t seg_count,
                        uint32_t cache_slot,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, float sigma,
                        h3_gpu_tensor *output, char *error,
                        size_t error_size) {
    /* Several paths build messages from `error[0] ? error : ...`; callers
     * are not required to zero the buffer, so start from a clean slate. */
    if (error && error_size) error[0] = 0;
    struct qw_dit_ti2i_cache *c =
        &dit->ti2i_cache[cache_slot ? 1 : 0];
    uint32_t rows = 0, target_rows = 0;
    for (uint32_t s = 0; s < seg_count; s++) rows += segs[s].rows;
    if (!seg_count || !segs[seg_count - 1].is_image) {
        return set_error(error, error_size, "ti2i: last run must be image");
    }
    target_rows = segs[seg_count - 1].rows;
    uint32_t t_base = rows - target_rows;
    uint32_t t_rows = target_rows;
    size_t elements = (size_t)rows * QWEN_HIDDEN;

    /* Cross-step cache: the cond/text rows' inputs never change across
     * sampling steps, so one full pass fills every layer's K/V for them
     * and later steps recompute only the target rows against the cached
     * K/V (the target workspace stays closed over the whole stack). */
    int reuse = c->k && c->rows == rows &&
                c->target_rows == target_rows &&
                c->seg_count == seg_count &&
                ti2i_segs_match(c->segs, segs, seg_count) &&
                !getenv("QWEN_NO_STEP_CACHE") &&
                !getenv("QWEN_FLASH_ATTN") && !getenv("QWEN_NO_PREFIX");
    /* A cache allocated this call is EMPTY until the full pass fills it:
     * any failure below must drop it again, or the next call's reuse
     * check would happily serve uninitialized/half-filled K/V. */
    int fresh = !reuse;
    uint32_t cache_idx = cache_slot ? 1 : 0;
    if (!reuse &&
        !ti2i_cache_alloc(dit, cache_idx, rows, target_rows, segs,
                          seg_count, error, error_size)) {
        return 0;
    }
    int target_only = reuse;

    /* Setup 在调用方 submit/begin 的任何时点都可能被进入（CFG 每步
     * 连续两次调用），统一兜底开启命令链。 */
    if (!h3_gpu_begin(dit->gpu) && !h3_gpu_continue(dit->gpu)) {
        if (fresh) ti2i_cache_free(dit, cache_idx);
        return set_error(error, error_size, "ti2i: setup begin failed");
    }

    h3_gpu_tensor *joint = NULL;
    h3_gpu_tensor *row_map_t = NULL, *limits_t = NULL;
    int ok = 1;
    if (!target_only) {
        joint = h3_gpu_tensor_new_bf16(dit->gpu, elements);
        uint32_t *row_map_host = malloc(sizeof(uint32_t) * rows);
        uint32_t *limits_host = malloc(sizeof(uint32_t) * rows);
        ok = joint && row_map_host && limits_host;
        if (ok) {
            /* Only the trailing target block reads the sampled timestep;
             * everything else modulates from t = 0. */
            uint32_t prefix = rows - target_rows;
            for (uint32_t i = 0; i < rows; i++)
                row_map_host[i] = i < prefix ? 1u : 0u;
            qw_dit_prefix_limits(segs, seg_count, limits_host);
            row_map_t = h3_gpu_tensor_from_u32(dit->gpu, row_map_host, rows);
            limits_t = h3_gpu_tensor_from_u32(dit->gpu, limits_host, rows);
            ok = row_map_t && limits_t;
        }
        free(row_map_host);
        if (ok) {
            ok = qw_dit_embed_segments(dit, text_in, latents, segs, seg_count,
                                       joint, error, error_size);
        }
        free(limits_host);
        if (!ok) {
            h3_gpu_tensor_free(joint);
            h3_gpu_tensor_free(row_map_t);
            h3_gpu_tensor_free(limits_t);
            if (fresh) ti2i_cache_free(dit, cache_idx);
            return set_error(error, error_size, "ti2i forward setup failed: %s",
                             error[0] ? error : "alloc");
        }
    } else {
        /* Target-only setup: re-embed the new target latents (this step's
         * joint row inputs), slice the rope tables to the target rows,
         * and gather the rows into the workspace that stays closed over
         * the whole stack. The target rows sit at segs' src_offset inside
         * the packed latents (cond rows only — the joint-row base t_base
         * includes the text rows and would misalign). */
        uint32_t lat_off = segs[seg_count - 1].src_offset;
        ok = h3_gpu_copy_bf16(dit->gpu, c->t_lat, 0, latents,
                              (size_t)lat_off * QWEN_Z_DIM,
                              (size_t)t_rows * QWEN_Z_DIM) &&
             run_linear(dit, c->th, c->t_lat, dit->img_in,
                        t_rows, QWEN_Z_DIM, QWEN_HIDDEN, error, error_size,
                        "img_in_t") &&
             h3_gpu_copy_f32(dit->gpu, c->rope_cos, 0, rope_cos,
                             (size_t)t_base * (QWEN_HEAD_DIM / 2),
                             (size_t)t_rows * (QWEN_HEAD_DIM / 2)) &&
             h3_gpu_copy_f32(dit->gpu, c->rope_sin, 0, rope_sin,
                             (size_t)t_base * (QWEN_HEAD_DIM / 2),
                             (size_t)t_rows * (QWEN_HEAD_DIM / 2));
        if (!ok) {
            return set_error(error, error_size,
                             "ti2i target setup failed: %s",
                             error[0] ? error : "alloc");
        }
    }

    float sigmas[2] = {sigma, 0.0f};
    float features[512];
    qw_dit_timestep_features(sigmas, 2, features);
    h3_gpu_tensor *features_t = h3_gpu_tensor_from_f32(dit->gpu, features, 512);
    h3_gpu_tensor *temb =
        h3_gpu_tensor_new_bf16(dit->gpu, 2 * QWEN_HIDDEN);
    h3_gpu_tensor *modulation =
        h3_gpu_tensor_new_bf16(dit->gpu, 2 * 4 * QWEN_HIDDEN);
    h3_gpu_tensor *norm_scale =
        h3_gpu_tensor_new_bf16(dit->gpu, 2 * QWEN_HIDDEN);
    ok = features_t && temb && modulation && norm_scale &&
         qw_dit_conditioning(dit, features_t, 2, temb, modulation,
                             norm_scale, error, error_size);
    {
        const char *dd = getenv("QWEN_DIT_DUMP");
        if (dd && dd[0]) {
            FILE *df = NULL;
            char fpath[4096];
            /* dump 的 read 需要命令已提交；submit 不重开命令缓冲，这里
             * 立即 begin，否则下方 `submit && begin` 链会因缓冲已关而
             * 短路成 forward 失败。 */
            h3_gpu_submit(dit->gpu);
            h3_gpu_begin(dit->gpu);
            snprintf(fpath, sizeof(fpath), "%s/features.bin", dd);
            df = fopen(fpath, "wb");
            if (df) { fwrite(features, 4, 512, df); fclose(df); }
            size_t tn = (size_t)2 * QWEN_HIDDEN;
            uint16_t *tb = malloc(sizeof(uint16_t) * tn);
            if (tb && h3_gpu_tensor_read_bf16(temb, tb, tn)) {
                char p[4096];
                snprintf(p, sizeof(p), "%s/temb.bin", dd);
                df = fopen(p, "wb");
                if (df) { fwrite(tb, 2, tn, df); fclose(df); }
            }
            free(tb);
        }
    }
    h3_gpu_tensor_free(features_t);
    h3_gpu_tensor_free(temb);
    {
        const char *dd = getenv("QWEN_DIT_DUMP");
        if (dd && dd[0]) {
            size_t mn = (size_t)2 * 4 * QWEN_HIDDEN;
            uint16_t *mb = malloc(sizeof(uint16_t) * mn);
            if (mb && h3_gpu_tensor_read_bf16(modulation, mb, mn)) {
                char p[4096];
                snprintf(p, sizeof(p), "%s/modulation.bin", dd);
                FILE *df = fopen(p, "wb");
                if (df) { fwrite(mb, 2, mn, df); fclose(df); }
            }
            free(mb);
            if (joint) { /* target-only steps carry no full hidden */
                size_t jn = (size_t)rows * QWEN_HIDDEN;
                uint16_t *jb = malloc(sizeof(uint16_t) * jn);
                if (jb && h3_gpu_tensor_read_bf16(joint, jb, jn)) {
                    char p[4096];
                    snprintf(p, sizeof(p), "%s/joint_in.bin", dd);
                    FILE *df = fopen(p, "wb");
                    if (df) { fwrite(jb, 2, jn, df); fclose(df); }
                }
                free(jb);
            }
        }
    }
    if (ok) {
        ok = h3_gpu_submit(dit->gpu) && h3_gpu_begin(dit->gpu);
    }
    if (ok && !target_only) {
        for (int i = 0; ok && i < dit->layers; i++) {
            ok = qw_dit_block(dit, i, joint, modulation, rope_cos, rope_sin,
                              row_map_t, rows, 0, limits_t, segs, seg_count,
                              c->k[i], c->v[i], NULL, error,
                              error_size);
        }
        qw_bp_report();
        if (ok) {
            ok = qw_dit_final(dit, joint, norm_scale, row_map_t, rows,
                              output, error, error_size);
        }
    } else if (ok) {
        /* Target-only pass: every layer computes just the target rows
         * (LN -> fused qkv -> norms/rope -> attention of the target
         * queries against the cached full K/V -> out/gate/MLP/gate),
         * with the workspace closed over the whole stack. */
        for (int i = 0; ok && i < dit->layers; i++) {
            const qw_dit_layer *layer = &dit->layer[i];
            ok = qw_gpu_layer_norm_mod_bf16(
                     dit->gpu, c->mod, c->th, modulation,
                     c->row0, t_rows, QWEN_HIDDEN, QWEN_DIT_SLOTS, 0,
                     1e-6f) &&
                 qw_gpu_qkv_bf16(dit->gpu, c->q, c->kt,
                                 c->vt, c->mod, layer->to_q,
                                 layer->to_k, layer->to_v, t_rows,
                                 QWEN_HIDDEN) &&
                 h3_gpu_head_rms_norm_bf16(dit->gpu, c->q,
                                           layer->norm_q, t_rows, QWEN_HEADS,
                                           QWEN_HEAD_DIM, 1e-6f) &&
                 h3_gpu_head_rms_norm_bf16(dit->gpu, c->kt,
                                           layer->norm_k, t_rows, QWEN_HEADS,
                                           QWEN_HEAD_DIM, 1e-6f) &&
                 qw_gpu_qk_rope_bf16(dit->gpu, c->q, c->kt,
                                     c->q, c->kt,
                                     c->rope_cos, c->rope_sin,
                                     t_rows, QWEN_HEADS, QWEN_HEAD_DIM) &&
                 h3_gpu_copy_bf16(dit->gpu, c->k[i],
                                  (size_t)t_base * QWEN_HIDDEN,
                                  c->kt, 0,
                                  (size_t)t_rows * QWEN_HIDDEN) &&
                 h3_gpu_copy_bf16(dit->gpu, c->v[i],
                                  (size_t)t_base * QWEN_HIDDEN,
                                  c->vt, 0,
                                  (size_t)t_rows * QWEN_HIDDEN) &&
                 qw_gpu_sdpa_rect_bf16(dit->gpu, c->attn, c->q,
                                       c->k[i], c->v[i],
                                       t_rows, rows, QWEN_HEADS,
                                       QWEN_HEAD_DIM,
                                       QWEN_DIT_ATTENTION_SCALE) &&
                 run_linear(dit, c->branch, c->attn,
                            layer->to_out, t_rows, QWEN_HIDDEN, QWEN_HIDDEN,
                            error, error_size, "to_out_t") &&
                 h3_gpu_gate_bf16(dit->gpu, c->th, c->th,
                                  c->branch, modulation,
                                  c->row0, t_rows, QWEN_HIDDEN,
                                  QWEN_DIT_SLOTS, 2) &&
                 qw_gpu_layer_norm_mod_bf16(
                     dit->gpu, c->mod, c->th, modulation,
                     c->row0, t_rows, QWEN_HIDDEN, QWEN_DIT_SLOTS, 1,
                     1e-6f) &&
                 h3_gpu_mlp_bf16(dit->gpu, c->branch, c->mod,
                                 layer->fc1, layer->fc2, t_rows, QWEN_HIDDEN,
                                 QWEN_DIT_MLP, QWEN_HIDDEN) &&
                 h3_gpu_gate_bf16(dit->gpu, c->th, c->th,
                                  c->branch, modulation,
                                  c->row0, t_rows, QWEN_HIDDEN,
                                  QWEN_DIT_SLOTS, 3);
            if (!ok) {
                set_error(error, error_size, "target block %d failed: %s", i,
                          h3_gpu_error(dit->gpu));
                break;
            }
        }
        /* Final norm + projection on the target rows only; the result
         * lands in the trailing segment of the full output. */
        if (ok) {
            h3_gpu_tensor *proj = h3_gpu_tensor_new_bf16(
                dit->gpu, (size_t)t_rows * QWEN_Z_DIM);
            ok = proj &&
                 qw_gpu_layer_norm_mod_bf16(
                     dit->gpu, c->mod, c->th, norm_scale,
                     c->row0, t_rows, QWEN_HIDDEN, 1, 0, 1e-6f) &&
                 run_linear(dit, proj, c->mod, dit->proj_out, t_rows,
                            QWEN_HIDDEN, QWEN_Z_DIM, error, error_size,
                            "proj_out_t") &&
                 h3_gpu_copy_bf16(dit->gpu, output,
                                  (size_t)t_base * QWEN_Z_DIM, proj, 0,
                                  (size_t)t_rows * QWEN_Z_DIM);
            h3_gpu_tensor_free(proj);
            if (!ok) {
                set_error(error, error_size, "ti2i final failed: %s",
                          h3_gpu_error(dit->gpu));
            }
        }
    }
    h3_gpu_tensor_free(joint);
    h3_gpu_tensor_free(row_map_t);
    h3_gpu_tensor_free(limits_t);
    h3_gpu_tensor_free(modulation);
    h3_gpu_tensor_free(norm_scale);
    if (!ok) {
        char detail[256];
        snprintf(detail, sizeof(detail), "%s", error[0] ? error : "unknown");
        /* A mid-pass failure leaves this call's freshly allocated K/V
         * cache half-filled; drop it so the next forward rebuilds from
         * scratch instead of reusing garbage. (A target-only failure
         * keeps the previously filled cache — cond K/V remain valid.) */
        if (fresh) ti2i_cache_free(dit, cache_idx);
        return set_error(error, error_size, "ti2i forward failed: %s", detail);
    }
    return 1;
}
