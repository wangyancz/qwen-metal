#include "qwen_vision.h"

#include "h3_safetensors.h"
#include "qwen_gpu_ext.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define VISION_HIDDEN 1152
#define VISION_HEADS 16
#define VISION_HEAD_DIM 72          /* 1152 / 16 */
#define VISION_ROPE_FREQ 18         /* head_dim / 4 */
#define VISION_MLP 4304
#define VISION_PATCH_IN 1536        /* 3 * 2 * 16 * 16 */
#define VISION_OUT 4096
#define VISION_MERGED_IN 4608       /* 1152 * 4 */
#define VISION_BLOCKS 27
#define VISION_POS_SIDE 48          /* sqrt(2304) */
#define VISION_DEEPSTACK 3
#define VISION_DEEP_BLOCKS {8, 16, 24}

struct qw_vision {
    int use_gpu;
    h3_gpu *gpu;                         /* 非拥有（引擎共享实例） */
    /* GPU 权重（bf16 tensors，与 host 平行布局）。 */
    h3_gpu_tensor *g_proj_w, *g_proj_b;
    struct {
        h3_gpu_tensor *norm1_w, *norm1_b;
        h3_gpu_tensor *qkv_w, *qkv_b;
        h3_gpu_tensor *proj_w, *proj_b;
        h3_gpu_tensor *norm2_w, *norm2_b;
        h3_gpu_tensor *fc1_w, *fc1_b;
        h3_gpu_tensor *fc2_w, *fc2_b;
    } g_block[VISION_BLOCKS];
    h3_gpu_tensor *g_merger_norm_w, *g_merger_norm_b;
    h3_gpu_tensor *g_merger_fc1_w, *g_merger_fc1_b;
    h3_gpu_tensor *g_merger_fc2_w, *g_merger_fc2_b;
    struct {
        h3_gpu_tensor *norm_w, *norm_b;
        h3_gpu_tensor *fc1_w, *fc1_b;
        h3_gpu_tensor *fc2_w, *fc2_b;
    } g_deep[VISION_DEEPSTACK];
    float *proj_w, *proj_b;              /* [1152,1536], [1152] */
    float *pos_embed;                    /* [2304*1152] row-major */
    struct {
        float *norm1_w, *norm1_b;        /* [1152] */
        float *qkv_w, *qkv_b;            /* [3456,1152], [3456] */
        float *proj_w, *proj_b;          /* [1152,1152] */
        float *norm2_w, *norm2_b;
        float *fc1_w, *fc1_b;            /* [4304,1152] */
        float *fc2_w, *fc2_b;            /* [1152,4304] */
    } block[VISION_BLOCKS];
    float *merger_norm_w, *merger_norm_b;    /* [1152] pre-shuffle */
    float *merger_fc1_w, *merger_fc1_b;      /* [4608,4608] */
    float *merger_fc2_w, *merger_fc2_b;      /* [4096,4608] */
    struct {
        float *norm_w, *norm_b;              /* [4608] post-shuffle */
        float *fc1_w, *fc1_b;                /* [4608,4608] */
        float *fc2_w, *fc2_b;                /* [4096,4608] */
    } deep[VISION_DEEPSTACK];
    float rope_inv_freq[VISION_ROPE_FREQ];   /* theta 10000 */
    const char *dump_dir;                    /* QWEN_VISION_DUMP */
};

void qw_vision_set_dump(qw_vision *v, const char *dir) {
    if (!v) return;
    v->dump_dir = dir;
    if (dir) mkdir(dir, 0755);
}

static void dump_f32(const qw_vision *v, const char *name, const float *data,
                     size_t count) {
    if (!v || !v->dump_dir) return;
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s.f32", v->dump_dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(data, sizeof(float), count, f);
    fclose(f);
}

/* bf16 舍入（round-to-nearest-even）：transformers 的 .to(bf16) 会把
 * inv_freq buffer 量化（P3 文本塔同款坑），参考实现频率是 bf16 的 */
static float bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    memcpy(&x, &u, 4);
    return x;
}

static float *bf16_to_f32(const h3_st_header *header, const h3_st_tensor *t) {
    size_t elements = (size_t)h3_st_tensor_elements(t);
    uint16_t *raw = malloc(elements * 2);
    if (!raw) return NULL;
    if (!h3_st_read_data(header, t, raw, elements * 2, NULL, 0)) {
        free(raw);
        return NULL;
    }
    float *out = malloc(sizeof(float) * elements);
    if (!out) {
        free(raw);
        return NULL;
    }
    for (size_t i = 0; i < elements; i++) {
        uint32_t wide = (uint32_t)raw[i] << 16;
        memcpy(&out[i], &wide, 4);
    }
    free(raw);
    return out;
}

static float *load_named(const char *dir, const char *name, size_t elements,
                         char *error, size_t error_size) {
    char path[1024];
    h3_st_header header;
    /* base 的 text_encoder 是 4 分片，turbo 是单文件——依次尝试。 */
    static const char *files[] = {
        "model-00001-of-00004.safetensors",
        "model.safetensors",
    };
    int opened = 0;
    for (size_t fi = 0; fi < sizeof(files) / sizeof(files[0]); fi++) {
        snprintf(path, sizeof(path), "%s/%s", dir, files[fi]);
        char probe[256];
        if (h3_st_read_header(path, &header, probe, sizeof(probe))) {
            opened = 1;
            break;
        }
    }
    if (!opened) {
        snprintf(error, error_size, "vision: no safetensors under %s", dir);
        return NULL;
    }
    const h3_st_tensor *t = h3_st_find(&header, name);
    if (!t || h3_st_tensor_elements(t) != elements ||
        t->dtype != H3_DTYPE_BF16) {
        snprintf(error, error_size, "vision weight %s missing/mismatch", name);
        h3_st_free_header(&header);
        return NULL;
    }
    float *out = bf16_to_f32(&header, t);
    h3_st_free_header(&header);
    if (!out) snprintf(error, error_size, "cannot read %s", name);
    return out;
}

/* 原始 bf16 bits 直读（GPU 权重上传路径，fp32 中转可省）。 */

/* fp32 → bf16 bits（round-to-nearest-even）。 */
static uint16_t f32_to_bf16_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* host fp32 → GPU bf16 tensor（一次性权重上传）。 */
static h3_gpu_tensor *up_bf16(h3_gpu *gpu, const float *f32, size_t n) {
    uint16_t *bits = malloc(n * 2);
    if (!bits) return NULL;
    for (size_t i = 0; i < n; i++) bits[i] = f32_to_bf16_bits(f32[i]);
    h3_gpu_tensor *t = h3_gpu_tensor_new_bf16(gpu, n);
    if (!t || !h3_gpu_tensor_write_bf16(t, bits, n)) {
        h3_gpu_tensor_free(t);
        free(bits);
        return NULL;
    }
    free(bits);
    return t;
}

/* GPU 模式下 host fp32 权重只有 CPU 回退路径（QWEN_VISION_CPU=1，create
 * 时二选一）会消费，上传成 bf16 tensor 后即可释放，省 ~2.2GB 常驻内存。
 * pos_embed 与 rope_inv_freq 仍被 GPU 路径的 CPU 侧插值/频率使用，保留。 */
static void free_host_weights(qw_vision *v) {
    free(v->proj_w); v->proj_w = NULL;
    free(v->proj_b); v->proj_b = NULL;
    for (int i = 0; i < VISION_BLOCKS; i++) {
        float **fields[] = {
            &v->block[i].norm1_w, &v->block[i].norm1_b,
            &v->block[i].qkv_w,   &v->block[i].qkv_b,
            &v->block[i].proj_w,  &v->block[i].proj_b,
            &v->block[i].norm2_w, &v->block[i].norm2_b,
            &v->block[i].fc1_w,   &v->block[i].fc1_b,
            &v->block[i].fc2_w,   &v->block[i].fc2_b,
        };
        for (size_t k = 0; k < sizeof(fields) / sizeof(fields[0]); k++) {
            free(*fields[k]);
            *fields[k] = NULL;
        }
    }
    free(v->merger_norm_w); v->merger_norm_w = NULL;
    free(v->merger_norm_b); v->merger_norm_b = NULL;
    free(v->merger_fc1_w);  v->merger_fc1_w = NULL;
    free(v->merger_fc1_b);  v->merger_fc1_b = NULL;
    free(v->merger_fc2_w);  v->merger_fc2_w = NULL;
    free(v->merger_fc2_b);  v->merger_fc2_b = NULL;
    for (int i = 0; i < VISION_DEEPSTACK; i++) {
        free(v->deep[i].norm_w); v->deep[i].norm_w = NULL;
        free(v->deep[i].norm_b); v->deep[i].norm_b = NULL;
        free(v->deep[i].fc1_w);  v->deep[i].fc1_w = NULL;
        free(v->deep[i].fc1_b);  v->deep[i].fc1_b = NULL;
        free(v->deep[i].fc2_w);  v->deep[i].fc2_w = NULL;
        free(v->deep[i].fc2_b);  v->deep[i].fc2_b = NULL;
    }
}

qw_vision *qw_vision_create(const char *model_dir, h3_gpu *gpu, char *error,
                            size_t error_size) {
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/text_encoder", model_dir);
    qw_vision *v = calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->gpu = gpu;
    /* QWEN_VISION_CPU=1 forces the CPU fallback; "0" must NOT (presence
     * alone used to disable the GPU path, so =0 read as =1). */
    const char *cpu_env = getenv("QWEN_VISION_CPU");
    v->use_gpu = gpu && !(cpu_env && *cpu_env && strcmp(cpu_env, "0") != 0);
    for (int i = 0; i < VISION_ROPE_FREQ; i++)
        v->rope_inv_freq[i] = bf16_round(
            (float)(1.0 / pow(10000.0, (2.0 * i) / 36.0)));

    char name[256];
    v->proj_w = load_named(dir, "model.visual.patch_embed.proj.weight",
                           VISION_HIDDEN * VISION_PATCH_IN, error, error_size);
    v->proj_b = load_named(dir, "model.visual.patch_embed.proj.bias",
                           VISION_HIDDEN, error, error_size);
    v->pos_embed = load_named(dir, "model.visual.pos_embed.weight",
                              2304 * VISION_HIDDEN, error, error_size);
    if (!v->proj_w || !v->proj_b || !v->pos_embed) goto fail;

    for (int i = 0; i < VISION_BLOCKS; i++) {
        struct { const char *suffix; size_t elements; float **dst; } items[] = {
            {".norm1.weight", VISION_HIDDEN, &v->block[i].norm1_w},
            {".norm1.bias", VISION_HIDDEN, &v->block[i].norm1_b},
            {".attn.qkv.weight", 3 * VISION_HIDDEN * VISION_HIDDEN, &v->block[i].qkv_w},
            {".attn.qkv.bias", 3 * VISION_HIDDEN, &v->block[i].qkv_b},
            {".attn.proj.weight", VISION_HIDDEN * VISION_HIDDEN, &v->block[i].proj_w},
            {".attn.proj.bias", VISION_HIDDEN, &v->block[i].proj_b},
            {".norm2.weight", VISION_HIDDEN, &v->block[i].norm2_w},
            {".norm2.bias", VISION_HIDDEN, &v->block[i].norm2_b},
            {".mlp.linear_fc1.weight", VISION_MLP * VISION_HIDDEN, &v->block[i].fc1_w},
            {".mlp.linear_fc1.bias", VISION_MLP, &v->block[i].fc1_b},
            {".mlp.linear_fc2.weight", VISION_HIDDEN * VISION_MLP, &v->block[i].fc2_w},
            {".mlp.linear_fc2.bias", VISION_HIDDEN, &v->block[i].fc2_b},
        };
        for (size_t k = 0; k < sizeof(items) / sizeof(items[0]); k++) {
            snprintf(name, sizeof(name), "model.visual.blocks.%d%s", i,
                     items[k].suffix);
            *items[k].dst = load_named(dir, name, items[k].elements, error,
                                       error_size);
            if (!*items[k].dst) goto fail;
        }
    }

    {
        struct { const char *suffix; size_t elements; float **dst; } merger[] = {
            {".norm.weight", VISION_HIDDEN, &v->merger_norm_w},
            {".norm.bias", VISION_HIDDEN, &v->merger_norm_b},
            {".linear_fc1.weight", VISION_MERGED_IN * VISION_MERGED_IN, &v->merger_fc1_w},
            {".linear_fc1.bias", VISION_MERGED_IN, &v->merger_fc1_b},
            {".linear_fc2.weight", VISION_OUT * VISION_MERGED_IN, &v->merger_fc2_w},
            {".linear_fc2.bias", VISION_OUT, &v->merger_fc2_b},
        };
        for (size_t k = 0; k < sizeof(merger) / sizeof(merger[0]); k++) {
            snprintf(name, sizeof(name), "model.visual.merger%s", merger[k].suffix);
            *merger[k].dst = load_named(dir, name, merger[k].elements, error,
                                        error_size);
            if (!*merger[k].dst) goto fail;
        }
        for (int d = 0; d < VISION_DEEPSTACK; d++) {
            struct { const char *suffix; size_t elements; float **dst; } deep[] = {
                {".norm.weight", VISION_MERGED_IN, &v->deep[d].norm_w},
                {".norm.bias", VISION_MERGED_IN, &v->deep[d].norm_b},
                {".linear_fc1.weight", VISION_MERGED_IN * VISION_MERGED_IN, &v->deep[d].fc1_w},
                {".linear_fc1.bias", VISION_MERGED_IN, &v->deep[d].fc1_b},
                {".linear_fc2.weight", VISION_OUT * VISION_MERGED_IN, &v->deep[d].fc2_w},
                {".linear_fc2.bias", VISION_OUT, &v->deep[d].fc2_b},
            };
            for (size_t k = 0; k < sizeof(deep) / sizeof(deep[0]); k++) {
                snprintf(name, sizeof(name),
                         "model.visual.deepstack_merger_list.%d%s", d,
                         deep[k].suffix);
                *deep[k].dst = load_named(dir, name, deep[k].elements, error,
                                          error_size);
                if (!*deep[k].dst) goto fail;
            }
        }
    }
    if (v->use_gpu) {
        v->g_proj_w = up_bf16(gpu, v->proj_w,
                              (size_t)VISION_HIDDEN * VISION_PATCH_IN);
        v->g_proj_b = up_bf16(gpu, v->proj_b, VISION_HIDDEN);
        if (!v->g_proj_w || !v->g_proj_b) goto fail;
        for (int i = 0; i < VISION_BLOCKS; i++) {
            struct {
                const float *src;
                size_t n;
                h3_gpu_tensor **dst;
            } up[] = {
                {v->block[i].norm1_w, VISION_HIDDEN, &v->g_block[i].norm1_w},
                {v->block[i].norm1_b, VISION_HIDDEN, &v->g_block[i].norm1_b},
                {v->block[i].qkv_w, 3ull * VISION_HIDDEN * VISION_HIDDEN,
                 &v->g_block[i].qkv_w},
                {v->block[i].qkv_b, 3ull * VISION_HIDDEN, &v->g_block[i].qkv_b},
                {v->block[i].proj_w, 1ull * VISION_HIDDEN * VISION_HIDDEN,
                 &v->g_block[i].proj_w},
                {v->block[i].proj_b, VISION_HIDDEN, &v->g_block[i].proj_b},
                {v->block[i].norm2_w, VISION_HIDDEN, &v->g_block[i].norm2_w},
                {v->block[i].norm2_b, VISION_HIDDEN, &v->g_block[i].norm2_b},
                {v->block[i].fc1_w, 1ull * VISION_MLP * VISION_HIDDEN,
                 &v->g_block[i].fc1_w},
                {v->block[i].fc1_b, VISION_MLP, &v->g_block[i].fc1_b},
                {v->block[i].fc2_w, 1ull * VISION_HIDDEN * VISION_MLP,
                 &v->g_block[i].fc2_w},
                {v->block[i].fc2_b, VISION_HIDDEN, &v->g_block[i].fc2_b},
            };
            for (size_t k = 0; k < sizeof(up) / sizeof(up[0]); k++) {
                *up[k].dst = up_bf16(gpu, up[k].src, up[k].n);
                if (!*up[k].dst) goto fail;
            }
        }
        v->g_merger_norm_w = up_bf16(gpu, v->merger_norm_w, VISION_HIDDEN);
        v->g_merger_norm_b = up_bf16(gpu, v->merger_norm_b, VISION_HIDDEN);
        v->g_merger_fc1_w = up_bf16(
            gpu, v->merger_fc1_w,
            (size_t)VISION_MERGED_IN * VISION_MERGED_IN);
        v->g_merger_fc1_b = up_bf16(gpu, v->merger_fc1_b, VISION_MERGED_IN);
        v->g_merger_fc2_w = up_bf16(
            gpu, v->merger_fc2_w, (size_t)VISION_OUT * VISION_MERGED_IN);
        v->g_merger_fc2_b = up_bf16(gpu, v->merger_fc2_b, VISION_OUT);
        if (!v->g_merger_norm_w || !v->g_merger_norm_b || !v->g_merger_fc1_w ||
            !v->g_merger_fc1_b || !v->g_merger_fc2_w || !v->g_merger_fc2_b)
            goto fail;
        for (int d = 0; d < VISION_DEEPSTACK; d++) {
            v->g_deep[d].norm_w = up_bf16(gpu, v->deep[d].norm_w,
                                          VISION_MERGED_IN);
            v->g_deep[d].norm_b = up_bf16(gpu, v->deep[d].norm_b,
                                          VISION_MERGED_IN);
            v->g_deep[d].fc1_w = up_bf16(
                gpu, v->deep[d].fc1_w,
                (size_t)VISION_MERGED_IN * VISION_MERGED_IN);
            v->g_deep[d].fc1_b = up_bf16(gpu, v->deep[d].fc1_b,
                                         VISION_MERGED_IN);
            v->g_deep[d].fc2_w = up_bf16(
                gpu, v->deep[d].fc2_w, (size_t)VISION_OUT * VISION_MERGED_IN);
            v->g_deep[d].fc2_b = up_bf16(gpu, v->deep[d].fc2_b, VISION_OUT);
            if (!v->g_deep[d].norm_w || !v->g_deep[d].norm_b ||
                !v->g_deep[d].fc1_w || !v->g_deep[d].fc1_b ||
                !v->g_deep[d].fc2_w || !v->g_deep[d].fc2_b)
                goto fail;
        }
    }
    if (v->use_gpu) free_host_weights(v);
    return v;

fail:
    qw_vision_free(v);
    return NULL;
}

void qw_vision_free(qw_vision *v) {
    if (!v) return;
    free(v->proj_w);
    free(v->proj_b);
    free(v->pos_embed);
    for (int i = 0; i < VISION_BLOCKS; i++) {
        free(v->block[i].norm1_w); free(v->block[i].norm1_b);
        free(v->block[i].qkv_w); free(v->block[i].qkv_b);
        free(v->block[i].proj_w); free(v->block[i].proj_b);
        free(v->block[i].norm2_w); free(v->block[i].norm2_b);
        free(v->block[i].fc1_w); free(v->block[i].fc1_b);
        free(v->block[i].fc2_w); free(v->block[i].fc2_b);
    }
    free(v->merger_norm_w); free(v->merger_norm_b);
    free(v->merger_fc1_w); free(v->merger_fc1_b);
    free(v->merger_fc2_w); free(v->merger_fc2_b);
    for (int i = 0; i < VISION_DEEPSTACK; i++) {
        free(v->deep[i].norm_w); free(v->deep[i].norm_b);
        free(v->deep[i].fc1_w); free(v->deep[i].fc1_b);
        free(v->deep[i].fc2_w); free(v->deep[i].fc2_b);
    }
    if (v->gpu) {
        h3_gpu_tensor_free(v->g_proj_w);
        h3_gpu_tensor_free(v->g_proj_b);
        for (int i = 0; i < VISION_BLOCKS; i++) {
            h3_gpu_tensor_free(v->g_block[i].norm1_w);
            h3_gpu_tensor_free(v->g_block[i].norm1_b);
            h3_gpu_tensor_free(v->g_block[i].qkv_w);
            h3_gpu_tensor_free(v->g_block[i].qkv_b);
            h3_gpu_tensor_free(v->g_block[i].proj_w);
            h3_gpu_tensor_free(v->g_block[i].proj_b);
            h3_gpu_tensor_free(v->g_block[i].norm2_w);
            h3_gpu_tensor_free(v->g_block[i].norm2_b);
            h3_gpu_tensor_free(v->g_block[i].fc1_w);
            h3_gpu_tensor_free(v->g_block[i].fc1_b);
            h3_gpu_tensor_free(v->g_block[i].fc2_w);
            h3_gpu_tensor_free(v->g_block[i].fc2_b);
        }
        h3_gpu_tensor_free(v->g_merger_norm_w);
        h3_gpu_tensor_free(v->g_merger_norm_b);
        h3_gpu_tensor_free(v->g_merger_fc1_w);
        h3_gpu_tensor_free(v->g_merger_fc1_b);
        h3_gpu_tensor_free(v->g_merger_fc2_w);
        h3_gpu_tensor_free(v->g_merger_fc2_b);
        for (int i = 0; i < VISION_DEEPSTACK; i++) {
            h3_gpu_tensor_free(v->g_deep[i].norm_w);
            h3_gpu_tensor_free(v->g_deep[i].norm_b);
            h3_gpu_tensor_free(v->g_deep[i].fc1_w);
            h3_gpu_tensor_free(v->g_deep[i].fc1_b);
            h3_gpu_tensor_free(v->g_deep[i].fc2_w);
            h3_gpu_tensor_free(v->g_deep[i].fc2_b);
        }
    }
    free(v);
}

/* ---- 数学辅助 ---- */

/* out[n, out_dim] = in[n, in_dim] x W[out_dim, in_dim]^T + b（行主序权重） */
#include <Accelerate/Accelerate.h>

static void gemm(const float *in, const float *w, const float *b,
                 float *out, size_t n, size_t in_dim, size_t out_dim) {
    /* out = in × wᵀ + b（w [out_dim, in_dim] 行主序） */
    if (n)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)n, (int)out_dim, (int)in_dim, 1.0f, in, (int)in_dim,
                    w, (int)in_dim, 0.0f, out, (int)out_dim);
    if (b)
        for (size_t i = 0; i < n; i++)
            for (size_t od = 0; od < out_dim; od++)
                out[i * out_dim + od] += b[od];
}

static void layer_norm(float *x, const float *w, const float *b, size_t n,
                       size_t dim, float eps) {
    for (size_t i = 0; i < n; i++) {
        float *row = x + i * dim;
        float mean = 0.0f;
        for (size_t d = 0; d < dim; d++) mean += row[d];
        mean /= (float)dim;
        float var = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            float diff = row[d] - mean;
            var += diff * diff;
        }
        var /= (float)dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (size_t d = 0; d < dim; d++)
            row[d] = (row[d] - mean) * inv * w[d] + b[d];
    }
}

static float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;  /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/* merge-block 布局的块内坐标（vision_utils 同款逆变换） */
static void merge_block_coords(size_t within, uint32_t w, uint32_t merge,
                               uint32_t *row, uint32_t *col) {
    uint32_t blocks_w = w / merge;
    uint32_t in_col = (uint32_t)(within % merge);
    uint32_t in_row = (uint32_t)((within / merge) % merge);
    uint32_t block_col = (uint32_t)((within / (merge * merge)) % blocks_w);
    uint32_t block_row = (uint32_t)(within / (merge * merge * blocks_w));
    *row = block_row * merge + in_row;
    *col = block_col * merge + in_col;
}

/* 2×2 视觉 token 合并 MLP：x [iseq, 1152] → 视图 [iseq/4, 4608] →
 * （post_norm ? LN(4608) : 先逐 token LN(1152)）→ fc1 → GELU → fc2。
 * pre_norm 的 LN 权重维度为 1152，post 为 4608。
 * 返回 0 = 分配失败（此前静默 return，调用方会把未初始化的 deep_out
 * 当有效结果消费）。 */
static int merged_mlp(const float *x, size_t iseq, const float *norm_w,
                      const float *norm_b, const float *fc1_w,
                      const float *fc1_b, const float *fc2_w,
                      const float *fc2_b, float *out, int post_norm) {
    size_t merged = iseq / 4;
    float *joint = malloc(sizeof(float) * merged * VISION_MERGED_IN);
    float *mid = malloc(sizeof(float) * merged * VISION_MERGED_IN);
    if (!joint || !mid) {
        free(joint);
        free(mid);
        return 0;
    }
    if (post_norm) {
        memcpy(joint, x, sizeof(float) * merged * VISION_MERGED_IN);
        layer_norm(joint, norm_w, norm_b, merged, VISION_MERGED_IN, 1e-6f);
    } else {
        float *tmp = malloc(sizeof(float) * iseq * VISION_HIDDEN);
        if (tmp) {
            memcpy(tmp, x, sizeof(float) * iseq * VISION_HIDDEN);
            layer_norm(tmp, norm_w, norm_b, iseq, VISION_HIDDEN, 1e-6f);
            memcpy(joint, tmp, sizeof(float) * merged * VISION_MERGED_IN);
            free(tmp);
        } else {
            /* pre-norm needs the full-width normalized copy; without it
             * there is nothing valid to write. */
            free(joint);
            free(mid);
            return 0;
        }
    }
    gemm(joint, fc1_w, fc1_b, mid, merged, VISION_MERGED_IN, VISION_MERGED_IN);
    for (size_t i = 0; i < merged * VISION_MERGED_IN; i++)
        mid[i] = gelu_tanh(mid[i]);
    gemm(mid, fc2_w, fc2_b, out, merged, VISION_MERGED_IN, VISION_OUT);
    free(joint);
    free(mid);
    return 1;
}

/* pos_embed 双线性插值（align_corners=true, side 48），输出 [iseq, 1152]。
 * GPU/CPU 两路共用。 */
static void vision_pos_embed(const qw_vision *v, size_t iseq, uint32_t h,
                             uint32_t w, float *pos_add) {
    for (size_t i = 0; i < iseq; i++) {
        uint32_t row, col;
        merge_block_coords(i, w, 2, &row, &col);
        float h_src = (float)row * (VISION_POS_SIDE - 1) / (float)(h - 1);
        float w_src = (float)col * (VISION_POS_SIDE - 1) / (float)(w - 1);
        uint32_t h_floor = (uint32_t)floorf(h_src);
        uint32_t w_floor = (uint32_t)floorf(w_src);
        float fy = h_src - (float)h_floor;
        float fx = w_src - (float)w_floor;
        for (int d = 0; d < VISION_HIDDEN; d++)
            pos_add[i * VISION_HIDDEN + d] = 0.0f;
        /* 四角权重 [左上 (1-fy)(1-fx) | 右上 (1-fy)fx |
         *           左下 fy(1-fx)      | 右下 fy·fx]，tap 顺序同参考 */
        for (int k = 0; k < 4; k++) {
            uint32_t hy = h_floor + (uint32_t)(k >= 2);
            uint32_t wx = w_floor + (uint32_t)(k % 2);
            if (hy >= VISION_POS_SIDE) hy = VISION_POS_SIDE - 1;
            if (wx >= VISION_POS_SIDE) wx = VISION_POS_SIDE - 1;
            float wy = (k >= 2) ? fy : 1.0f - fy;
            float ww = (k % 2) ? fx : 1.0f - fx;
            float weight = wy * ww;
            if (weight == 0.0f) continue;
            const float *pos_row = v->pos_embed +
                ((size_t)hy * VISION_POS_SIDE + wx) * VISION_HIDDEN;
            for (int d = 0; d < VISION_HIDDEN; d++)
                pos_add[i * VISION_HIDDEN + d] += weight * pos_row[d];
        }
    }
}

/* GPU 单图段：hidden [iseq,1536] fp32 host → merged/deepstack 写入
 * 预分配的 GPU 大 tensor（merged_off/deep_off 为段偏移，单位为行）。 */
static int vision_segment_gpu(qw_vision *v, const float *hidden, uint32_t h,
                              uint32_t w, h3_gpu_tensor *g_merged_all,
                              size_t merged_off, h3_gpu_tensor *g_deep_all,
                              size_t deep_total, size_t deep_off,
                              char *error, size_t error_size) {
    h3_gpu *gpu = v->gpu;
    size_t iseq = (size_t)h * w;
    size_t merged = iseq / 4;
    static const int deep_at[VISION_DEEPSTACK] = VISION_DEEP_BLOCKS;
    int ok = 1;

    /* patch embed：hidden 上传 → cast bf16 → linear */
    h3_gpu_tensor *g_in_f32 =
        h3_gpu_tensor_new_f32(gpu, iseq * VISION_PATCH_IN);
    h3_gpu_tensor *g_in =
        h3_gpu_tensor_new_bf16(gpu, iseq * VISION_PATCH_IN);
    h3_gpu_tensor *g_h = h3_gpu_tensor_new_bf16(gpu, iseq * VISION_HIDDEN);
    h3_gpu_tensor *g_pos = h3_gpu_tensor_new_bf16(gpu, iseq * VISION_HIDDEN);
    h3_gpu_tensor *g_cos = h3_gpu_tensor_new_f32(
        gpu, iseq * (VISION_HEAD_DIM / 2));
    h3_gpu_tensor *g_sin = h3_gpu_tensor_new_f32(
        gpu, iseq * (VISION_HEAD_DIM / 2));
    h3_gpu_tensor *g_norm = h3_gpu_tensor_new_bf16(gpu, iseq * VISION_HIDDEN);
    h3_gpu_tensor *g_qkv = h3_gpu_tensor_new_bf16(gpu, iseq * 3 * VISION_HIDDEN);
    h3_gpu_tensor *g_q = h3_gpu_tensor_new_bf16(gpu, iseq * VISION_HIDDEN);
    h3_gpu_tensor *g_k = h3_gpu_tensor_new_bf16(gpu, iseq * VISION_HIDDEN);
    h3_gpu_tensor *g_v = h3_gpu_tensor_new_bf16(gpu, iseq * VISION_HIDDEN);
    h3_gpu_tensor *g_attn = h3_gpu_tensor_new_bf16(gpu, iseq * VISION_HIDDEN);
    h3_gpu_tensor *g_tmp = h3_gpu_tensor_new_bf16(gpu, iseq * VISION_HIDDEN);
    h3_gpu_tensor *g_mlp = h3_gpu_tensor_new_bf16(gpu, iseq * VISION_MLP);
    h3_gpu_tensor *g_joint = h3_gpu_tensor_new_bf16(gpu, merged * VISION_MERGED_IN);
    h3_gpu_tensor *g_ln4 = h3_gpu_tensor_new_bf16(gpu, merged * VISION_MERGED_IN);
    h3_gpu_tensor *g_mid = h3_gpu_tensor_new_bf16(gpu, merged * VISION_MERGED_IN);
    h3_gpu_tensor *g_seg = h3_gpu_tensor_new_bf16(gpu, merged * VISION_OUT);
    ok = g_in_f32 && g_in && g_h && g_pos && g_cos && g_sin && g_norm &&
         g_qkv && g_q && g_k && g_v && g_attn && g_tmp && g_mlp && g_joint &&
         g_ln4 && g_mid && g_seg;
    ok = ok && h3_gpu_tensor_write_f32(g_in_f32, hidden,
                                       iseq * VISION_PATCH_IN);
    ok = ok && h3_gpu_cast_f32_to_bf16(gpu, g_in, g_in_f32,
                                       (uint32_t)(iseq * VISION_PATCH_IN));
    h3_gpu_tensor_free(g_in_f32);
    g_in_f32 = NULL;

    /* patch embed + pos_embed 相加（插值在 CPU，[iseq,1152] fp32）。 */
    ok = ok && h3_gpu_linear_bf16(gpu, g_h, g_in, v->g_proj_w, v->g_proj_b,
                                  (uint32_t)iseq, VISION_PATCH_IN, VISION_HIDDEN);
    h3_gpu_tensor_free(g_in);
    g_in = NULL;
    {
        float *pos_add = malloc(sizeof(float) * iseq * VISION_HIDDEN);
        ok = ok && pos_add;
        if (ok) {
            vision_pos_embed(v, iseq, h, w, pos_add);
            uint16_t *bits = malloc(sizeof(uint16_t) * iseq * VISION_HIDDEN);
            ok = ok && bits;
            if (ok) {
                for (size_t i = 0; i < iseq * VISION_HIDDEN; i++)
                    bits[i] = f32_to_bf16_bits(pos_add[i]);
                ok = h3_gpu_tensor_write_bf16(g_pos, bits,
                                              iseq * VISION_HIDDEN);
            }
            free(bits);
        }
        free(pos_add);
        ok = ok && qw_gpu_vision_add_bf16(gpu, g_h, g_h, g_pos,
                                          (uint32_t)(iseq * VISION_HIDDEN));
    }
    h3_gpu_tensor_free(g_pos);
    g_pos = NULL;

    /* 轴向 rope 表（[iseq, 36] fp32，唯一角 [h18|w18]；fp64 角度同 CPU 版）。 */
    {
        float *cos_row = malloc(sizeof(float) * iseq * (VISION_HEAD_DIM / 2));
        float *sin_row = malloc(sizeof(float) * iseq * (VISION_HEAD_DIM / 2));
        ok = ok && cos_row && sin_row;
        if (ok) {
            for (size_t i = 0; i < iseq; i++) {
                uint32_t row, col;
                merge_block_coords(i, w, 2, &row, &col);
                for (int f = 0; f < VISION_ROPE_FREQ; f++) {
                    double fh = (double)row * (double)v->rope_inv_freq[f];
                    double fw = (double)col * (double)v->rope_inv_freq[f];
                    cos_row[i * (VISION_HEAD_DIM / 2) + f] = (float)cos(fh);
                    sin_row[i * (VISION_HEAD_DIM / 2) + f] = (float)sin(fh);
                    cos_row[i * (VISION_HEAD_DIM / 2) + VISION_ROPE_FREQ + f] =
                        (float)cos(fw);
                    sin_row[i * (VISION_HEAD_DIM / 2) + VISION_ROPE_FREQ + f] =
                        (float)sin(fw);
                }
            }
            ok = h3_gpu_tensor_write_f32(g_cos, cos_row,
                                         iseq * (VISION_HEAD_DIM / 2)) &&
                 h3_gpu_tensor_write_f32(g_sin, sin_row,
                                         iseq * (VISION_HEAD_DIM / 2));
        }
        free(cos_row);
        free(sin_row);
    }

    const float scale = 1.0f / sqrtf((float)VISION_HEAD_DIM);
    for (int bi = 0; ok && bi < VISION_BLOCKS; bi++) {
        ok = ok && qw_gpu_vision_ln_bf16(gpu, g_norm, g_h, v->g_block[bi].norm1_w,
                                         v->g_block[bi].norm1_b, (uint32_t)iseq,
                                         VISION_HIDDEN, 1e-6f);
        ok = ok && h3_gpu_linear_bf16(gpu, g_qkv, g_norm, v->g_block[bi].qkv_w,
                                      v->g_block[bi].qkv_b, (uint32_t)iseq,
                                      VISION_HIDDEN, 3 * VISION_HIDDEN);
        ok = ok && qw_gpu_vision_split3_bf16(gpu, g_q, g_qkv, (uint32_t)iseq,
                                             VISION_HIDDEN, 0) &&
             qw_gpu_vision_split3_bf16(gpu, g_k, g_qkv, (uint32_t)iseq,
                                       VISION_HIDDEN, 1) &&
             qw_gpu_vision_split3_bf16(gpu, g_v, g_qkv, (uint32_t)iseq,
                                       VISION_HIDDEN, 2);
        /* q/k 就地 rope（内核成对读后写，in-place 安全）。 */
        ok = ok && qw_gpu_vision_rope_bf16(gpu, g_q, g_q, g_cos, g_sin, (uint32_t)iseq,
                                           VISION_HEADS, VISION_HEAD_DIM) &&
             qw_gpu_vision_rope_bf16(gpu, g_k, g_k, g_cos, g_sin, (uint32_t)iseq,
                                     VISION_HEADS, VISION_HEAD_DIM);
        ok = ok && qw_gpu_sdpa_rect_bf16(gpu, g_attn, g_q, g_k, g_v,
                                         (uint32_t)iseq, (uint32_t)iseq,
                                         VISION_HEADS, VISION_HEAD_DIM,
                                         scale);
        ok = ok && h3_gpu_linear_bf16(gpu, g_tmp, g_attn,
                                      v->g_block[bi].proj_w,
                                      v->g_block[bi].proj_b, (uint32_t)iseq,
                                      VISION_HIDDEN, VISION_HIDDEN);
        ok = ok && qw_gpu_vision_add_bf16(gpu, g_h, g_h, g_tmp,
                                          (uint32_t)(iseq * VISION_HIDDEN));
        ok = ok && qw_gpu_vision_ln_bf16(gpu, g_norm, g_h,
                                         v->g_block[bi].norm2_w,
                                         v->g_block[bi].norm2_b, (uint32_t)iseq,
                                         VISION_HIDDEN, 1e-6f);
        ok = ok && h3_gpu_linear_bf16(gpu, g_mlp, g_norm, v->g_block[bi].fc1_w,
                                      v->g_block[bi].fc1_b, (uint32_t)iseq,
                                      VISION_HIDDEN, VISION_MLP);
        ok = ok && qw_gpu_vision_gelu_bf16(gpu, g_mlp, g_mlp,
                                           (uint32_t)(iseq * VISION_MLP));
        ok = ok && h3_gpu_linear_bf16(gpu, g_tmp, g_mlp, v->g_block[bi].fc2_w,
                                      v->g_block[bi].fc2_b, (uint32_t)iseq, VISION_MLP,
                                      VISION_HIDDEN);
        ok = ok && qw_gpu_vision_add_bf16(gpu, g_h, g_h, g_tmp,
                                          (uint32_t)(iseq * VISION_HIDDEN));
        if (bi == deep_at[0] || bi == deep_at[1] || bi == deep_at[2]) {
            int idx = bi == deep_at[0] ? 0 : bi == deep_at[1] ? 1 : 2;
            ok = ok && qw_gpu_vision_merge2x2_bf16(gpu, g_joint, g_h, (uint32_t)iseq,
                                                   VISION_HIDDEN);
            ok = ok && qw_gpu_vision_ln_bf16(gpu, g_ln4, g_joint,
                                             v->g_deep[idx].norm_w,
                                             v->g_deep[idx].norm_b, (uint32_t)merged,
                                             VISION_MERGED_IN, 1e-6f);
            ok = ok && h3_gpu_linear_bf16(gpu, g_mid, g_ln4,
                                          v->g_deep[idx].fc1_w,
                                          v->g_deep[idx].fc1_b, (uint32_t)merged,
                                          VISION_MERGED_IN, VISION_MERGED_IN);
            ok = ok && qw_gpu_vision_gelu_bf16(gpu, g_mid, g_mid,
                                               (uint32_t)(merged * VISION_MERGED_IN));
            ok = ok && h3_gpu_linear_bf16(gpu, g_tmp, g_mid,
                                          v->g_deep[idx].fc2_w,
                                          v->g_deep[idx].fc2_b, (uint32_t)merged,
                                          VISION_MERGED_IN, VISION_OUT);
            /* 输出段写回 deep_all（[3, deep_total, 4096] 布局）。 */
            ok = ok && h3_gpu_copy_bf16(
                           gpu, g_deep_all,
                           ((size_t)idx * deep_total + deep_off) * VISION_OUT,
                           g_tmp, 0, merged * VISION_OUT);
        }
    }

    /* merger：LN(1152) → 2×2 合并 → fc1 → GELU → fc2 → merged_all 段。 */
    ok = ok && qw_gpu_vision_ln_bf16(gpu, g_norm, g_h, v->g_merger_norm_w,
                                     v->g_merger_norm_b, (uint32_t)iseq, VISION_HIDDEN,
                                     1e-6f);
    ok = ok && qw_gpu_vision_merge2x2_bf16(gpu, g_joint, g_norm, (uint32_t)iseq,
                                           VISION_HIDDEN);
    ok = ok && h3_gpu_linear_bf16(gpu, g_mid, g_joint, v->g_merger_fc1_w,
                                  v->g_merger_fc1_b, (uint32_t)merged, VISION_MERGED_IN,
                                  VISION_MERGED_IN);
    ok = ok && qw_gpu_vision_gelu_bf16(gpu, g_mid, g_mid,
                                       (uint32_t)(merged * VISION_MERGED_IN));
    ok = ok && h3_gpu_linear_bf16(gpu, g_seg, g_mid, v->g_merger_fc2_w,
                                  v->g_merger_fc2_b, (uint32_t)merged, VISION_MERGED_IN,
                                  VISION_OUT);
    ok = ok && h3_gpu_copy_bf16(gpu, g_merged_all, merged_off * VISION_OUT,
                                g_seg, 0, merged * VISION_OUT);

    /* g_h is the running [iseq, 1152] activation the whole tower reads
     * and writes; it was the only workspace never released, leaking one
     * such buffer per ti2i request in a resident session. */
    h3_gpu_tensor_free(g_h);
    h3_gpu_tensor_free(g_cos);
    h3_gpu_tensor_free(g_sin);
    h3_gpu_tensor_free(g_norm);
    h3_gpu_tensor_free(g_qkv);
    h3_gpu_tensor_free(g_q);
    h3_gpu_tensor_free(g_k);
    h3_gpu_tensor_free(g_v);
    h3_gpu_tensor_free(g_attn);
    h3_gpu_tensor_free(g_tmp);
    h3_gpu_tensor_free(g_mlp);
    h3_gpu_tensor_free(g_joint);
    h3_gpu_tensor_free(g_ln4);
    h3_gpu_tensor_free(g_mid);
    h3_gpu_tensor_free(g_seg);
    if (!ok)
        snprintf(error, error_size, "vision: gpu forward failed: %s",
                 h3_gpu_error(gpu));
    return ok;
}

/* 单图段完整塔前向（图间注意力隔离，逐图独立）。deep_out[0..2] 为调用方
 * 预置的 [merged, 4096] 段指针。 */
static int vision_segment(qw_vision *v, const float *hidden, uint32_t h,
                          uint32_t w, float *merged_out, float **deep_out,
                          char *error, size_t error_size) {
    size_t iseq = (size_t)h * w;
    float *h_state = malloc(sizeof(float) * iseq * VISION_HIDDEN);
    float *qkv = malloc(sizeof(float) * iseq * 3 * VISION_HIDDEN);
    float *mlp_buf = malloc(sizeof(float) * iseq * VISION_MLP);
    float *attn_buf = malloc(sizeof(float) * iseq * VISION_HIDDEN);
    float *cos_row = malloc(sizeof(float) * iseq * VISION_HEAD_DIM);
    float *sin_row = malloc(sizeof(float) * iseq * VISION_HEAD_DIM);
    float *scores = malloc(sizeof(float) * iseq);
    float *pos_add = malloc(sizeof(float) * iseq * VISION_HIDDEN);
    float *norm_buf = malloc(sizeof(float) * iseq * VISION_HIDDEN);
    if (!h_state || !qkv || !mlp_buf || !attn_buf || !cos_row || !sin_row ||
        !scores || !pos_add || !norm_buf) {
        free(h_state); free(qkv); free(mlp_buf); free(attn_buf);
        free(cos_row); free(sin_row); free(scores); free(pos_add);
        free(norm_buf);
        snprintf(error, error_size, "vision: out of memory");
        return 0;
    }

    /* 1. patch embed */
    gemm(hidden, v->proj_w, v->proj_b, h_state, iseq, VISION_PATCH_IN,
         VISION_HIDDEN);

    dump_f32(v, "vision_patch_embed", h_state, iseq * VISION_HIDDEN);

    /* 2. pos_embed 双线性插值（align_corners=true, side 48）并相加 */
    for (size_t i = 0; i < iseq; i++) {
        uint32_t row, col;
        merge_block_coords(i, w, 2, &row, &col);
        float h_src = (float)row * (VISION_POS_SIDE - 1) / (float)(h - 1);
        float w_src = (float)col * (VISION_POS_SIDE - 1) / (float)(w - 1);
        uint32_t h_floor = (uint32_t)floorf(h_src);
        uint32_t w_floor = (uint32_t)floorf(w_src);
        float fy = h_src - (float)h_floor;
        float fx = w_src - (float)w_floor;
        for (int d = 0; d < VISION_HIDDEN; d++)
            pos_add[i * VISION_HIDDEN + d] = 0.0f;
        /* 四角权重 [左上 (1-fy)(1-fx) | 右上 (1-fy)fx |
         *           左下 fy(1-fx)      | 右下 fy·fx]，tap 顺序同参考 */
        for (int k = 0; k < 4; k++) {
            uint32_t hy = h_floor + (uint32_t)(k >= 2);
            uint32_t wx = w_floor + (uint32_t)(k % 2);
            if (hy >= VISION_POS_SIDE) hy = VISION_POS_SIDE - 1;
            if (wx >= VISION_POS_SIDE) wx = VISION_POS_SIDE - 1;
            float wy = (k >= 2) ? fy : 1.0f - fy;
            float ww = (k % 2) ? fx : 1.0f - fx;
            float weight = wy * ww;
            if (weight == 0.0f) continue;
            const float *pos_row = v->pos_embed +
                ((size_t)hy * VISION_POS_SIDE + wx) * VISION_HIDDEN;
            for (int d = 0; d < VISION_HIDDEN; d++)
                pos_add[i * VISION_HIDDEN + d] += weight * pos_row[d];
        }
    }
    for (size_t i = 0; i < iseq * VISION_HIDDEN; i++) h_state[i] += pos_add[i];

    /* 3. 轴向 rope cos/sin（未合并 token，block-major 坐标）；
     *    角度走 double，与参考 float64 频率乘法同带 */
    for (size_t i = 0; i < iseq; i++) {
        uint32_t row, col;
        merge_block_coords(i, w, 2, &row, &col);
        for (int f = 0; f < VISION_ROPE_FREQ; f++) {
            double fh = (double)row * (double)v->rope_inv_freq[f];
            double fw = (double)col * (double)v->rope_inv_freq[f];
            float ch = (float)cos(fh);
            float sh = (float)sin(fh);
            float cw = (float)cos(fw);
            float sw = (float)sin(fw);
            float *c = cos_row + i * VISION_HEAD_DIM;
            float *s = sin_row + i * VISION_HEAD_DIM;
            c[f] = ch;
            s[f] = sh;
            c[VISION_ROPE_FREQ + f] = cw;
            s[VISION_ROPE_FREQ + f] = sw;
            c[VISION_HEAD_DIM / 2 + f] = ch;
            s[VISION_HEAD_DIM / 2 + f] = sh;
            c[VISION_HEAD_DIM / 2 + VISION_ROPE_FREQ + f] = cw;
            s[VISION_HEAD_DIM / 2 + VISION_ROPE_FREQ + f] = sw;
        }
    }

    dump_f32(v, "vision_rope_cos", cos_row, iseq * VISION_HEAD_DIM);
    dump_f32(v, "vision_rope_sin", sin_row, iseq * VISION_HEAD_DIM);

    /* 4. 27 blocks：norm1 → qkv → rope → 逐图注意力 → proj → 残差；
     *    norm2 → mlp → 残差；block 8/16/24 挂 deepstack */
    for (int bi = 0; bi < VISION_BLOCKS; bi++) {
        /* norm 就地会破坏残差底座（hidden = hidden + attn(norm1(hidden))），
         * LN 输出必须写独立缓冲，残差始终加回原始 h_state */
        memcpy(norm_buf, h_state, sizeof(float) * iseq * VISION_HIDDEN);
        layer_norm(norm_buf, v->block[bi].norm1_w, v->block[bi].norm1_b, iseq,
                   VISION_HIDDEN, 1e-6f);
        gemm(norm_buf, v->block[bi].qkv_w, v->block[bi].qkv_b, qkv, iseq,
             VISION_HIDDEN, 3 * VISION_HIDDEN);
        /* qkv 行内布局：reshape(seq, 3, heads, head_dim) → 行内即
         * [q(1152)|k(1152)|v(1152)] 段式，段内 head-major（permute 只换维度
         * 序不换内存布局，unbind 切片各自取 a=0/1/2） */
        for (size_t i = 0; i < iseq; i++) {
            const float *c = cos_row + i * VISION_HEAD_DIM;
            const float *s = sin_row + i * VISION_HEAD_DIM;
            for (int head = 0; head < VISION_HEADS; head++) {
                float *q = qkv + i * 3 * VISION_HIDDEN + head * VISION_HEAD_DIM;
                float *k = q + VISION_HIDDEN;
                /* 旋转对 (d, d+36)，cos/sin 布局 [h18|w18|h18|w18] 已保证
                 * 对内共享角度；必须成对处理（就地覆盖会破坏原值） */
                for (int d = 0; d < VISION_HEAD_DIM / 2; d++) {
                    int p = d + VISION_HEAD_DIM / 2;
                    float q0 = q[d], q1 = q[p];
                    q[d] = q0 * c[d] - q1 * s[d];
                    q[p] = q1 * c[p] + q0 * s[p];
                    float k0 = k[d], k1 = k[p];
                    k[d] = k0 * c[d] - k1 * s[d];
                    k[p] = k1 * c[p] + k0 * s[p];
                }
            }
        }
        /* 逐图注意力（单段 = 全序列，非因果） */
        float scale = 1.0f / sqrtf((float)VISION_HEAD_DIM);
        for (int head = 0; head < VISION_HEADS; head++) {
            for (size_t i = 0; i < iseq; i++) {
                const float *q = qkv + i * 3 * VISION_HIDDEN +
                                 head * VISION_HEAD_DIM;
                float max_score = -3.402823466e+38f;
                for (size_t j = 0; j < iseq; j++) {
                    const float *k = qkv + j * 3 * VISION_HIDDEN +
                                     VISION_HIDDEN + head * VISION_HEAD_DIM;
                    scores[j] =
                        cblas_sdot(VISION_HEAD_DIM, q, 1, k, 1) * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }
                float denom = 0.0f;
                for (size_t j = 0; j < iseq; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    denom += scores[j];
                }
                float *o = attn_buf + i * VISION_HIDDEN +
                           head * VISION_HEAD_DIM;
                for (int d = 0; d < VISION_HEAD_DIM; d++) o[d] = 0.0f;
                for (size_t j = 0; j < iseq; j++) {
                    const float *vec = qkv + j * 3 * VISION_HIDDEN +
                                       2 * VISION_HIDDEN + head * VISION_HEAD_DIM;
                    float weight = scores[j] / denom;
                    for (int d = 0; d < VISION_HEAD_DIM; d++)
                        o[d] += weight * vec[d];
                }
            }
        }
        gemm(attn_buf, v->block[bi].proj_w, v->block[bi].proj_b, qkv, iseq,
             VISION_HIDDEN, VISION_HIDDEN);
        for (size_t i = 0; i < iseq * VISION_HIDDEN; i++)
            h_state[i] += qkv[i];

        memcpy(norm_buf, h_state, sizeof(float) * iseq * VISION_HIDDEN);
        layer_norm(norm_buf, v->block[bi].norm2_w, v->block[bi].norm2_b, iseq,
                   VISION_HIDDEN, 1e-6f);
        gemm(norm_buf, v->block[bi].fc1_w, v->block[bi].fc1_b, mlp_buf, iseq,
             VISION_HIDDEN, VISION_MLP);
        for (size_t i = 0; i < iseq * VISION_MLP; i++)
            mlp_buf[i] = gelu_tanh(mlp_buf[i]);
        gemm(mlp_buf, v->block[bi].fc2_w, v->block[bi].fc2_b, attn_buf, iseq,
             VISION_MLP, VISION_HIDDEN);
        for (size_t i = 0; i < iseq * VISION_HIDDEN; i++)
            h_state[i] += attn_buf[i];

        {
            int deep_idx = bi == 8 ? 0 : bi == 16 ? 1 : bi == 24 ? 2 : -1;
            if (deep_idx >= 0 &&
                !merged_mlp(h_state, iseq, v->deep[deep_idx].norm_w,
                            v->deep[deep_idx].norm_b,
                            v->deep[deep_idx].fc1_w, v->deep[deep_idx].fc1_b,
                            v->deep[deep_idx].fc2_w, v->deep[deep_idx].fc2_b,
                            deep_out[deep_idx], 1)) {
                snprintf(error, error_size, "vision: out of memory");
                goto fail;
            }
        }
        {
            char tag[32];
            snprintf(tag, sizeof(tag), "vision_block%02d", bi);
            dump_f32(v, tag, h_state, iseq * VISION_HIDDEN);
        }
    }

    /* 5. merger（LN(1152) pre-shuffle → 2×2 合并 → fc1 → GELU → fc2） */
    {
        float *tmp = malloc(sizeof(float) * iseq * VISION_HIDDEN);
        if (!tmp) {
            snprintf(error, error_size, "vision: out of memory");
            goto fail;
        }
        memcpy(tmp, h_state, sizeof(float) * iseq * VISION_HIDDEN);
        layer_norm(tmp, v->merger_norm_w, v->merger_norm_b, iseq,
                   VISION_HIDDEN, 1e-6f);
        size_t merged = iseq / 4;
        float *joint = malloc(sizeof(float) * merged * VISION_MERGED_IN);
        float *mid = malloc(sizeof(float) * merged * VISION_MERGED_IN);
        if (!joint || !mid) {
            free(tmp); free(joint); free(mid);
            snprintf(error, error_size, "vision: out of memory");
            goto fail;
        }
        for (size_t m = 0; m < merged; m++)
            memcpy(joint + m * VISION_MERGED_IN, tmp + m * 4 * VISION_HIDDEN,
                   sizeof(float) * VISION_MERGED_IN);
        gemm(joint, v->merger_fc1_w, v->merger_fc1_b, mid, merged,
             VISION_MERGED_IN, VISION_MERGED_IN);
        for (size_t i = 0; i < merged * VISION_MERGED_IN; i++)
            mid[i] = gelu_tanh(mid[i]);
        gemm(mid, v->merger_fc2_w, v->merger_fc2_b, merged_out, merged,
             VISION_MERGED_IN, VISION_OUT);
        dump_f32(v, "vision_merger", merged_out, merged * VISION_OUT);
        for (int i = 0; i < VISION_DEEPSTACK; i++) {
            char tag[32];
            snprintf(tag, sizeof(tag), "vision_deepstack%d", i);
            dump_f32(v, tag, deep_out[i], merged * VISION_OUT);
        }
        free(joint);
        free(mid);
        free(tmp);
    }

    free(h_state);
    free(qkv);
    free(mlp_buf);
    free(attn_buf);
    free(cos_row);
    free(sin_row);
    free(scores);
    free(norm_buf);
    free(pos_add);
    return 1;
fail:
    free(h_state);
    free(qkv);
    free(mlp_buf);
    free(attn_buf);
    free(cos_row);
    free(sin_row);
    free(scores);
    free(norm_buf);
    free(pos_add);
    return 0;
}

/* GPU 大 tensor → host fp32（bf16 bits 读回 + 上采样）。 */
static int read_bf16_to_f32(h3_gpu_tensor *t, float *dst, size_t count) {
    uint16_t *bits = malloc(count * 2);
    if (!bits || !h3_gpu_tensor_read_bf16(t, bits, count)) {
        free(bits);
        return 0;
    }
    for (size_t i = 0; i < count; i++) {
        uint32_t wide = (uint32_t)bits[i] << 16;
        memcpy(&dst[i], &wide, 4);
    }
    free(bits);
    return 1;
}

static int vision_forward_gpu(qw_vision *v, const float *hidden,
                              const uint32_t *grid_thw, uint32_t n_images,
                              float *merged, float *deepstack, char *error,
                              size_t error_size) {
    h3_gpu *gpu = v->gpu;
    size_t merged_total = 0;
    for (uint32_t img = 0; img < n_images; img++) {
        uint32_t t = grid_thw[img * 3], h = grid_thw[img * 3 + 1],
                 w = grid_thw[img * 3 + 2];
        if (!t || !h || !w || h % 2 || w % 2) {
            snprintf(error, error_size, "vision forward: bad grid");
            return 0;
        }
        merged_total += (size_t)t * (h / 2) * (w / 2);
    }
    h3_gpu_tensor *g_merged_all =
        h3_gpu_tensor_new_bf16(gpu, merged_total * VISION_OUT);
    h3_gpu_tensor *g_deep_all = h3_gpu_tensor_new_bf16(
        gpu, (size_t)VISION_DEEPSTACK * merged_total * VISION_OUT);
    if (!g_merged_all || !g_deep_all) {
        h3_gpu_tensor_free(g_merged_all);
        h3_gpu_tensor_free(g_deep_all);
        snprintf(error, error_size, "vision: out of memory (gpu)");
        return 0;
    }
    /* 命令状态：调用方可能持有打开的缓冲（begin 失败=曾打开→submit 恢复
     * →begin），整段塔编码进一条命令，末尾统一 submit。 */
    if (!h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) {
        snprintf(error, error_size, "vision: begin failed: %s",
                 h3_gpu_error(gpu));
        h3_gpu_tensor_free(g_merged_all);
        h3_gpu_tensor_free(g_deep_all);
        return 0;
    }
    size_t patch_offset = 0, merged_offset = 0;
    int ok = 1;
    for (uint32_t img = 0; ok && img < n_images; img++) {
        uint32_t h = grid_thw[img * 3 + 1], w = grid_thw[img * 3 + 2];
        uint32_t t = grid_thw[img * 3];
        for (uint32_t tt = 0; ok && tt < t; tt++) {
            ok = vision_segment_gpu(
                v, hidden + patch_offset * VISION_PATCH_IN, h, w, g_merged_all,
                merged_offset, g_deep_all, merged_total, merged_offset, error,
                error_size);
            patch_offset += (size_t)h * w;
            merged_offset += (size_t)(h / 2) * (w / 2);
        }
    }
    ok = ok && h3_gpu_submit(gpu);
    ok = ok && read_bf16_to_f32(g_merged_all, merged,
                                merged_total * VISION_OUT);
    ok = ok && read_bf16_to_f32(g_deep_all, deepstack,
                                (size_t)VISION_DEEPSTACK * merged_total *
                                    VISION_OUT);
    h3_gpu_tensor_free(g_merged_all);
    h3_gpu_tensor_free(g_deep_all);
    if (!ok && !error[0])
        snprintf(error, error_size, "vision: gpu forward failed");
    return ok;
}

int qw_vision_forward(qw_vision *v, const float *hidden,
                      const uint32_t *grid_thw, uint32_t n_images,
                      float *merged, float *deepstack,
                      char *error, size_t error_size) {
    if (!v || !hidden || !grid_thw || !n_images || !merged || !deepstack) {
        snprintf(error, error_size, "vision forward: bad arguments");
        return 0;
    }
    if (v->use_gpu)
        return vision_forward_gpu(v, hidden, grid_thw, n_images, merged,
                                  deepstack, error, error_size);
    size_t merged_total = 0;
    for (uint32_t img = 0; img < n_images; img++) {
        uint32_t t = grid_thw[img * 3], h = grid_thw[img * 3 + 1],
                 w = grid_thw[img * 3 + 2];
        if (!t || !h || !w || h % 2 || w % 2) {
            snprintf(error, error_size, "vision forward: bad grid");
            return 0;
        }
        merged_total += (size_t)t * (h / 2) * (w / 2);
    }
    size_t patch_offset = 0, merged_offset = 0;
    for (uint32_t img = 0; img < n_images; img++) {
        uint32_t t = grid_thw[img * 3], h = grid_thw[img * 3 + 1],
                 w = grid_thw[img * 3 + 2];
        for (uint32_t tt = 0; tt < t; tt++) {
            float *deep_out[VISION_DEEPSTACK];
            for (int i = 0; i < VISION_DEEPSTACK; i++)
                deep_out[i] = deepstack +
                    ((size_t)i * merged_total + merged_offset) * VISION_OUT;
            if (!vision_segment(v, hidden + patch_offset * VISION_PATCH_IN, h,
                                w, merged + merged_offset * VISION_OUT,
                                deep_out, error, error_size))
                return 0;
            patch_offset += (size_t)h * w;
            merged_offset += (size_t)(h / 2) * (w / 2);
        }
    }
    return 1;
}
