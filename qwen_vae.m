// qwen_vae.m — Qwen-Image-2.1 VAE decoder, image path, fp32.
//
// The released AutoencoderKLQwenImage21 is a Wan-heritage causal 3D
// network, but the image specialization folds the temporal axis away:
// QwenImage21CausalConv3d squeezes T and applies a 2D convolution, and
// the first-call "Rep" branch of every upsample3d skips its time_conv
// entirely. Decoding a single latent frame is therefore a pure 2D
// network, which is what this implementation runs. All activations are
// fp32 (production casts the whole VAE to fp32) with channels-last
// [H, W, C] rows, matching the NDHWC MPSGraph convolution layout.

#import "qwen_vae.h"
#import "qwen_gpu_ext.h"
#import "h3_weights.h"
#import "qwen_json.h"

#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>

#define QWEN_VAE_OUT 4
#define QWEN_VAE_UP_BLOCKS 5
#define QWEN_VAE_RESNETS 3

typedef struct {
    h3_gpu_tensor *norm1;      // [in_dim]
    h3_gpu_tensor *conv1_w;    // [out, in, 1, 3, 3]
    h3_gpu_tensor *conv1_w9;   // [9, out_pad64, in] GEMM layout
    h3_gpu_tensor *conv1_b;    // [out]
    h3_gpu_tensor *norm2;      // [out_dim]
    h3_gpu_tensor *conv2_w;    // [out, out, 1, 3, 3]
    h3_gpu_tensor *conv2_w9;   // [9, out_pad64, out] GEMM layout
    h3_gpu_tensor *conv2_b;    // [out]
    h3_gpu_tensor *shortcut_w; // [out, in, 1, 1], NULL when Identity
    h3_gpu_tensor *shortcut_b; // [out]
} qw_vae_resnet;

typedef struct {
    qw_vae_resnet resnets[QWEN_VAE_RESNETS];
    int up_flag;               // upsampler + DupUp shortcut present
    uint32_t in_dim, out_dim;
    uint32_t factor_t, factor_s;
    h3_gpu_tensor *resample_w; // [out, in, 3, 3]
    h3_gpu_tensor *resample_w9; // [9, out_pad64, out] GEMM layout
    h3_gpu_tensor *resample_b; // [out]
} qw_vae_up_block;

struct qw_vae {
    h3_gpu *gpu;
    float latents_mean[QWEN_VAE_Z], latents_std[QWEN_VAE_Z];
    h3_gpu_tensor *post_quant_w, *post_quant_b;
    h3_gpu_tensor *conv_in_w, *conv_in_w9, *conv_in_b;
    qw_vae_resnet mid_resnets[2];
    h3_gpu_tensor *attn_norm;
    h3_gpu_tensor *attn_qkv_w; // fused [3C, C]
    h3_gpu_tensor *attn_qkv_b;
    h3_gpu_tensor *attn_proj_w, *attn_proj_b;
    qw_vae_up_block up_blocks[QWEN_VAE_UP_BLOCKS];
    h3_gpu_tensor *norm_out;
    h3_gpu_tensor *conv_out_w, *conv_out_w9, *conv_out_b;
    // encoder (ti2i condition latents)
    h3_gpu_tensor *enc_conv_in_w, *enc_conv_in_w9, *enc_conv_in_b;
    qw_vae_resnet enc_mid_resnets[2];
    h3_gpu_tensor *enc_attn_norm, *enc_attn_qkv_w, *enc_attn_qkv_b;
    h3_gpu_tensor *enc_attn_proj_w, *enc_attn_proj_b;
    h3_gpu_tensor *enc_norm_out;
    h3_gpu_tensor *enc_conv_out_w, *enc_conv_out_w9, *enc_conv_out_b;
    h3_gpu_tensor *quant_w, *quant_b;
    struct {
        qw_vae_resnet resnets[2];
        h3_gpu_tensor *down_w, *down_b; // stride-2 3x3 (downsample2d)
        int down_flag;                  // downsampler present
        uint32_t in_dim, out_dim;
        uint32_t group;                 // AvgDown3D group size
        uint32_t avg_ft;                // AvgDown3D factor_t
    } down_blocks[5];
};

// ---------------------------------------------------------------- debug

static int dump_tensor(h3_gpu *gpu, const char *dir, const char *name,
                       const h3_gpu_tensor *tensor, size_t count) {
    if (!dir || !*dir) return 1;
    if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) return 0;
    float *host = malloc(sizeof(float) * count);
    if (!host) return 0;
    if (!h3_gpu_tensor_read_f32(tensor, host, count)) {
        free(host);
        return 0;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *file = fopen(path, "wb");
    int ok = file != NULL;
    if (ok) ok = fwrite(host, sizeof(float), count, file) == count;
    if (file) fclose(file);
    free(host);
    return ok;
}

__attribute__((unused)) static void dump_host_f32(const char *dir, const char *name,
                          const float *data, size_t count) {
    if (!dir || !*dir) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *file = fopen(path, "wb");
    if (!file) return;
    fwrite(data, sizeof(float), count, file);
    fclose(file);
}

// ---------------------------------------------------------------- loader

/* The base checkpoint stores the VAE in F32, the turbo one in BF16;
 * accept either and normalize to F32 (the decoder is fp32 by design). */
static h3_gpu_tensor *load_w(h3_weight_store *store, h3_gpu *gpu,
                             const char *name, int ndim,
                             const uint64_t *shape, char *error,
                             size_t error_size) {
    h3_gpu_tensor *tensor = h3_weight_load_f32(store, gpu, name, ndim, shape,
                                               error, error_size);
    if (tensor) return tensor;
    error[0] = 0;
    tensor = h3_weight_load_bf16(store, gpu, name, ndim, shape, error,
                                 error_size);
    if (!tensor) {
        snprintf(error, error_size, "vae weight %s: %s", name, error);
        return NULL;
    }
    size_t count = h3_gpu_tensor_elements(tensor);
    h3_gpu_tensor *converted = h3_gpu_tensor_new_f32(gpu, count);
    if (!converted ||
        !h3_gpu_cast_bf16_to_f32(gpu, converted, tensor,
                                 (uint32_t)count)) {
        snprintf(error, error_size, "vae weight %s: bf16 cast failed", name);
        h3_gpu_tensor_free(tensor);
        h3_gpu_tensor_free(converted);
        return NULL;
    }
    h3_gpu_tensor_free(tensor);
    return converted;
}

static int load_resnet(qw_vae *vae, qw_vae_resnet *r, h3_weight_store *store,
                       const char *prefix, uint32_t in_dim, uint32_t out_dim,
                       char *error, size_t error_size) {
    h3_gpu *gpu = vae->gpu;
    char name[256];
    // CausalConv3d subclasses nn.Conv2d, so weights are 4D [out, in, kh, kw]
    // (no temporal axis); RMS gammas are (dim, 1, 1, 1).
    uint64_t shape4[4];
    snprintf(name, sizeof(name), "%s.norm1.gamma", prefix);
    shape4[0] = in_dim; shape4[1] = 1; shape4[2] = 1; shape4[3] = 1;
    r->norm1 = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name), "%s.conv1.weight", prefix);
    shape4[0] = out_dim; shape4[1] = in_dim; shape4[2] = 3; shape4[3] = 3;
    r->conv1_w = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name), "%s.conv1.bias", prefix);
    shape4[0] = out_dim;
    r->conv1_b = load_w(store, gpu, name, 1, shape4, error, error_size);
    snprintf(name, sizeof(name), "%s.norm2.gamma", prefix);
    shape4[0] = out_dim; shape4[1] = 1; shape4[2] = 1; shape4[3] = 1;
    r->norm2 = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name), "%s.conv2.weight", prefix);
    shape4[0] = out_dim; shape4[1] = out_dim; shape4[2] = 3; shape4[3] = 3;
    r->conv2_w = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name), "%s.conv2.bias", prefix);
    shape4[0] = out_dim;
    r->conv2_b = load_w(store, gpu, name, 1, shape4, error, error_size);
    if (in_dim != out_dim) {
        snprintf(name, sizeof(name), "%s.conv_shortcut.weight", prefix);
        shape4[0] = out_dim; shape4[1] = in_dim; shape4[2] = 1; shape4[3] = 1;
        r->shortcut_w = load_w(store, gpu, name, 4, shape4, error,
                               error_size);
        snprintf(name, sizeof(name), "%s.conv_shortcut.bias", prefix);
        shape4[0] = out_dim;
        r->shortcut_b = load_w(store, gpu, name, 1, shape4, error,
                               error_size);
    }
    return r->norm1 && r->conv1_w && r->conv1_b && r->norm2 && r->conv2_w &&
           r->conv2_b && (in_dim == out_dim || (r->shortcut_w &&
                                                r->shortcut_b));
}

static void free_resnet(qw_vae_resnet *r) {
    h3_gpu_tensor_free(r->norm1);
    h3_gpu_tensor_free(r->conv1_w);
    h3_gpu_tensor_free(r->conv1_w9);
    h3_gpu_tensor_free(r->conv1_b);
    h3_gpu_tensor_free(r->norm2);
    h3_gpu_tensor_free(r->conv2_w);
    h3_gpu_tensor_free(r->conv2_w9);
    h3_gpu_tensor_free(r->conv2_b);
    h3_gpu_tensor_free(r->shortcut_w);
    h3_gpu_tensor_free(r->shortcut_b);
    memset(r, 0, sizeof(*r));
}

/* Repack a [CO, CI, 9] conv weight into the GEMM layout consumed by the
 * matmul2d convolution. */
static int vae_repack_conv(qw_vae *vae, h3_gpu_tensor *w, uint32_t co,
                           uint32_t ci, h3_gpu_tensor **w9) {
    uint32_t co_pad = (co + 63u) & ~63u;
    h3_gpu_tensor *t = h3_gpu_tensor_new_f32(vae->gpu,
                                             (size_t)9 * co_pad * ci);
    if (!t) return 0;
    if (!qw_gpu_vae_conv_repack_f32(vae->gpu, t, w, ci, co, co_pad)) {
        h3_gpu_tensor_free(t);
        return 0;
    }
    *w9 = t;
    return 1;
}

qw_vae *qw_vae_create(h3_gpu *gpu, const char *model_dir,
                      char *error, size_t error_size) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/vae", model_dir);
    h3_weight_store *store = h3_weight_store_open(path, error, error_size);
    if (!store) return NULL;

    qw_vae *vae = calloc(1, sizeof(qw_vae));
    if (!vae) {
        snprintf(error, error_size, "out of memory");
        h3_weight_store_free(store);
        return NULL;
    }
    vae->gpu = gpu;

    // Latent-space normalization statistics (vae/config.json), applied
    // by callers before decode.
    snprintf(path, sizeof(path), "%s/vae/config.json", model_dir);
    FILE *config_file = fopen(path, "rb");
    if (config_file) {
        fseek(config_file, 0, SEEK_END);
        long size = ftell(config_file);
        fseek(config_file, 0, SEEK_SET);
        char *text = size > 0 ? malloc((size_t)size + 1) : NULL;
        if (text && fread(text, 1, (size_t)size, config_file) == (size_t)size) {
            text[size] = 0;
            qw_json *config = NULL;
            if (qw_json_parse(text, &config, error, error_size)) {
                const qw_json *mean = qw_json_get(config, "latents_mean");
                const qw_json *std = qw_json_get(config, "latents_std");
                for (int c = 0; c < QWEN_VAE_Z; c++) {
                    vae->latents_mean[c] =
                        (float)qw_json_number(qw_json_at(mean, (size_t)c));
                    vae->latents_std[c] =
                        (float)qw_json_number(qw_json_at(std, (size_t)c));
                }
                qw_json_free(config);
            } else {
                error[0] = 0;  // statistics optional; decode guards
            }
            free(text);
        } else {
            free(text);
        }
        fclose(config_file);
    }

    // Channel plan (decoder_base_dim 144, dim_mult [1,2,4,8,8]):
    // dims = [1152, 1152, 1152, 576, 288, 144]; up blocks pair them,
    // stage 0-2 temporal (image path skips time_conv), stage 3 spatial
    // only, stage 4 no upsampling. DupUp factor_t follows the same list.
    static const uint32_t up_in[QWEN_VAE_UP_BLOCKS] = {
        1152, 1152, 1152, 576, 288};
    static const uint32_t up_out[QWEN_VAE_UP_BLOCKS] = {
        1152, 1152, 576, 288, 144};
    static const uint32_t up_ft[QWEN_VAE_UP_BLOCKS] = {2, 2, 2, 1, 1};

    char name[256];
    uint64_t shape4[4];
    int ok = 1;

    snprintf(name, sizeof(name), "post_quant_conv.weight");
    shape4[0] = QWEN_VAE_Z; shape4[1] = QWEN_VAE_Z; shape4[2] = 1;
    shape4[3] = 1;
    vae->post_quant_w = load_w(store, gpu, name, 4, shape4, error,
                               error_size);
    snprintf(name, sizeof(name), "post_quant_conv.bias");
    shape4[0] = QWEN_VAE_Z;
    vae->post_quant_b = load_w(store, gpu, name, 1, shape4, error,
                               error_size);

    snprintf(name, sizeof(name), "decoder.conv_in.weight");
    shape4[0] = 1152; shape4[1] = QWEN_VAE_Z; shape4[2] = 3; shape4[3] = 3;
    vae->conv_in_w = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name), "decoder.conv_in.bias");
    shape4[0] = 1152;
    vae->conv_in_b = load_w(store, gpu, name, 1, shape4, error, error_size);

    // Mid block: resnet -> attention -> resnet.
    ok = ok && load_resnet(vae, &vae->mid_resnets[0], store,
                           "decoder.mid_block.resnets.0", 1152, 1152, error,
                           error_size);
    ok = ok && load_resnet(vae, &vae->mid_resnets[1], store,
                           "decoder.mid_block.resnets.1", 1152, 1152, error,
                           error_size);
    snprintf(name, sizeof(name), "decoder.mid_block.attentions.0.norm.gamma");
    shape4[0] = 1152; shape4[1] = 1; shape4[2] = 1;
    vae->attn_norm = load_w(store, gpu, name, 3, shape4, error, error_size);
    snprintf(name, sizeof(name),
             "decoder.mid_block.attentions.0.to_qkv.weight");
    shape4[0] = 3456; shape4[1] = 1152; shape4[2] = 1; shape4[3] = 1;
    vae->attn_qkv_w = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name),
             "decoder.mid_block.attentions.0.to_qkv.bias");
    shape4[0] = 3456;
    vae->attn_qkv_b = load_w(store, gpu, name, 1, shape4, error, error_size);
    snprintf(name, sizeof(name),
             "decoder.mid_block.attentions.0.proj.weight");
    shape4[0] = 1152; shape4[1] = 1152; shape4[2] = 1; shape4[3] = 1;
    vae->attn_proj_w = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name), "decoder.mid_block.attentions.0.proj.bias");
    shape4[0] = 1152;
    vae->attn_proj_b = load_w(store, gpu, name, 1, shape4, error, error_size);

    for (int i = 0; ok && i < QWEN_VAE_UP_BLOCKS; i++) {
        qw_vae_up_block *block = &vae->up_blocks[i];
        block->in_dim = up_in[i];
        block->out_dim = up_out[i];
        block->factor_t = up_ft[i];
        block->factor_s = 2;
        block->up_flag = i < 4;
        for (int r = 0; ok && r < QWEN_VAE_RESNETS; r++) {
            uint32_t in = r == 0 ? block->in_dim : block->out_dim;
            snprintf(name, sizeof(name), "decoder.up_blocks.%d.resnets.%d",
                     i, r);
            ok = load_resnet(vae, &block->resnets[r], store, name, in,
                             block->out_dim, error, error_size);
        }
        if (ok && block->up_flag) {
            // The ResidualUpBlock upsampler runs after the resnets, so it
            // maps out_dim -> out_dim.
            snprintf(name, sizeof(name),
                     "decoder.up_blocks.%d.upsampler.resample.1.weight", i);
            shape4[0] = block->out_dim; shape4[1] = block->out_dim;
            shape4[2] = 3; shape4[3] = 3;
            block->resample_w = load_w(store, gpu, name, 4, shape4, error,
                                       error_size);
            snprintf(name, sizeof(name),
                     "decoder.up_blocks.%d.upsampler.resample.1.bias", i);
            shape4[0] = block->out_dim;
            block->resample_b = load_w(store, gpu, name, 1, shape4, error,
                                       error_size);
            ok = block->resample_w && block->resample_b;
        }
    }

    snprintf(name, sizeof(name), "decoder.norm_out.gamma");
    shape4[0] = 144; shape4[1] = 1; shape4[2] = 1; shape4[3] = 1;
    vae->norm_out = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name), "decoder.conv_out.weight");
    shape4[0] = QWEN_VAE_OUT; shape4[1] = 144; shape4[2] = 3; shape4[3] = 3;
    vae->conv_out_w = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name), "decoder.conv_out.bias");
    shape4[0] = QWEN_VAE_OUT;
    vae->conv_out_b = load_w(store, gpu, name, 1, shape4, error, error_size);
    // ---- encoder (ti2i) ----
    static const uint32_t down_in[5] = {96, 96, 192, 384, 768};
    static const uint32_t down_out[5] = {96, 192, 384, 768, 768};
    snprintf(name, sizeof(name), "encoder.conv_in.weight");
    shape4[0] = 96; shape4[1] = 4; shape4[2] = 3; shape4[3] = 3;
    vae->enc_conv_in_w = load_w(store, gpu, name, 4, shape4, error,
                                error_size);
    snprintf(name, sizeof(name), "encoder.conv_in.bias");
    shape4[0] = 96;
    vae->enc_conv_in_b = load_w(store, gpu, name, 1, shape4, error,
                                error_size);
    for (int i = 0; ok && i < 5; i++) {
        vae->down_blocks[i].in_dim = down_in[i];
        vae->down_blocks[i].out_dim = down_out[i];
        vae->down_blocks[i].down_flag = i < 4;
        vae->down_blocks[i].group = down_in[i] * 4 / down_out[i];
        vae->down_blocks[i].avg_ft = i == 0 ? 1u : 2u;
        for (int r = 0; ok && r < 2; r++) {
            uint32_t in = r == 0 ? down_in[i] : down_out[i];
            snprintf(name, sizeof(name), "encoder.down_blocks.%d.resnets.%d",
                     i, r);
            ok = load_resnet(vae, &vae->down_blocks[i].resnets[r], store,
                             name, in, down_out[i], error, error_size);
        }
        if (ok && i < 4) {
            snprintf(name, sizeof(name),
                     "encoder.down_blocks.%d.downsampler.resample.1.weight",
                     i);
            shape4[0] = down_out[i]; shape4[1] = down_out[i];
            shape4[2] = 3; shape4[3] = 3;
            vae->down_blocks[i].down_w = load_w(store, gpu, name, 4, shape4,
                                                error, error_size);
            snprintf(name, sizeof(name),
                     "encoder.down_blocks.%d.downsampler.resample.1.bias", i);
            shape4[0] = down_out[i];
            vae->down_blocks[i].down_b = load_w(store, gpu, name, 1, shape4,
                                                error, error_size);
            ok = ok && vae->down_blocks[i].down_w &&
                 vae->down_blocks[i].down_b;
        }
    }
    ok = ok && load_resnet(vae, &vae->enc_mid_resnets[0], store,
                           "encoder.mid_block.resnets.0", 768, 768, error,
                           error_size);
    ok = ok && load_resnet(vae, &vae->enc_mid_resnets[1], store,
                           "encoder.mid_block.resnets.1", 768, 768, error,
                           error_size);
    snprintf(name, sizeof(name), "encoder.mid_block.attentions.0.norm.gamma");
    shape4[0] = 768; shape4[1] = 1; shape4[2] = 1;
    vae->enc_attn_norm = load_w(store, gpu, name, 3, shape4, error,
                                error_size);
    snprintf(name, sizeof(name),
             "encoder.mid_block.attentions.0.to_qkv.weight");
    shape4[0] = 2304; shape4[1] = 768; shape4[2] = 1; shape4[3] = 1;
    vae->enc_attn_qkv_w = load_w(store, gpu, name, 4, shape4, error,
                                 error_size);
    snprintf(name, sizeof(name),
             "encoder.mid_block.attentions.0.to_qkv.bias");
    shape4[0] = 2304;
    vae->enc_attn_qkv_b = load_w(store, gpu, name, 1, shape4, error,
                                 error_size);
    snprintf(name, sizeof(name),
             "encoder.mid_block.attentions.0.proj.weight");
    shape4[0] = 768; shape4[1] = 768; shape4[2] = 1; shape4[3] = 1;
    vae->enc_attn_proj_w = load_w(store, gpu, name, 4, shape4, error,
                                  error_size);
    snprintf(name, sizeof(name), "encoder.mid_block.attentions.0.proj.bias");
    shape4[0] = 768;
    vae->enc_attn_proj_b = load_w(store, gpu, name, 1, shape4, error,
                                  error_size);
    snprintf(name, sizeof(name), "encoder.norm_out.gamma");
    shape4[0] = 768; shape4[1] = 1; shape4[2] = 1; shape4[3] = 1;
    vae->enc_norm_out = load_w(store, gpu, name, 4, shape4, error,
                               error_size);
    snprintf(name, sizeof(name), "encoder.conv_out.weight");
    shape4[0] = 128; shape4[1] = 768; shape4[2] = 3; shape4[3] = 3;
    vae->enc_conv_out_w = load_w(store, gpu, name, 4, shape4, error,
                                 error_size);
    snprintf(name, sizeof(name), "encoder.conv_out.bias");
    shape4[0] = 128;
    vae->enc_conv_out_b = load_w(store, gpu, name, 1, shape4, error,
                                 error_size);
    snprintf(name, sizeof(name), "quant_conv.weight");
    shape4[0] = 128; shape4[1] = 128; shape4[2] = 1; shape4[3] = 1;
    vae->quant_w = load_w(store, gpu, name, 4, shape4, error, error_size);
    snprintf(name, sizeof(name), "quant_conv.bias");
    shape4[0] = 128;
    vae->quant_b = load_w(store, gpu, name, 1, shape4, error, error_size);
    ok = ok && vae->enc_conv_in_w && vae->enc_conv_in_b && vae->enc_attn_norm &&
         vae->enc_attn_qkv_w && vae->enc_attn_qkv_b && vae->enc_attn_proj_w &&
         vae->enc_attn_proj_b && vae->enc_norm_out && vae->enc_conv_out_w &&
         vae->enc_conv_out_b && vae->quant_w && vae->quant_b;
    h3_weight_store_free(store);
    if (!ok || !vae->post_quant_w || !vae->post_quant_b || !vae->conv_in_w ||
        !vae->conv_in_b || !vae->attn_norm || !vae->attn_qkv_w ||
        !vae->attn_qkv_b || !vae->attn_proj_w || !vae->attn_proj_b ||
        !vae->norm_out || !vae->conv_out_w || !vae->conv_out_b) {
        qw_vae_free(vae);
        return NULL;
    }

    // One-time repack of every 3x3 conv weight into the GEMM layout.
    // The caller may hold an open command buffer: reconcile and restore
    // that state around our own submit.
    int was_open = !h3_gpu_begin(gpu);
    if (was_open &&
        (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu))) {
        snprintf(error, error_size, "vae repack: %s", h3_gpu_error(gpu));
        qw_vae_free(vae);
        return NULL;
    }
    ok = ok && vae_repack_conv(vae, vae->conv_in_w, 1152, QWEN_VAE_Z,
                               &vae->conv_in_w9);
    for (int i = 0; ok && i < 2; i++) {
        ok = vae_repack_conv(vae, vae->mid_resnets[i].conv1_w, 1152, 1152,
                             &vae->mid_resnets[i].conv1_w9);
        ok = ok && vae_repack_conv(vae, vae->mid_resnets[i].conv2_w, 1152,
                                   1152, &vae->mid_resnets[i].conv2_w9);
    }
    for (int i = 0; ok && i < QWEN_VAE_UP_BLOCKS; i++) {
        qw_vae_up_block *block = &vae->up_blocks[i];
        for (int r = 0; ok && r < QWEN_VAE_RESNETS; r++) {
            qw_vae_resnet *res = &block->resnets[r];
            uint32_t in = r == 0 ? block->in_dim : block->out_dim;
            ok = vae_repack_conv(vae, res->conv1_w, block->out_dim, in,
                                 &res->conv1_w9);
            ok = ok && vae_repack_conv(vae, res->conv2_w, block->out_dim,
                                       block->out_dim, &res->conv2_w9);
        }
        if (ok && block->up_flag)
            ok = vae_repack_conv(vae, block->resample_w, block->out_dim,
                                 block->out_dim, &block->resample_w9);
    }
    ok = ok && vae_repack_conv(vae, vae->conv_out_w, QWEN_VAE_OUT, 144,
                               &vae->conv_out_w9);
    ok = ok && vae_repack_conv(vae, vae->enc_conv_in_w, 96, 4,
                               &vae->enc_conv_in_w9);
    for (int i = 0; ok && i < 5; i++) {
        for (int r = 0; ok && r < 2; r++) {
            qw_vae_resnet *res = &vae->down_blocks[i].resnets[r];
            uint32_t in = r == 0 ? vae->down_blocks[i].in_dim
                                 : vae->down_blocks[i].out_dim;
            ok = vae_repack_conv(vae, res->conv1_w,
                                 vae->down_blocks[i].out_dim, in,
                                 &res->conv1_w9);
            ok = ok && vae_repack_conv(vae, res->conv2_w,
                                       vae->down_blocks[i].out_dim,
                                       vae->down_blocks[i].out_dim,
                                       &res->conv2_w9);
        }
    }
    for (int i = 0; ok && i < 2; i++) {
        ok = vae_repack_conv(vae, vae->enc_mid_resnets[i].conv1_w, 768, 768,
                             &vae->enc_mid_resnets[i].conv1_w9);
        ok = ok && vae_repack_conv(vae, vae->enc_mid_resnets[i].conv2_w, 768,
                                   768, &vae->enc_mid_resnets[i].conv2_w9);
    }
    ok = ok && vae_repack_conv(vae, vae->enc_conv_out_w, 128, 768,
                               &vae->enc_conv_out_w9);
    if (ok) ok = h3_gpu_submit(gpu);
    if (ok && was_open) ok = h3_gpu_begin(gpu);
    if (!ok) {
        snprintf(error, error_size, "vae weight repack failed: %s",
                 h3_gpu_error(gpu));
        qw_vae_free(vae);
        return NULL;
    }
    return vae;
}

void qw_vae_free(qw_vae *vae) {
    if (!vae) return;
    h3_gpu_tensor_free(vae->post_quant_w);
    h3_gpu_tensor_free(vae->post_quant_b);
    h3_gpu_tensor_free(vae->conv_in_w);
    h3_gpu_tensor_free(vae->conv_in_w9);
    h3_gpu_tensor_free(vae->conv_in_b);
    free_resnet(&vae->mid_resnets[0]);
    free_resnet(&vae->mid_resnets[1]);
    h3_gpu_tensor_free(vae->attn_norm);
    h3_gpu_tensor_free(vae->attn_qkv_w);
    h3_gpu_tensor_free(vae->attn_qkv_b);
    h3_gpu_tensor_free(vae->attn_proj_w);
    h3_gpu_tensor_free(vae->attn_proj_b);
    for (int i = 0; i < QWEN_VAE_UP_BLOCKS; i++) {
        for (int r = 0; r < QWEN_VAE_RESNETS; r++)
            free_resnet(&vae->up_blocks[i].resnets[r]);
        h3_gpu_tensor_free(vae->up_blocks[i].resample_w);
        h3_gpu_tensor_free(vae->up_blocks[i].resample_w9);
        h3_gpu_tensor_free(vae->up_blocks[i].resample_b);
    }
    h3_gpu_tensor_free(vae->norm_out);
    h3_gpu_tensor_free(vae->conv_out_w);
    h3_gpu_tensor_free(vae->conv_out_b);
    h3_gpu_tensor_free(vae->enc_conv_in_w);
    h3_gpu_tensor_free(vae->enc_conv_in_w9);
    h3_gpu_tensor_free(vae->enc_conv_in_b);
    for (int i = 0; i < 5; i++) {
        for (int r = 0; r < 2; r++) free_resnet(&vae->down_blocks[i].resnets[r]);
        h3_gpu_tensor_free(vae->down_blocks[i].down_w);
        h3_gpu_tensor_free(vae->down_blocks[i].down_b);
    }
    free_resnet(&vae->enc_mid_resnets[0]);
    free_resnet(&vae->enc_mid_resnets[1]);
    h3_gpu_tensor_free(vae->enc_attn_norm);
    h3_gpu_tensor_free(vae->enc_attn_qkv_w);
    h3_gpu_tensor_free(vae->enc_attn_qkv_b);
    h3_gpu_tensor_free(vae->enc_attn_proj_w);
    h3_gpu_tensor_free(vae->enc_attn_proj_b);
    h3_gpu_tensor_free(vae->enc_norm_out);
    h3_gpu_tensor_free(vae->enc_conv_out_w);
    h3_gpu_tensor_free(vae->enc_conv_out_w9);
    h3_gpu_tensor_free(vae->enc_conv_out_b);
    h3_gpu_tensor_free(vae->quant_w);
    h3_gpu_tensor_free(vae->quant_b);
    free(vae);
}

// --------------------------------------------------------------- forward

/* RMSNorm over channels, out of place: returns a fresh tensor and frees
 * the input (safe — encoded commands retain their resources). */
static int vae_norm(h3_gpu *gpu, h3_gpu_tensor **px, h3_gpu_tensor *gamma,
                    uint32_t pixels, uint32_t dim) {
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, (size_t)pixels * dim);
    if (!out) return 0;
    if (!qw_gpu_vae_rms_norm_f32(gpu, out, *px, gamma, pixels, dim,
                                 sqrtf((float)dim), 1e-12f)) {
        h3_gpu_tensor_free(out);
        return 0;
    }
    h3_gpu_tensor_free(*px);
    *px = out;
    return 1;
}

/* 3x3 spatial convolution (temporal axis folded away by the released
 * image path). Input and output are channels-last [H, W, C]. Default
 * path: 9-tap accumulating GEMM on the matmul2d tensor pipes over a
 * ring-padded copy of the activation (~40x the direct kernel). The
 * direct fp32 kernel remains as a fallback (H3_VAE_SCALAR_CONV=1);
 * the MPSGraph conv3d is never used: the MPSCommand buffer wrapper
 * owns its commit, which double-commits under per-stage submit/begin
 * cycling. */
static int vae_conv3x3(h3_gpu *gpu, h3_gpu_tensor **px, uint32_t height,
                       uint32_t width, uint32_t in_dim, uint32_t out_dim,
                       h3_gpu_tensor *weight, h3_gpu_tensor *weight9,
                       h3_gpu_tensor *bias, h3_gpu_tensor *pad_arena) {
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(
        gpu, (size_t)height * width * out_dim);
    if (!out) return 0;
    int ok;
    if (weight9 && !getenv("H3_VAE_SCALAR_CONV")) {
        ok = qw_gpu_vae_conv_mm_f32(gpu, out, *px, weight9, bias, height,
                                    width, in_dim, out_dim, pad_arena);
    } else {
        ok = qw_gpu_vae_conv3x3_f32(gpu, out, *px, weight, bias, height,
                                    width, in_dim, out_dim);
    }
    if (ok) {
        /* Commit right away: the decode encodes dozens of convs into
         * one command stream, and Metal retains every resource an
         * uncommitted command references — including the ring-pad
         * scratch (up to ~2.4 GB per conv at 2752x1536) and each
         * freed activation. Per-conv submits bound that retention to
         * a single stage (peak ~100+ GB otherwise). */
        ok = h3_gpu_submit(gpu) && h3_gpu_begin(gpu);
    }
    if (ok) {
        h3_gpu_tensor_free(*px);
        *px = out;
    } else {
        h3_gpu_tensor_free(out);
    }
    return ok;
}

static int vae_resnet(qw_vae *vae, qw_vae_resnet *r, h3_gpu_tensor **px,
                      uint32_t *height, uint32_t *width, uint32_t in_dim,
                      uint32_t out_dim, h3_gpu_tensor *pad_arena,
                      const char *dump_dir, const char *dump_tag) {
    h3_gpu *gpu = vae->gpu;
    h3_gpu_tensor *x = *px; // residual base; kept until the final add
    uint32_t pixels = *height * *width;
    char name[128];
    h3_gpu_tensor *h_owned = NULL;
    h3_gpu_tensor *h;
    if (r->shortcut_w) {
        h_owned = h3_gpu_tensor_new_f32(gpu, (size_t)pixels * out_dim);
        if (!h_owned) return 0;
        if (!h3_gpu_linear_f32(gpu, h_owned, x, r->shortcut_w, r->shortcut_b,
                               pixels, in_dim, out_dim)) {
            h3_gpu_tensor_free(h_owned);
            return 0;
        }
        h = h_owned;
    } else {
        h = x; // Identity shortcut borrows the input buffer.
    }
    snprintf(name, sizeof(name), "%s_shortcut", dump_tag);
    int ok = dump_tensor(gpu, dump_dir, name, h, (size_t)pixels * out_dim);

    h3_gpu_tensor *cur = h3_gpu_tensor_new_f32(gpu, (size_t)pixels * in_dim);
    ok = ok && cur;
    if (ok) {
        ok = qw_gpu_vae_rms_norm_f32(gpu, cur, x, r->norm1, pixels, in_dim,
                                     sqrtf((float)in_dim), 1e-12f);
    }
    snprintf(name, sizeof(name), "%s_norm1", dump_tag);
    ok = ok && dump_tensor(gpu, dump_dir, name, cur,
                           (size_t)pixels * in_dim);
    ok = ok && h3_gpu_silu_f32(gpu, cur, cur, pixels * in_dim);
    ok = ok && vae_conv3x3(gpu, &cur, *height, *width, in_dim, out_dim,
                           r->conv1_w, r->conv1_w9, r->conv1_b, pad_arena);
    snprintf(name, sizeof(name), "%s_conv1", dump_tag);
    ok = ok && dump_tensor(gpu, dump_dir, name, cur,
                           (size_t)pixels * out_dim);
    if (ok) {
        ok = qw_gpu_vae_rms_norm_f32(gpu, cur, cur, r->norm2, pixels,
                                     out_dim, sqrtf((float)out_dim), 1e-12f);
    }
    snprintf(name, sizeof(name), "%s_norm2", dump_tag);
    ok = ok && dump_tensor(gpu, dump_dir, name, cur,
                           (size_t)pixels * out_dim);
    ok = ok && h3_gpu_silu_f32(gpu, cur, cur, pixels * out_dim);
    ok = ok && vae_conv3x3(gpu, &cur, *height, *width, out_dim, out_dim,
                           r->conv2_w, r->conv2_w9, r->conv2_b, pad_arena);
    snprintf(name, sizeof(name), "%s_conv2", dump_tag);
    ok = ok && dump_tensor(gpu, dump_dir, name, cur,
                           (size_t)pixels * out_dim);
    ok = ok && qw_gpu_vae_add_f32(gpu, cur, cur, h, pixels * out_dim);
    if (ok) ok = h3_gpu_submit(gpu) && h3_gpu_begin(gpu);
    h3_gpu_tensor_free(h_owned);
    if (!ok) {
        /* Ownership follows vae_conv3x3/vae_norm: on failure *px stays
         * untouched (the caller frees it exactly once at its own exit)
         * and only this function's private buffers die here. */
        h3_gpu_tensor_free(cur);
        return 0;
    }
    h3_gpu_tensor_free(x);
    *px = cur;
    return 1;
}

/* Mid-block single-head spatial attention over the folded frame.
 * Input stays alive as the residual; work happens on fresh buffers. */
static int vae_attention(qw_vae *vae, h3_gpu_tensor **px, uint32_t *height,
                         uint32_t *width, uint32_t dim,
                         const h3_gpu_tensor *norm_g,
                         const h3_gpu_tensor *qkv_w, const h3_gpu_tensor *qkv_b,
                         const h3_gpu_tensor *proj_w, const h3_gpu_tensor *proj_b,
                         const char *dump_dir, const char *dump_tag) {
    h3_gpu *gpu = vae->gpu;
    h3_gpu_tensor *x = *px;
    uint32_t pixels = *height * *width;
    char name[128];

    h3_gpu_tensor *cur = h3_gpu_tensor_new_f32(gpu, (size_t)pixels * dim);
    h3_gpu_tensor *q = h3_gpu_tensor_new_f32(gpu, (size_t)pixels * dim);
    h3_gpu_tensor *k = h3_gpu_tensor_new_f32(gpu, (size_t)pixels * dim);
    // v is written transposed [C, pixels] by the split (part 2).
    h3_gpu_tensor *v = h3_gpu_tensor_new_f32(gpu, (size_t)pixels * dim);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_f32(gpu,
                                               (size_t)pixels * dim * 3);
    h3_gpu_tensor *scores = h3_gpu_tensor_new_f32(gpu,
                                                  (size_t)pixels * pixels);
    h3_gpu_tensor *attended = h3_gpu_tensor_new_f32(gpu,
                                                    (size_t)pixels * dim);
    int ok = cur && q && k && v && qkv && scores && attended;
    ok = ok && qw_gpu_vae_rms_norm_f32(gpu, cur, x, norm_g, pixels,
                                       dim, sqrtf((float)dim), 1e-12f);
    snprintf(name, sizeof(name), "%s_norm", dump_tag);
    ok = ok && dump_tensor(gpu, dump_dir, name, cur, (size_t)pixels * dim);
    ok = ok && h3_gpu_linear_f32(gpu, qkv, cur, qkv_w, qkv_b, pixels, dim,
                                 dim * 3);
    snprintf(name, sizeof(name), "%s_qkv", dump_tag);
    ok = ok && dump_tensor(gpu, dump_dir, name, qkv,
                           (size_t)pixels * dim * 3);
    ok = ok && qw_gpu_vae_split3_f32(gpu, q, qkv, pixels, dim, 0);
    ok = ok && qw_gpu_vae_split3_f32(gpu, k, qkv, pixels, dim, 1);
    ok = ok && qw_gpu_vae_split3_f32(gpu, v, qkv, pixels, dim, 2);

    // scores[p, o] = sum_c q[p, c] * k[o, c] via the row-major GEMM, then
    // the SDPA softmax with scale 1/sqrt(head_dim).
    ok = ok && h3_gpu_linear_f32(gpu, scores, q, k, NULL, pixels, dim,
                                 pixels);
    /* Softmax must run per query row: the GPU row-softmax carries context
     * across rows (P5 trap), which only shows up once pixels > one block.
     * Read back and do it on the host. */
    {
        float *sh = malloc(sizeof(float) * (size_t)pixels * pixels);
        ok = ok && sh && h3_gpu_submit(gpu) && h3_gpu_begin(gpu);
        ok = ok && h3_gpu_tensor_read_f32(scores, sh, (size_t)pixels * pixels);
        if (ok) {
            float scale = (float)(1.0 / sqrt((double)dim));
            for (uint32_t p = 0; p < pixels; p++) {
                float *row = sh + (size_t)p * pixels;
                float m = -3.402823466e+38f;
                for (uint32_t o = 0; o < pixels; o++) {
                    row[o] *= scale;
                    if (row[o] > m) m = row[o];
                }
                float denom = 0.0f;
                for (uint32_t o = 0; o < pixels; o++) {
                    row[o] = expf(row[o] - m);
                    denom += row[o];
                }
                for (uint32_t o = 0; o < pixels; o++) row[o] /= denom;
            }
            ok = h3_gpu_tensor_write_f32(scores, sh,
                                         (size_t)pixels * pixels);
        }
        free(sh);
    }
    // attn[p, c] = sum_q probs[p, q] * v_t[c, q] — the split leaves v
    // transposed [C, pixels], exactly the GEMM weight layout [out, in].
    ok = ok && h3_gpu_linear_f32(gpu, attended, scores, v, NULL, pixels,
                                 pixels, dim);
    ok = ok && h3_gpu_linear_f32(gpu, cur, attended, proj_w, proj_b,
                                 pixels, dim, dim);
    snprintf(name, sizeof(name), "%s_proj", dump_tag);
    ok = ok && dump_tensor(gpu, dump_dir, name, cur, (size_t)pixels * dim);
    ok = ok && qw_gpu_vae_add_f32(gpu, cur, cur, x, pixels * dim);
    h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(v);
    h3_gpu_tensor_free(qkv);
    h3_gpu_tensor_free(scores);
    h3_gpu_tensor_free(attended);
    if (ok) ok = h3_gpu_submit(gpu) && h3_gpu_begin(gpu);
    if (!ok) {
        /* Same ownership contract as vae_resnet/vae_conv3x3: on failure
         * the caller keeps *px (its single free at the function exit
         * stays valid). */
        h3_gpu_tensor_free(cur);
        return 0;
    }
    h3_gpu_tensor_free(x);
    *px = cur;
    return 1;
}

/* Ring-pad scratch size for one conv (fp32 elements), as consumed by
 * qw_gpu_vae_conv_mm_f32. */
static size_t vae_pad_elems(uint32_t height, uint32_t width, uint32_t in_c) {
    uint32_t wp = width + 2u;
    uint32_t m_span =
        (uint32_t)(((size_t)(height - 1) * wp + width + 63) / 64 * 64);
    uint32_t pad_rows = m_span + 2u * wp + 2u;
    return (size_t)pad_rows * in_c;
}

int qw_vae_decode(qw_vae *vae, const h3_gpu_tensor *z,
                  uint32_t latent_height, uint32_t latent_width,
                  h3_gpu_tensor *output, char *error, size_t error_size) {
    h3_gpu *gpu = vae->gpu;
    const char *dump_dir = getenv("QWEN_VAE_DUMP");
    size_t z_count = h3_gpu_tensor_elements(z);
    if (!latent_height || !latent_width ||
        z_count != (size_t)QWEN_VAE_Z * latent_height * latent_width) {
        snprintf(error, error_size,
                 "vae latent size %zu != 64 * %u * %u", z_count,
                 latent_height, latent_width);
        return 0;
    }
    size_t pixels = z_count / QWEN_VAE_Z;
    uint32_t height = latent_height;
    uint32_t width = latent_width;
    int ok = 1;

    // NCHW [64, H, W] -> channels-last [H, W, 64].
    h3_gpu_tensor *x = h3_gpu_tensor_new_f32(gpu, z_count);
    ok = x != NULL;
    ok = ok && qw_gpu_vae_chw_to_hwc_f32(gpu, x, z, QWEN_VAE_Z, height,
                                         width);

    // post_quant 1x1: [64 -> 64].
    h3_gpu_tensor *next = h3_gpu_tensor_new_f32(gpu, z_count);
    ok = ok && next;
    ok = ok && h3_gpu_linear_f32(gpu, next, x, vae->post_quant_w,
                                 vae->post_quant_b, (uint32_t)pixels,
                                 QWEN_VAE_Z, QWEN_VAE_Z);
    h3_gpu_tensor_free(x);
    x = next;
    ok = ok && dump_tensor(gpu, dump_dir, "m900_post_quant_conv", x, z_count);

    // Pad-scratch arena: the largest conv pad over the whole chain,
    // allocated once and reused by every conv. Metal never returns
    // freed MTLBuffer pages to the OS, so per-conv scratch of distinct
    // sizes leaves permanently-mapped fragmentation behind (the pad
    // alone is ~4.9 GB at the largest size).
    size_t arena_elems = vae_pad_elems(height, width, QWEN_VAE_Z);
    {
        static const uint32_t walk_in[QWEN_VAE_UP_BLOCKS] = {1152, 1152,
                                                             1152, 576, 288};
        static const uint32_t walk_out[QWEN_VAE_UP_BLOCKS] = {1152, 1152,
                                                              576, 288, 144};
        uint32_t h = height, w = width;
        size_t e;
        for (int i = 0; i < QWEN_VAE_UP_BLOCKS; i++) {
            e = vae_pad_elems(h, w, walk_in[i]);
            if (e > arena_elems) arena_elems = e;
            e = vae_pad_elems(h, w, walk_out[i]);
            if (e > arena_elems) arena_elems = e;
            if (i < QWEN_VAE_UP_BLOCKS - 1) {
                e = vae_pad_elems(h * 2, w * 2, walk_out[i]);
                if (e > arena_elems) arena_elems = e;
                h *= 2;
                w *= 2;
            }
        }
        e = vae_pad_elems(h, w, 144);
        if (e > arena_elems) arena_elems = e;
    }
    h3_gpu_tensor *pad_arena = h3_gpu_tensor_new_f32(gpu, arena_elems);
    ok = ok && pad_arena != NULL;

    // conv_in: 3x3, 64 -> 1152.
    ok = ok && vae_conv3x3(gpu, &x, height, width, QWEN_VAE_Z, 1152,
                           vae->conv_in_w, vae->conv_in_w9, vae->conv_in_b,
                           pad_arena);
    ok = ok && dump_tensor(gpu, dump_dir, "m001_conv_in", x,
                           (size_t)height * width * 1152);

    // Mid block: resnet -> attention -> resnet.
    ok = ok && vae_resnet(vae, &vae->mid_resnets[0], &x, &height, &width,
                          1152, 1152, pad_arena, dump_dir, "mid_r0");
    ok = ok && vae_attention(vae, &x, &height, &width, 1152, vae->attn_norm,
                             vae->attn_qkv_w, vae->attn_qkv_b,
                             vae->attn_proj_w, vae->attn_proj_b, dump_dir,
                             "attn");
    ok = ok && vae_resnet(vae, &vae->mid_resnets[1], &x, &height, &width,
                          1152, 1152, pad_arena, dump_dir, "mid_r1");

    // Up blocks: 3 resnets, then (nearest 2x + conv) + DupUp shortcut.
    for (int i = 0; ok && i < QWEN_VAE_UP_BLOCKS; i++) {
        qw_vae_up_block *block = &vae->up_blocks[i];
        char tag[64];
        // The DupUp shortcut reads the block INPUT (pre-resnets), so keep
        // an owned copy while the resnets consume the original.
        h3_gpu_tensor *x_copy = NULL;
        if (block->up_flag) {
            x_copy = h3_gpu_tensor_new_f32(gpu, (size_t)height * width *
                                                      block->in_dim);
            ok = x_copy != NULL;
            ok = ok && h3_gpu_copy_f32(gpu, x_copy, 0, x, 0,
                                       (size_t)height * width *
                                           block->in_dim);
        }
        for (int r = 0; ok && r < QWEN_VAE_RESNETS; r++) {
            uint32_t in = r == 0 ? block->in_dim : block->out_dim;
            snprintf(tag, sizeof(tag), "up%d_r%d", i, r);
            ok = vae_resnet(vae, &block->resnets[r], &x, &height, &width, in,
                            block->out_dim, pad_arena, dump_dir, tag);
        }
        if (ok && block->up_flag) {
            uint32_t old_height = height;
            uint32_t old_width = width;
            uint32_t new_height = height * 2;
            uint32_t new_width = width * 2;
            h3_gpu_tensor *up = h3_gpu_tensor_new_f32(
                gpu, (size_t)new_height * new_width * block->out_dim);
            ok = up != NULL;
            ok = ok && qw_gpu_vae_nearest2x_f32(gpu, up, x, height, width,
                                                block->out_dim);
            snprintf(tag, sizeof(tag), "up%d_nearest", i);
            ok = ok && dump_tensor(gpu, dump_dir, tag, up,
                                   (size_t)new_height * new_width *
                                       block->out_dim);
            h3_gpu_tensor_free(x);
            x = up;
            ok = ok && vae_conv3x3(gpu, &x, new_height, new_width,
                                   block->out_dim, block->out_dim,
                                   block->resample_w, block->resample_w9,
                                   block->resample_b, pad_arena);
            snprintf(tag, sizeof(tag), "up%d_resample", i);
            ok = ok && dump_tensor(gpu, dump_dir, tag, x,
                                   (size_t)new_height * new_width *
                                       block->out_dim);

            h3_gpu_tensor *shortcut = h3_gpu_tensor_new_f32(
                gpu, (size_t)new_height * new_width * block->out_dim);
            ok = ok && shortcut;
            ok = ok && qw_gpu_vae_dupup_f32(gpu, shortcut, x_copy, old_height,
                                            old_width, block->in_dim,
                                            block->out_dim, block->factor_t,
                                            block->factor_s,
                                            block->factor_t - 1);
            snprintf(tag, sizeof(tag), "up%d_shortcut", i);
            ok = ok && dump_tensor(gpu, dump_dir, tag, shortcut,
                                   (size_t)new_height * new_width *
                                       block->out_dim);
            ok = ok && qw_gpu_vae_add_f32(gpu, x, x, shortcut,
                                          (uint32_t)((size_t)new_height *
                                                     new_width *
                                                     block->out_dim));
            h3_gpu_tensor_free(shortcut);
            height = new_height;
            width = new_width;
        }
        h3_gpu_tensor_free(x_copy);
        if (ok) ok = h3_gpu_submit(gpu) && h3_gpu_begin(gpu);
    }

    // Head: RMSNorm -> SiLU -> conv 3x3 (144 -> 4) -> clamp.
    ok = ok && vae_norm(gpu, &x, vae->norm_out, height * width, 144);
    ok = ok && dump_tensor(gpu, dump_dir, "m139_norm_out", x,
                           (size_t)height * width * 144);
    ok = ok && h3_gpu_silu_f32(gpu, x, x, height * width * 144);
    ok = ok && vae_conv3x3(gpu, &x, height, width, 144, QWEN_VAE_OUT,
                           vae->conv_out_w, vae->conv_out_w9,
                           vae->conv_out_b, pad_arena);
    ok = ok && dump_tensor(gpu, dump_dir, "m140_conv_out", x,
                           (size_t)height * width * QWEN_VAE_OUT);
    ok = ok && h3_gpu_clip_f32(gpu, x, x, height * width * QWEN_VAE_OUT,
                               -1.0f, 1.0f);

    // Channels-last [H, W, 4] -> NCHW [4, H, W].
    ok = ok && qw_gpu_vae_hwc_to_chw_f32(gpu, output, x, QWEN_VAE_OUT, height,
                                         width);
    h3_gpu_tensor_free(x);
    h3_gpu_tensor_free(pad_arena);
    if (!ok) {
        snprintf(error, error_size, "vae decode failed: %s",
                 h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

const float *qw_vae_latents_mean(const qw_vae *vae) {
    return vae && vae->latents_mean[0] != 0.0f ? vae->latents_mean : NULL;
}

// ------------------------------------------------------------- encoder

/* CPU: stride-2 3x3 conv over right-bottom-padded input (ZeroPad2d
 * (0, 1, 0, 1) + Conv2d stride 2), channels-last. */
__attribute__((unused)) static int cpu_downsample(const float *in, uint32_t h, uint32_t w, uint32_t c,
                          const float *w4d, const float *bias, uint32_t oc,
                          float *out) {
    uint32_t oh = h / 2, ow = w / 2;
    for (uint32_t y = 0; y < oh; y++) {
        for (uint32_t x = 0; x < ow; x++) {
            float *dst = out + ((size_t)y * ow + x) * oc;
            for (uint32_t o = 0; o < oc; o++) dst[o] = bias[o];
            for (uint32_t ci = 0; ci < c; ci++) {
                for (int ky = 0; ky < 3; ky++) {
                    uint32_t iy = y * 2 + (uint32_t)ky;
                    if (iy >= h) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        uint32_t ix = x * 2 + (uint32_t)kx;
                        if (ix >= w) continue;
                        float v = in[((size_t)iy * w + ix) * c + ci];
                        const float *kw = w4d +
                            ((size_t)ci * 3 + ky) * 3 + kx;
                        for (uint32_t o = 0; o < oc; o++)
                            dst[o] += kw[(size_t)o * c * 9] * v;
                    }
                }
            }
        }
    }
    return 1;
}

/* CPU: AvgDown3D single-frame. factor_t = 2 blocks left-pad a zero
 * frame, so half the output channels collapse to zero and the rest are
 * 2x2 spatial means; factor_t = 1 with factor_s = 2 is the plain 2x2
 * mean; group == 1 is the identity. */
__attribute__((unused)) static void cpu_avg_down(const float *in, uint32_t h, uint32_t w, uint32_t c,
                         uint32_t oc, uint32_t ft, float *out) {
    uint32_t factor = ft * 4;
    uint32_t group = c * factor / oc;
    if (group == 1 && factor == 1) {
        memcpy(out, in, sizeof(float) * (size_t)h * w * c);
        return;
    }
    uint32_t sh = ft == 2 ? h : h / 2;
    uint32_t sw = ft == 2 ? w : w / 2;
    (void)sh;
    (void)sw;
    uint32_t oh = ft == 2 ? h / 2 : h / 2;
    uint32_t ow = ft == 2 ? w / 2 : w / 2;
    for (uint32_t y = 0; y < oh; y++) {
        for (uint32_t x = 0; x < ow; x++) {
            float *dst = out + ((size_t)y * ow + x) * oc;
            for (uint32_t o = 0; o < oc; o++) {
                float acc = 0.0f;
                for (uint32_t g = 0; g < group; g++) {
                    uint32_t n = o * group + g;
                    uint32_t ci = n / factor;
                    uint32_t rem = n % factor;
                    uint32_t fti = rem / 4;
                    uint32_t hr = (rem % 4) / 2;
                    uint32_t wr = rem % 2;
                    float v = 0.0f;
                    if (!(ft == 2 && fti == 0))
                        v = in[(((size_t)y * 2 + hr) * w + (x * 2 + wr)) * c +
                               ci];
                    acc += v;
                }
                dst[o] = acc / (float)group;
            }
        }
    }
}

/* Encoder pass: input [H, W, 4] channels-last fp32 in [-1, 1]; output
 * packed [H/16 * W/16, 64] bf16 rows (mu, mean/std normalized),
 * row-major over [h][w]. */
static uint16_t vae_f32_to_bits(float value) {
    /* round-to-nearest-even, matching PyTorch's f32 -> bf16 cast */
    uint32_t bits;
    memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

int qw_vae_encode(qw_vae *vae, const h3_gpu_tensor *input, uint32_t height,
                  uint32_t width, h3_gpu_tensor *latents, char *error,
                  size_t error_size) {
    h3_gpu *gpu = vae->gpu;
    /* vae_pad_elems walks (height-1) below: a zero side would underflow
     * into a nonsense — possibly enormous — arena size before any kernel
     * guard could reject it. Decode validates the same way. */
    if (!height || !width) {
        snprintf(error, error_size, "vae encode: zero latent size");
        return 0;
    }
    /* The packing stage divides by latents_std per channel: with stats
     * missing (create tolerates a broken config.json and leaves zeros)
     * that silently produced ±inf/NaN latents that propagated into the
     * DiT. The decode side checks the same getter convention. */
    if (vae->latents_std[0] == 0.0f) {
        snprintf(error, error_size, "vae encode: latents stats missing");
        return 0;
    }
    /* 每块按「该块所有 conv 的最大输入通道」算 pad：down_i 的 conv2 输入是
     * conv1 的 2 倍（96→192→384→768），只按 conv1 算会让 conv2 的 pad 缓冲
     * 越界（大尺寸才会暴露：小尺寸靠分配器的页内余量侥幸不踩）。 */
    static const uint32_t dims[6] = {96, 192, 384, 768, 768, 768};
    uint32_t h = height, w = width;
    size_t pad_max = 0;
    for (int stage = 0; stage < 6; stage++) {
        size_t p = vae_pad_elems(h, w, dims[stage]);
        if (p > pad_max) pad_max = p;
        if (stage < 4) {
            h /= 2;
            w /= 2;
        }
    }
    h = height;
    w = width;
    h3_gpu_tensor *pad_arena = h3_gpu_tensor_new_f32(gpu, pad_max);
    h3_gpu_tensor *x = h3_gpu_tensor_new_f32(
        gpu, (size_t)height * width * 4);
    int ok = pad_arena && x;
    ok = ok && h3_gpu_copy_f32(gpu, x, 0, input, 0,
                               (size_t)height * width * 4);

    // conv_in
    ok = ok && vae_conv3x3(gpu, &x, h, w, 4, 96, vae->enc_conv_in_w,
                           vae->enc_conv_in_w9, vae->enc_conv_in_b,
                           pad_arena);
    const char *dump_dir = getenv("QWEN_VAE_DUMP");
    ok = ok && dump_tensor(gpu, dump_dir, "enc_conv_in", x,
                           (size_t)h * w * 96);

    // down blocks（全 GPU：avg_shortcut 用 GPU-GPU 拷贝留底，无 host 回读。
    // copy 排队后 resnets 会替换并 free x（竞技场复用），必须先 submit 让
    // copy 落地。）
    for (int i = 0; ok && i < 5; i++) {
        uint32_t ic = vae->down_blocks[i].in_dim;
        uint32_t oc = vae->down_blocks[i].out_dim;
        h3_gpu_tensor *block_in =
            h3_gpu_tensor_new_f32(gpu, (size_t)h * w * ic);
        ok = ok && block_in &&
             h3_gpu_copy_f32(gpu, block_in, 0, x, 0, (size_t)h * w * ic) &&
             h3_gpu_submit(gpu) && h3_gpu_begin(gpu);
        for (int r = 0; ok && r < 2; r++) {
            uint32_t in = r == 0 ? ic : oc;
            char rtag[64];
            snprintf(rtag, sizeof(rtag), "enc_d%d_resnets%d", i, r);
            ok = vae_resnet(vae, &vae->down_blocks[i].resnets[r], &x, &h, &w,
                            in, oc, pad_arena, dump_dir, rtag);
        }
        if (ok && vae->down_blocks[i].down_flag) {
            uint32_t nh = h / 2, nw = w / 2;
            h3_gpu_tensor *g_avg =
                h3_gpu_tensor_new_f32(gpu, (size_t)nh * nw * oc);
            h3_gpu_tensor *g_ds =
                h3_gpu_tensor_new_f32(gpu, (size_t)nh * nw * oc);
            ok = ok && g_avg && g_ds;
            ok = ok && qw_gpu_vae_avgdown_f32(gpu, g_avg, block_in, h, w, ic,
                                              oc, vae->down_blocks[i].avg_ft);
            char tag2[64];
            snprintf(tag2, sizeof(tag2), "enc_d%d_avg", i);
            ok = ok && dump_tensor(gpu, dump_dir, tag2, g_avg,
                                   (size_t)nh * nw * oc);
            ok = ok && qw_gpu_vae_downsample_f32(
                            gpu, g_ds, x, vae->down_blocks[i].down_w,
                            vae->down_blocks[i].down_b, h, w, oc, oc);
            snprintf(tag2, sizeof(tag2), "enc_d%d_ds", i);
            ok = ok && dump_tensor(gpu, dump_dir, tag2, g_ds,
                                   (size_t)nh * nw * oc);
            ok = ok && qw_gpu_vae_add_f32(gpu, g_ds, g_ds, g_avg,
                                          (uint32_t)nh * nw * oc);
            h3_gpu_tensor_free(g_avg);
            h3_gpu_tensor_free(x);
            x = g_ds;
            h = nh;
            w = nw;
        } else if (ok) {
            /* down_flag=0: AvgDown is the identity; residual add with the
             * block input (GPU-GPU). */
            size_t count = (size_t)h * w * oc;
            h3_gpu_tensor *summed = h3_gpu_tensor_new_f32(gpu, count);
            ok = ok && summed &&
                 qw_gpu_vae_add_f32(gpu, summed, x, block_in,
                                    (uint32_t)count);
            h3_gpu_tensor_free(x);
            x = summed;
        }
        h3_gpu_tensor_free(block_in);
    }

    // mid: resnet -> attention -> resnet
    ok = ok && vae_resnet(vae, &vae->enc_mid_resnets[0], &x, &h, &w, 768,
                          768, pad_arena, dump_dir, "enc_mid_resnets0");
    ok = ok && vae_attention(vae, &x, &h, &w, 768, vae->enc_attn_norm,
                             vae->enc_attn_qkv_w, vae->enc_attn_qkv_b,
                             vae->enc_attn_proj_w, vae->enc_attn_proj_b,
                             dump_dir, "enc_mid_attn");
    ok = ok && vae_resnet(vae, &vae->enc_mid_resnets[1], &x, &h, &w, 768,
                          768, pad_arena, dump_dir, "enc_mid_resnets1");
    ok = ok && dump_tensor(gpu, dump_dir, "enc_mid", x, (size_t)h * w * 768);

    // head: RMS -> SiLU -> conv3x3 (768 -> 128) -> quant 1x1
    ok = ok && vae_norm(gpu, &x, vae->enc_norm_out, h * w, 768);
    ok = ok && h3_gpu_silu_f32(gpu, x, x, (uint32_t)((size_t)h * w * 768));
    ok = ok && vae_conv3x3(gpu, &x, h, w, 768, 128, vae->enc_conv_out_w,
                           vae->enc_conv_out_w9, vae->enc_conv_out_b,
                           pad_arena);
    ok = ok && dump_tensor(gpu, dump_dir, "enc_conv_out_c", x,
                           (size_t)h * w * 128);
    h3_gpu_tensor *q = h3_gpu_tensor_new_f32(gpu, (size_t)h * w * 128);
    ok = ok && q && h3_gpu_linear_f32(gpu, q, x, vae->quant_w, vae->quant_b,
                                      (uint32_t)((size_t)h * w), 128, 128);
    h3_gpu_tensor_free(x);
    x = q;
    ok = ok && dump_tensor(gpu, dump_dir, "enc_quant", x,
                           (size_t)h * w * 128);
    if (ok) ok = h3_gpu_submit(gpu) && h3_gpu_begin(gpu);

    // Pack mu (first 64 channels), mean/std normalized, into rows.
    if (ok) {
        size_t count = (size_t)h * w * 128;
        float *host = malloc(sizeof(float) * count);
        uint16_t *packed =
            malloc(sizeof(uint16_t) * (size_t)h * w * QWEN_VAE_Z);
        ok = ok && host && packed &&
             h3_gpu_tensor_read_f32(x, host, count);
        if (ok) {
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t xx = 0; xx < w; xx++) {
                    const float *row = host + ((size_t)y * w + xx) * 128;
                    uint16_t *dst =
                        packed + ((size_t)y * w + xx) * QWEN_VAE_Z;
                    for (int c = 0; c < QWEN_VAE_Z; c++) {
                        float mu = (row[c] - vae->latents_mean[c]) /
                                   vae->latents_std[c];
                        dst[c] = vae_f32_to_bits(mu);
                    }
                }
            }
            ok = h3_gpu_tensor_write_bf16(latents, packed,
                                          (size_t)h * w * QWEN_VAE_Z);
        }
        free(host);
        free(packed);
    }
    h3_gpu_tensor_free(x);
    h3_gpu_tensor_free(pad_arena);
    if (!ok) {
        snprintf(error, error_size, "vae encode failed: %s",
                 h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

const float *qw_vae_latents_std(const qw_vae *vae) {
    return vae && vae->latents_std[0] != 0.0f ? vae->latents_std : NULL;
}
