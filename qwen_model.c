#include "qwen_model.h"

#include "h3_safetensors.h"
#include "qwen_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int dir_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static char *read_file_text(const char *path, char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        if (error && error_size) snprintf(error, error_size, "cannot open %s", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        if (error && error_size) snprintf(error, error_size, "cannot seek %s", path);
        return NULL;
    }
    long size = ftell(file);
    if (size < 0 || (unsigned long)size > (64u << 20)) {
        fclose(file);
        if (error && error_size) snprintf(error, error_size, "bad size %s", path);
        return NULL;
    }
    rewind(file);
    char *text = malloc((size_t)size + 1);
    if (!text) {
        fclose(file);
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return NULL;
    }
    size_t read = fread(text, 1, (size_t)size, file);
    fclose(file);
    if (read != (size_t)size) {
        free(text);
        if (error && error_size) snprintf(error, error_size, "short read %s", path);
        return NULL;
    }
    text[size] = '\0';
    return text;
}

static int parse_config_file(const char *path, qw_json **out, char *error,
                             size_t error_size) {
    char *text = read_file_text(path, error, error_size);
    if (!text) return 0;
    int ok = qw_json_parse(text, out, error, error_size);
    free(text);
    if (!ok && error && error_size) {
        size_t used = strlen(error);
        snprintf(error + used, error_size - used, " in %s", path);
    }
    return ok;
}

/* Copies a json array of numbers or booleans into an int array. */
static size_t json_int_array(const qw_json *value, int *out, size_t capacity) {
    if (!value || qw_json_type(value) != QWEN_JSON_ARRAY) return 0;
    size_t total = qw_json_count(value);
    size_t count = total < capacity ? total : capacity;
    for (size_t i = 0; i < count; i++) {
        const qw_json *item = qw_json_at(value, i);
        out[i] = qw_json_type(item) == QWEN_JSON_BOOL
            ? qw_json_bool(item)
            : (int)qw_json_number(item);
    }
    return total;
}

static size_t json_double_array(const qw_json *value, double *out,
                                size_t capacity) {
    if (!value || qw_json_type(value) != QWEN_JSON_ARRAY) return 0;
    size_t total = qw_json_count(value);
    size_t count = total < capacity ? total : capacity;
    for (size_t i = 0; i < count; i++) {
        out[i] = qw_json_number(qw_json_at(value, i));
    }
    return total;
}

static int scan_component(const char *root, const char *name,
                          h3_component_info *info, char *error,
                          size_t error_size) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", root, name);
    if (!dir_exists(path)) {
        snprintf(error, error_size, "missing component directory %s", path);
        return 0;
    }
    return h3_st_inventory_dir(path, info, error, error_size);
}

int qw_model_scan(const char *root, qw_model_config *config, char *error,
                  size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!root || !config) {
        snprintf(error, error_size, "invalid arguments");
        return 0;
    }
    memset(config, 0, sizeof(*config));
    snprintf(config->root, sizeof(config->root), "%s", root);

    char path[4096];

    /* ---- transformer ---- */
    snprintf(path, sizeof(path), "%s/transformer/config.json", root);
    qw_json *transformer = NULL;
    if (!parse_config_file(path, &transformer, error, error_size)) return 0;
    config->dit_layers = (int)qw_json_get_number(transformer, "num_layers", 32);
    config->dit_heads = (int)qw_json_get_number(transformer, "num_attention_heads", 32);
    config->dit_head_dim = (int)qw_json_get_number(transformer, "attention_head_dim", 128);
    config->dit_in_channels = (int)qw_json_get_number(transformer, "in_channels", 64);
    config->dit_out_channels = (int)qw_json_get_number(transformer, "out_channels", 64);
    config->dit_context_dim = (int)qw_json_get_number(transformer, "context_in_dim", 4096);
    config->dit_mlp_ratio = (int)qw_json_get_number(transformer, "mlp_ratio", 3);
    config->dit_patch_size = (int)qw_json_get_number(transformer, "patch_size", 1);
    config->dit_eps = qw_json_get_number(transformer, "eps", 1e-6);
    config->causal_condition = qw_json_get_bool(transformer, "causal_condition", 1);
    {
        const qw_json *axes = qw_json_get(transformer, "axes_dims_rope");
        int fallback[3] = {16, 56, 56};
        size_t count = json_int_array(axes, config->dit_rope_axes, 3);
        if (count != 3) memcpy(config->dit_rope_axes, fallback, sizeof(fallback));
    }
    qw_json_free(transformer);

    /* ---- text_encoder ---- */
    snprintf(path, sizeof(path), "%s/text_encoder/config.json", root);
    qw_json *text_encoder = NULL;
    if (!parse_config_file(path, &text_encoder, error, error_size)) return 0;
    const qw_json *text_config = qw_json_get(text_encoder, "text_config");
    if (!text_config) {
        qw_json_free(text_encoder);
        snprintf(error, error_size, "text_encoder config lacks text_config");
        return 0;
    }
    config->text_layers = (int)qw_json_get_number(text_config, "num_hidden_layers", 36);
    config->text_hidden = (int)qw_json_get_number(text_config, "hidden_size", 4096);
    config->text_mlp = (int)qw_json_get_number(text_config, "intermediate_size", 12288);
    config->text_heads = (int)qw_json_get_number(text_config, "num_attention_heads", 32);
    config->text_kv_heads = (int)qw_json_get_number(text_config, "num_key_value_heads", 8);
    config->text_head_dim = (int)qw_json_get_number(text_config, "head_dim", 128);
    config->text_rope_theta = qw_json_get_number(text_config, "rope_theta", 5000000.0);
    config->text_rms_eps = qw_json_get_number(text_config, "rms_norm_eps", 1e-6);
    config->text_vocab = (size_t)qw_json_get_number(text_config, "vocab_size", 151936);
    {
        const qw_json *scaling = qw_json_get(text_config, "rope_scaling");
        const qw_json *section = qw_json_get(scaling, "mrope_section");
        int fallback[3] = {24, 20, 20};
        size_t count = json_int_array(section, config->text_mrope, 3);
        if (count != 3) memcpy(config->text_mrope, fallback, sizeof(fallback));
    }
    qw_json_free(text_encoder);

    /* ---- vae ---- */
    snprintf(path, sizeof(path), "%s/vae/config.json", root);
    qw_json *vae = NULL;
    if (!parse_config_file(path, &vae, error, error_size)) return 0;
    config->vae_z_dim = (int)qw_json_get_number(vae, "z_dim", 64);
    config->vae_in_channels = (int)qw_json_get_number(vae, "in_channels", 4);
    config->vae_out_channels = (int)qw_json_get_number(vae, "out_channels", 4);
    config->vae_base_dim = (int)qw_json_get_number(vae, "base_dim", 96);
    config->vae_decoder_base_dim = (int)qw_json_get_number(vae, "decoder_base_dim", 144);
    config->vae_num_res_blocks = (int)qw_json_get_number(vae, "num_res_blocks", 2);
    config->vae_scale_spatial = (int)qw_json_get_number(vae, "scale_factor_spatial", 16);
    config->vae_scale_temporal = (int)qw_json_get_number(vae, "scale_factor_temporal", 8);
    config->vae_dim_mult_count =
        json_int_array(qw_json_get(vae, "dim_mult"), config->vae_dim_mult, 8);
    config->vae_temperal_count = json_int_array(
        qw_json_get(vae, "temperal_downsample"), config->vae_temperal_downsample, 8);
    config->vae_latents_count = json_double_array(
        qw_json_get(vae, "latents_mean"), config->vae_latents_mean, 64);
    size_t std_count =
        json_double_array(qw_json_get(vae, "latents_std"), config->vae_latents_std, 64);
    /* json_*_array clamp the copy to the fixed array but return the true
     * element count: an oversized config would leave count > array size
     * and any consumer iterating the count reads past the array. */
    if (config->vae_dim_mult_count > 8 || config->vae_temperal_count > 8 ||
        config->vae_latents_count > 64 || std_count > 64) {
        qw_json_free(vae);
        snprintf(error, error_size, "vae config arrays exceed fixed sizes");
        return 0;
    }
    if (std_count != config->vae_latents_count) {
        qw_json_free(vae);
        snprintf(error, error_size, "vae latents_mean/std length mismatch");
        return 0;
    }
    qw_json_free(vae);

    /* ---- scheduler ---- */
    snprintf(path, sizeof(path), "%s/scheduler/scheduler_config.json", root);
    qw_json *scheduler = NULL;
    if (!parse_config_file(path, &scheduler, error, error_size)) return 0;
    config->sched_base_shift = qw_json_get_number(scheduler, "base_shift", 0.5);
    config->sched_max_shift = qw_json_get_number(scheduler, "max_shift", 0.9);
    config->sched_shift_terminal = qw_json_get_number(scheduler, "shift_terminal", 0.02);
    config->sched_base_seq = (int)qw_json_get_number(scheduler, "base_image_seq_len", 256);
    config->sched_max_seq = (int)qw_json_get_number(scheduler, "max_image_seq_len", 8192);
    config->sched_dynamic_shifting = qw_json_get_bool(scheduler, "use_dynamic_shifting", 1);
    qw_json_free(scheduler);

    /* ---- weight inventories ---- */
    if (!scan_component(root, "transformer", &config->transformer_info, error,
                        error_size)) {
        return 0;
    }
    if (!scan_component(root, "text_encoder", &config->text_encoder_info, error,
                        error_size)) {
        return 0;
    }
    if (!scan_component(root, "vae", &config->vae_info, error, error_size)) {
        return 0;
    }
    return 1;
}

static void print_component(const char *name, const h3_component_info *info) {
    printf("  %-13s %2llu file%s %3llu tensor%s %7.2f GiB (%.2f GiB tensor data)\n",
           name, (unsigned long long)info->files, info->files == 1 ? "" : "s",
           (unsigned long long)info->tensors, info->tensors == 1 ? "" : "s",
           (double)info->bytes / 1073741824.0,
           (double)info->tensor_bytes / 1073741824.0);
}

void qw_model_print(const qw_model_config *config) {
    printf("Qwen-Image-2.1 checkpoint: %s\n", config->root);
    printf("transformer:\n");
    printf("  single-stream DiT: %d layers, hidden %d (%d heads x %d), "
           "mlp %d (ratio %d), eps %g\n",
           config->dit_layers, config->dit_heads * config->dit_head_dim,
           config->dit_heads, config->dit_head_dim,
           config->dit_heads * config->dit_head_dim * config->dit_mlp_ratio,
           config->dit_mlp_ratio, config->dit_eps);
    printf("  latents %d channels in / %d out, context %d, patch %d, "
           "rope axes %d/%d/%d, causal_condition %s\n",
           config->dit_in_channels, config->dit_out_channels,
           config->dit_context_dim, config->dit_patch_size,
           config->dit_rope_axes[0], config->dit_rope_axes[1],
           config->dit_rope_axes[2],
           config->causal_condition ? "yes" : "no");
    printf("text encoder (Qwen3-VL language tower):\n");
    printf("  %d layers, hidden %d, mlp %d, GQA %d/%d heads x %d, "
           "vocab %zu\n",
           config->text_layers, config->text_hidden, config->text_mlp,
           config->text_heads, config->text_kv_heads, config->text_head_dim,
           config->text_vocab);
    printf("  rope theta %g, interleaved mrope section %d/%d/%d, "
           "rms eps %g\n",
           config->text_rope_theta, config->text_mrope[0],
           config->text_mrope[1], config->text_mrope[2], config->text_rms_eps);
    printf("vae:\n");
    printf("  z_dim %d, channels %d in / %d out (RGBA-capable), base %d, "
           "decoder base %d, res blocks %d\n",
           config->vae_z_dim, config->vae_in_channels,
           config->vae_out_channels, config->vae_base_dim,
           config->vae_decoder_base_dim, config->vae_num_res_blocks);
    printf("  dim_mult");
    for (size_t i = 0; i < config->vae_dim_mult_count; i++) {
        printf(" %d", config->vae_dim_mult[i]);
    }
    printf(", temperal_downsample");
    for (size_t i = 0; i < config->vae_temperal_count; i++) {
        printf(" %s", config->vae_temperal_downsample[i] ? "T" : "F");
    }
    printf(" (%dx spatial, %dx temporal)\n", config->vae_scale_spatial,
           config->vae_scale_temporal);
    printf("scheduler: flow-match euler, dynamic shift %s (%g..%g over "
           "seq %d..%d), terminal %g\n",
           config->sched_dynamic_shifting ? "on" : "off",
           config->sched_base_shift, config->sched_max_shift,
           config->sched_base_seq, config->sched_max_seq,
           config->sched_shift_terminal);
    printf("weights:\n");
    print_component("transformer", &config->transformer_info);
    print_component("text_encoder", &config->text_encoder_info);
    print_component("vae", &config->vae_info);
}
