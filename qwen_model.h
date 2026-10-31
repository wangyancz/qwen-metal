#ifndef QWEN_MODEL_H
#define QWEN_MODEL_H

#include "h3.h" /* h3_component_info (vendored inventory struct) */

#include <stddef.h>

/* Scanned configuration of a released Qwen-Image-2.1 checkpoint:
 *
 *   model_index.json
 *   transformer/config.json
 *   text_encoder/config.json
 *   vae/config.json
 *   scheduler/scheduler_config.json
 *
 * Every field carries the released default, so a missing optional key
 * still yields a usable description; required files are enforced. */
typedef struct {
    char root[4096];

    /* transformer/config.json */
    int dit_layers;
    int dit_heads;
    int dit_head_dim;
    int dit_in_channels;
    int dit_out_channels;
    int dit_context_dim;
    int dit_mlp_ratio;
    int dit_patch_size;
    int dit_rope_axes[3];
    double dit_eps;
    int causal_condition;

    /* text_encoder/config.json (text_config) */
    int text_layers;
    int text_hidden;
    int text_mlp;
    int text_heads;
    int text_kv_heads;
    int text_head_dim;
    double text_rope_theta;
    double text_rms_eps;
    int text_mrope[3];
    size_t text_vocab;

    /* vae/config.json */
    int vae_z_dim;
    int vae_in_channels;
    int vae_out_channels;
    int vae_base_dim;
    int vae_decoder_base_dim;
    int vae_num_res_blocks;
    int vae_dim_mult[8];
    size_t vae_dim_mult_count;
    int vae_temperal_downsample[8];
    size_t vae_temperal_count;
    int vae_scale_spatial;
    int vae_scale_temporal;
    double vae_latents_mean[64];
    double vae_latents_std[64];
    size_t vae_latents_count;

    /* scheduler/scheduler_config.json */
    double sched_base_shift;
    double sched_max_shift;
    double sched_shift_terminal;
    int sched_base_seq;
    int sched_max_seq;
    int sched_dynamic_shifting;

    /* Weight inventories per component directory. */
    h3_component_info transformer_info;
    h3_component_info text_encoder_info;
    h3_component_info vae_info;
} qw_model_config;

int qw_model_scan(const char *root, qw_model_config *config, char *error,
                  size_t error_size);
void qw_model_print(const qw_model_config *config);

#endif
