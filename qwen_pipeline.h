#ifndef QWEN_PIPELINE_H
#define QWEN_PIPELINE_H

#include "h3_gpu.h"
#include "qwen_dit.h"
#include "qwen_host.h"
#include "qwen_text_encoder.h"
#include "qwen_vae.h"
#include "qwen_vision.h"

#include <stddef.h>
#include <stdint.h>

/* Text-to-image generation: the released QwenImage21Pipeline loop
 * (FlowMatchEulerDiscreteScheduler with exponential dynamic shifting)
 * on top of the parity-proven DiT / VAE. Latents run through the loop in
 * fp32 with bf16 casts at the transformer input, mirroring the released
 * bf16 dtype flow. */

typedef struct {
    uint32_t width;      /* output pixels, multiple of 32 */
    uint32_t height;
    uint32_t steps;      /* denoise steps (turbo models: 8) */
    uint64_t seed;
    float cfg_scale;     /* > 1 enables classifier-free guidance */
    const char *negative; /* CFG negative prompt (cfg > 1 requires it) */
    uint32_t output_res; /* ti2i: reference-conditioned output resolution
                          * (calculate_dimensions square area; 0 = width/
                          * height as-is, t2i) */
} qw_pipe_params;

/* Deterministic seeded RNG for the initial latents (xorshift64* +
 * Box-Muller; exact torch parity is neither possible nor required). */
void qw_pipe_seed(uint64_t seed);
float qw_pipe_randn(void);

/* Optional per-step progress hook for embedders (the serve worker):
 * invoked after each completed denoising step — t2i and ti2i loops
 * alike — with the 1-based completed step count and the total. NULL
 * clears the hook (the CLI/tests default). */
typedef void (*qw_pipe_progress_fn)(uint32_t steps_done, uint32_t steps,
                                    void *ctx);
void qw_pipe_set_progress(qw_pipe_progress_fn cb, void *ctx);

/* Scheduler behaviour from scheduler/scheduler_config.json — the two
 * released checkpoints differ: the base model ships
 * use_dynamic_shifting=true with a 0.02 terminal stretch, while the
 * turbo distillation ships false/null (its released sample_sigmas are
 * consumed as-is). Applying the base rules to the turbo schedule
 * corrupts the sigma trajectory (soft images). */
typedef struct {
    int dynamic_shifting;   /* exponential mu shift (base: on) */
    double base_seq;        /* mu = exp(seq_len * m + b) parameters */
    double max_seq;
    double base_shift;
    double max_shift;
    int has_terminal;       /* shift_terminal present (base: 0.02) */
    double shift_terminal;
} qw_sched_cfg;

/* Denoiser sigmas (steps + 1 entries, ending at 0). Mirrors
 * FlowMatchEulerDiscreteScheduler.set_timesteps with the released
 * scheduler config (cfg): fp32 linspace(1, 1/steps), then — only when
 * the config enables it — the exponential dynamic shift with
 * mu = f(seq_len) and the terminal stretch. turbo_sigmas (optional)
 * replaces the linspace input (the released sample_sigmas list); the
 * turbo config consumes them as-is, exactly like diffusers. Returns 0
 * on failure. */
int qw_pipe_schedule(uint32_t steps, uint32_t seq_len,
                     const double *turbo_sigmas, uint32_t turbo_count,
                     const qw_sched_cfg *cfg,
                     double *sigmas_out, char *error, size_t error_size);

/* Joint RoPE tables [text_rows + gh*gw, 64] fp32 (interleaved pairs,
 * axes_dim [16, 56, 56], theta 1e4). Text rows ride position t on all
 * three axes; image tokens carry the constant frame position text_rows
 * with centered height/width indices. */
int qw_pipe_rope_tables(h3_gpu *gpu, uint32_t text_rows, uint32_t gh,
                        uint32_t gw, h3_gpu_tensor **rope_cos,
                        h3_gpu_tensor **rope_sin, char *error,
                        size_t error_size);

/* Run the denoising loop. latents is [gh*gw, 64] fp32 (normalized
 * space); text tensors are the txt_in inputs (post text encoder).
 * cfg_scale > 1 requires neg_text_in. Uses qw_dit_forward for every
 * step and euler updates sigma_next - sigma. */
int qw_pipe_denoise(qw_dit *dit, h3_gpu *gpu, const h3_gpu_tensor *text_in,
                    uint32_t text_rows, const h3_gpu_tensor *neg_text_in,
                    uint32_t neg_rows, float *latents, uint32_t gh,
                    uint32_t gw, uint32_t steps, float cfg_scale,
                    const double *turbo_sigmas, uint32_t turbo_count,
                    const qw_sched_cfg *sched,
                    char *error, size_t error_size);

/* Decode denoised latents to an RGBA image (width x height, 8bit).
 * Applies the inverse latent normalization (z * std + mean) before the
 * VAE and clamps to sRGB. */
int qw_pipe_decode(qw_vae *vae, h3_gpu *gpu, const float *latents,
                   uint32_t gh, uint32_t gw, uint8_t **rgba_out,
                   char *error, size_t error_size);

/* Full text-to-image: caller supplies the txt_in inputs. Output is a
 * malloc'd RGBA buffer. */
int qw_pipe_generate(qw_dit *dit, qw_vae *vae, h3_gpu *gpu,
                     const h3_gpu_tensor *text_in, uint32_t text_rows,
                     const h3_gpu_tensor *neg_text_in, uint32_t neg_rows,
                     const qw_pipe_params *params, const double *turbo_sigmas,
                     uint32_t turbo_count, const qw_sched_cfg *sched,
                     uint8_t **rgba_out, char *error, size_t error_size);

/* ti2i 多块 rope：text 段三轴共用递增位置、image 块 frame 冻结在当前
 * position 且 h/w 走居中零基准网格、position += max(h, w)。block_h/w 与
 * text_before（各块前文本行数）按官方 QwenImage21Rope。行数 =
 * Σtext_before + Σh*w。 */
int qw_pipe_rope_tables_blocks(h3_gpu *gpu, const uint32_t *block_h,
                               const uint32_t *block_w,
                               const uint32_t *text_before,
                               uint32_t n_blocks, h3_gpu_tensor **rope_cos,
                               h3_gpu_tensor **rope_sin, char *error,
                               size_t error_size);

/* ti2i 生成（多参考图）：完整管线——逐图白底合成/resize → 视觉塔（多图
 * 一次前向）→ 文本塔 ti2i 组装 → VAE encode 各图条件 latents → joint 序列
 * 去噪 → decode。rgba_images[i] 为 [in_ws[i]*in_hs[i]*4]。
 * cfg_scale > 1 需要 params->negative（缺失时告警并降回 1，官方同款）；
 * width/height 必须为非零的 32 倍数（入口校验）。 */
#define QW_TI2I_MAX_IMAGES 8
int qw_pipe_generate_ti2i(qw_dit *dit, qw_vae *vae, qw_text *text,
                          qw_vision *vision, h3_gpu *gpu,
                          const char *prompt, const uint8_t *const *rgba_images,
                          const uint32_t *in_ws, const uint32_t *in_hs,
                          uint32_t n_images, const qw_pipe_params *params,
                          const double *turbo_sigmas, uint32_t turbo_count,
                          const qw_sched_cfg *sched, uint8_t **rgba_out,
                          char *error, size_t error_size);

#endif
