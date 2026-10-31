#ifndef QWEN_VAE_H
#define QWEN_VAE_H

#include "h3_gpu.h"

#include <stddef.h>

/* Qwen-Image-2.1 VAE decoder (AutoencoderKLQwenImage21), image path.
 *
 * The released decoder is a Wan-heritage causal 3D network, but the image
 * specialization folds the temporal axis away (QwenImage21CausalConv3d
 * squeezes T), and the first-call "Rep" path of the temporal upsampler
 * skips its time_conv entirely — so decoding a single latent frame is a
 * pure 2D network. All activations are fp32 (production runs the VAE in
 * fp32) with channels-last [H, W, C] rows. */

#define QWEN_VAE_Z 64

typedef struct qw_vae qw_vae;

qw_vae *qw_vae_create(h3_gpu *gpu, const char *model_dir,
                      char *error, size_t error_size);
void qw_vae_free(qw_vae *vae);

/* Decode one denormalized latent frame.
 *   z:      fp32 tensor of 64 * latent_height * latent_width elements,
 *           NCHW [1, 64, latent_height, latent_width].
 *   output: fp32 tensor of 4 * (16H) * (16W) elements, NCHW [4, 16H, 16W],
 *           clamped to [-1, 1] (RGBA; the alpha channel is channel 3).
 * Non-square latents are supported (the official aspect-ratio table).
 * Set QWEN_VAE_DUMP=<dir> to dump stage tensors for the parity harness. */
int qw_vae_decode(qw_vae *vae, const h3_gpu_tensor *z,
                  uint32_t latent_height, uint32_t latent_width,
                  h3_gpu_tensor *output, char *error, size_t error_size);

/* Encoder pass (ti2i): input [H, W, 4] channels-last fp32 in [-1, 1];
 * output packed [H/16 * W/16, 64] bf16 rows (mu, mean/std normalized),
 * row-major over [h][w]. */
int qw_vae_encode(qw_vae *vae, const h3_gpu_tensor *input, uint32_t height,
                  uint32_t width, h3_gpu_tensor *latents, char *error,
                  size_t error_size);

/* Latent normalization statistics (vae/config.json latents_mean/std);
 * NULL before qw_pipe_decode uses them when absent. */
const float *qw_vae_latents_mean(const qw_vae *vae);
const float *qw_vae_latents_std(const qw_vae *vae);

#endif /* QWEN_VAE_H */
