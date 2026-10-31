#include "qwen_pipeline.h"

#include <mach/mach.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Seeded RNG: splitmix64 warmup + Box-Muller. Exact torch parity is
 * neither possible nor required (the reference trajectory lives on a
 * different device); reproducibility per seed is what matters. */
static uint64_t pipe_rng_state;
static int pipe_rng_has_spare;
static float pipe_rng_spare;

void qw_pipe_seed(uint64_t seed) {
    pipe_rng_state = seed ? seed : 0x9e3779b97f4a7c15ull;
    pipe_rng_state ^= pipe_rng_state >> 30;
    pipe_rng_state *= 0xbf58476d1ce4e5b9ull;
    pipe_rng_state ^= pipe_rng_state >> 27;
    pipe_rng_state *= 0x94d049bb133111ebull;
    pipe_rng_state ^= pipe_rng_state >> 31;
    pipe_rng_has_spare = 0;
}

static uint64_t pipe_rng_next(void) {
    uint64_t z = pipe_rng_state;
    pipe_rng_state = z * 0x9e3779b97f4a7c15ull + 0x94d049bb133111ebull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

float qw_pipe_randn(void) {
    if (pipe_rng_has_spare) {
        pipe_rng_has_spare = 0;
        return pipe_rng_spare;
    }
    double u1 = (double)(pipe_rng_next() >> 11) * (1.0 / 9007199254740992.0);
    double u2 = (double)(pipe_rng_next() >> 11) * (1.0 / 9007199254740992.0);
    if (u1 < 1e-300) u1 = 1e-300;
    double radius = sqrt(-2.0 * log(u1));
    double angle = 2.0 * 3.14159265358979323846 * u2;
    pipe_rng_spare = (float)(radius * sin(angle));
    pipe_rng_has_spare = 1;
    return (float)(radius * cos(angle));
}

/* Progress hook (embedders, e.g. the serve worker): fired after every
 * completed denoising step from either the t2i or the ti2i loop. */
static qw_pipe_progress_fn pipe_progress_cb;
static void *pipe_progress_ctx;

/* QWEN_TI2I_TIMING=1: per-section wall-clock marks on stderr (the
 * encode phase of ti2i scales with the reference-image count; these
 * marks locate which section). */
static int pipe_timing_on = -1;
/* Set once by qw_pipe_generate_ti2i: lets pipe_mark also report h3 live
 * bytes + phys_footprint per stage (QWEN_SERVE_MEM=1) so a memory climb
 * can be attributed to the request stage that caused it. */
static h3_gpu *pipe_mark_gpu;

static double pipe_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double pipe_footprint_gb(void) {
    struct task_vm_info info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info,
                  &count) != KERN_SUCCESS)
        return -1.0;
    return (double)info.phys_footprint / 1e9;
}

static double pipe_timing_t0;

static void pipe_mark(const char *label) {
    if (pipe_timing_on < 0)
        pipe_timing_on = getenv("QWEN_TI2I_TIMING") != NULL;
    if (!pipe_timing_on) return;
    double now = pipe_now();
    if (pipe_mark_gpu && getenv("QWEN_SERVE_MEM")) {
        h3_gpu_stats st;
        h3_gpu_get_stats(pipe_mark_gpu, &st);
        fprintf(stderr,
                "[ti2i] %-22s %7.2fs (+%.2fs) live=%.2fGB footprint=%.2fGB\n",
                label, now - pipe_timing_t0, now - pipe_timing_t0,
                (double)st.live_bytes / 1e9, pipe_footprint_gb());
    } else {
        fprintf(stderr, "[ti2i] %-22s %7.2fs (+%.2fs)\n", label,
                now - pipe_timing_t0, now - pipe_timing_t0);
    }
    pipe_timing_t0 = now;
}

void qw_pipe_set_progress(qw_pipe_progress_fn cb, void *ctx) {
    pipe_progress_cb = cb;
    pipe_progress_ctx = ctx;
}

static void pipe_progress(uint32_t steps_done, uint32_t steps) {
    if (pipe_progress_cb) pipe_progress_cb(steps_done, steps, pipe_progress_ctx);
}

/* Scheduler: mirrors FlowMatchEulerDiscreteScheduler.set_timesteps with
 * the released config (exponential dynamic shifting, terminal 0.02).
 * Known divergence: the reference keeps the timestep in bf16 (t is
 * quantized to bf16 before timestep_features), we feed fp32 — a ~0.2%
 * trajectory offset, visually imperceptible; t2i parity was validated
 * with the fp32 path. */
static float fp32(double value) { return (float)value; }

int qw_pipe_schedule(uint32_t steps, uint32_t seq_len,
                     const double *turbo_sigmas, uint32_t turbo_count,
                     const qw_sched_cfg *cfg,
                     double *sigmas_out, char *error, size_t error_size) {
    if (!steps || !seq_len || !sigmas_out) {
        snprintf(error, error_size, "schedule: bad arguments");
        return 0;
    }
    if (turbo_sigmas && turbo_count != steps) {
        snprintf(error, error_size, "schedule: turbo sigmas %u != steps %u",
                 turbo_count, steps);
        return 0;
    }
    if (!cfg) {
        snprintf(error, error_size, "schedule: missing scheduler config");
        return 0;
    }
    double exp_mu = 0.0;
    if (cfg->dynamic_shifting) {
        /* A corrupt config (max_seq == base_seq) would make m infinite
         * and every sigma NaN — the run then silently produces pure
         * noise. Collapse to the constant base shift instead. */
        double m = cfg->max_seq != cfg->base_seq
                       ? (cfg->max_shift - cfg->base_shift) /
                             (cfg->max_seq - cfg->base_seq)
                       : 0.0;
        double b = cfg->base_shift - m * cfg->base_seq;
        exp_mu = exp((double)seq_len * m + b);
    }

    float *shifted = malloc(sizeof(float) * steps);
    if (!shifted) {
        snprintf(error, error_size, "schedule: out of memory");
        return 0;
    }
    for (uint32_t i = 0; i < steps; i++) {
        double raw = turbo_sigmas
                         ? turbo_sigmas[i]
                         : (steps == 1
                                ? 1.0
                                : 1.0 + (double)i * ((1.0 / steps) - 1.0) /
                                            (double)(steps - 1));
        float t = fp32(raw);
        float value;
        if (cfg->dynamic_shifting) {
            float inverse = fp32(1.0 / t);
            float base = fp32(inverse - 1.0f);
            value = fp32(exp_mu / (exp_mu + (double)base));
        } else {
            /* Static shift; every released scheduler config ships
             * shift = 1.0, which makes this a no-op — the turbo
             * distillation's sample_sigmas are consumed as-is. */
            double s = 1.0;
            value = fp32(s * raw / (1.0 + (s - 1.0) * raw));
        }
        shifted[i] = value;
    }
    if (cfg->has_terminal) {
        float tail = fp32(1.0 - (double)shifted[steps - 1]);
        /* shift_terminal >= 1 would divide by zero and pin every sigma
         * to 1.0 (dt=0 steps); treat it as "no terminal stretch". */
        float scale = cfg->shift_terminal < 1.0f
                          ? fp32((double)tail / (1.0 - cfg->shift_terminal))
                          : 0.0f;
        if (scale != 0.0f) {
            for (uint32_t i = 0; i < steps; i++) {
                float one_minus = fp32(1.0 - (double)shifted[i]);
                shifted[i] = fp32(1.0 - (double)one_minus / (double)scale);
            }
        }
    }
    for (uint32_t i = 0; i < steps; i++) sigmas_out[i] = shifted[i];
    sigmas_out[steps] = 0.0;
    free(shifted);
    return 1;
}

/* RoPE tables [text_rows + gh*gw, 64] fp32, interleaved pairs, axes
 * (16, 56, 56), theta 1e4. Text tokens ride position t on all three
 * axes; image tokens keep the constant frame position text_rows with
 * centered height/width indices (negative positions fine: cos/sin of
 * negative angles match the reference table's negative entries). */
int qw_pipe_rope_tables(h3_gpu *gpu, uint32_t text_rows, uint32_t gh,
                        uint32_t gw, h3_gpu_tensor **rope_cos,
                        h3_gpu_tensor **rope_sin, char *error,
                        size_t error_size) {
    if (!text_rows || !gh || !gw) {
        snprintf(error, error_size, "rope: bad arguments");
        return 0;
    }
    uint32_t rows = text_rows + gh * gw;
    h3_gpu_tensor *cos_tensor = h3_gpu_tensor_new_f32(gpu, (size_t)rows * 64);
    h3_gpu_tensor *sin_tensor = h3_gpu_tensor_new_f32(gpu, (size_t)rows * 64);
    float *cos_host = malloc(sizeof(float) * rows * 64);
    float *sin_host = malloc(sizeof(float) * rows * 64);
    int ok = cos_tensor && sin_tensor && cos_host && sin_host;
    static const double axis_dim[3] = {16.0, 56.0, 56.0};
    long half_h = (long)(gh - gh / 2);
    long half_w = (long)(gw - gw / 2);
    for (uint32_t t = 0; ok && t < rows; t++) {
        double position[3];
        if (t < text_rows) {
            position[0] = position[1] = position[2] = (double)t;
        } else {
            uint32_t index = t - text_rows;
            uint32_t h = index / gw;
            uint32_t w = index % gw;
            position[0] = (double)text_rows;
            position[1] = (double)((long)h - half_h);
            position[2] = (double)((long)w - half_w);
        }
        size_t base = (size_t)t * 64;
        size_t pair = 0;
        for (int axis = 0; axis < 3; axis++) {
            uint32_t pairs = (uint32_t)(axis_dim[axis] / 2.0);
            for (uint32_t p = 0; p < pairs; p++, pair++) {
                double frequency =
                    pow(10000.0, -2.0 * (double)p / axis_dim[axis]);
                double angle = position[axis] * frequency;
                cos_host[base + pair] = (float)cos(angle);
                sin_host[base + pair] = (float)sin(angle);
            }
        }
    }
    if (ok)
        ok = h3_gpu_tensor_write_f32(cos_tensor, cos_host,
                                     (size_t)rows * 64) &&
             h3_gpu_tensor_write_f32(sin_tensor, sin_host,
                                     (size_t)rows * 64);
    free(cos_host);
    free(sin_host);
    if (!ok) {
        h3_gpu_tensor_free(cos_tensor);
        h3_gpu_tensor_free(sin_tensor);
        snprintf(error, error_size, "rope: table build failed");
        return 0;
    }
    *rope_cos = cos_tensor;
    *rope_sin = sin_tensor;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Denoise loop                                                        */

static uint16_t host_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float host_from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

int qw_pipe_denoise(qw_dit *dit, h3_gpu *gpu, const h3_gpu_tensor *text_in,
                    uint32_t text_rows, const h3_gpu_tensor *neg_text_in,
                    uint32_t neg_rows, float *latents, uint32_t gh,
                    uint32_t gw, uint32_t steps, float cfg_scale,
                    const double *turbo_sigmas, uint32_t turbo_count,
                    const qw_sched_cfg *sched,
                    char *error, size_t error_size) {
    uint32_t image_rows = gh * gw;
    size_t count = (size_t)image_rows * 64;
    int guided = cfg_scale > 1.0f && neg_text_in && neg_rows;
    if (cfg_scale > 1.0f && !guided)
        fprintf(stderr,
                "t2i: cfg %.2f without a negative prompt stays at 1.0\n",
                cfg_scale);
    double *sigmas = malloc(sizeof(double) * ((size_t)steps + 1));
    uint16_t *latent_host = malloc(sizeof(uint16_t) * count);
    uint16_t *pred_host =
        malloc(sizeof(uint16_t) * (size_t)(text_rows + image_rows) * 64);
    float *neg_host = guided ? malloc(sizeof(float) * count) : NULL;
    size_t pos_pred_count = (size_t)(text_rows + image_rows) * 64;
    size_t neg_pred_count =
        guided ? (size_t)(neg_rows + image_rows) * 64 : 0;
    uint16_t *neg_pred_host =
        guided ? malloc(sizeof(uint16_t) * neg_pred_count) : NULL;
    h3_gpu_tensor *rope_cos = NULL, *rope_sin = NULL;
    h3_gpu_tensor *neg_cos = NULL, *neg_sin = NULL;
    h3_gpu_tensor *latent_in = NULL, *pred = NULL, *neg_pred = NULL;
    int ok = sigmas && latent_host && pred_host &&
             (!guided || (neg_host && neg_pred_host));
    if (!ok) {
        snprintf(error, error_size, "denoise: out of memory");
        goto done;
    }
    /* Pre-loop failures must clear ok: it still holds the malloc-check
     * result here, and returning 1 would report success with the raw
     * initial-noise latents (observed as garbage images with broken
     * alpha once the serve worker exposed step-count mismatches). */
    if (!qw_pipe_schedule(steps, image_rows, turbo_sigmas, turbo_count,
                          sched, sigmas, error, error_size)) {
        ok = 0;
        goto done;
    }
    if (!qw_pipe_rope_tables(gpu, text_rows, gh, gw, &rope_cos, &rope_sin,
                             error, error_size)) {
        ok = 0;
        goto done;
    }
    if (guided &&
        !qw_pipe_rope_tables(gpu, neg_rows, gh, gw, &neg_cos, &neg_sin, error,
                             error_size)) {
        ok = 0;
        goto done;
    }
    latent_in = h3_gpu_tensor_new_bf16(gpu, count);
    pred = h3_gpu_tensor_new_bf16(gpu, (size_t)(text_rows + image_rows) * 64);
    if (guided)
        neg_pred = h3_gpu_tensor_new_bf16(
            gpu, (size_t)(neg_rows + image_rows) * 64);
    ok = latent_in && pred && (!guided || neg_pred);
    if (!ok) {
        snprintf(error, error_size, "denoise: allocation failed");
        goto done;
    }

    for (uint32_t step = 0; ok && step < steps; step++) {
        for (size_t i = 0; i < count; i++)
            latent_host[i] = host_bf16(latents[i]);
        if (!h3_gpu_tensor_write_bf16(latent_in, latent_host, count)) {
            snprintf(error, error_size, "denoise: latent upload failed");
            ok = 0;
            break;
        }
        if (!qw_dit_forward(dit, text_in, text_rows, latent_in, image_rows,
                            (float)sigmas[step], rope_cos, rope_sin, pred,
                            error, error_size)) {
            ok = 0;
            break;
        }
        if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) {
            snprintf(error, error_size, "%s", h3_gpu_error(gpu));
            ok = 0;
            break;
        }
        if (!h3_gpu_tensor_read_bf16(pred, pred_host, pos_pred_count)) {
            snprintf(error, error_size, "denoise: pred read failed");
            ok = 0;
            break;
        }
        if (guided) {
            if (!qw_dit_forward(dit, neg_text_in, neg_rows, latent_in,
                                image_rows, (float)sigmas[step], neg_cos,
                                neg_sin, neg_pred, error, error_size)) {
                ok = 0;
                break;
            }
            if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) {
                snprintf(error, error_size, "%s", h3_gpu_error(gpu));
                ok = 0;
                break;
            }
            if (!h3_gpu_tensor_read_bf16(neg_pred, neg_pred_host,
                                         neg_pred_count)) {
                snprintf(error, error_size, "denoise: neg pred read failed");
                ok = 0;
                break;
            }
            for (size_t i = 0; i < count; i++)
                neg_host[i] = host_from_bf16(neg_pred_host[
                    (size_t)neg_rows * 64 + i]);
            /* cond 分支的 pred 已在上方读过——每步恰两次 forward。 */
        }
        float dt = (float)(sigmas[step + 1] - sigmas[step]);
        for (size_t i = 0; i < count; i++) {
            float prediction = host_from_bf16(
                pred_host[(size_t)text_rows * 64 + i]);
            if (guided)
                prediction = neg_host[i] +
                             cfg_scale * (prediction - neg_host[i]);
            latents[i] += dt * prediction;
        }
        pipe_progress(step + 1, steps);
    }

done:
    h3_gpu_tensor_free(latent_in);
    h3_gpu_tensor_free(pred);
    h3_gpu_tensor_free(neg_pred);
    h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(neg_cos);
    h3_gpu_tensor_free(neg_sin);
    free(sigmas);
    free(latent_host);
    free(pred_host);
    free(neg_pred_host);
    free(neg_host);
    return ok;
}

/* ------------------------------------------------------------------ */
/* Decode                                                              */

int qw_pipe_decode(qw_vae *vae, h3_gpu *gpu, const float *latents,
                   uint32_t gh, uint32_t gw, uint8_t **rgba_out,
                   char *error, size_t error_size) {
    uint32_t pixels = gh * gw;
    uint32_t out_w = gw * 16, out_h = gh * 16;
    size_t z_count = (size_t)pixels * QWEN_VAE_Z;
    float *z_host = malloc(sizeof(float) * z_count);
    uint8_t *rgba = malloc((size_t)out_w * out_h * 4);
    float *frame = malloc(sizeof(float) * (size_t)out_w * out_h * 4);
    h3_gpu_tensor *z = h3_gpu_tensor_new_f32(gpu, z_count);
    h3_gpu_tensor *image = h3_gpu_tensor_new_f32(
        gpu, (size_t)4 * out_w * out_h);
    int ok = z_host && rgba && frame && z && image;
    if (!ok) {
        snprintf(error, error_size, "decode: allocation failed");
        goto done;
    }
    /* channels-last rows -> NCHW planes, inverse latent normalization. */
    const float *mean = qw_vae_latents_mean(vae);
    const float *std = qw_vae_latents_std(vae);
    if (!mean || !std) {
        snprintf(error, error_size, "decode: latent statistics missing");
        ok = 0;
        goto done;
    }
    for (uint32_t c = 0; c < QWEN_VAE_Z; c++) {
        for (uint32_t p = 0; p < pixels; p++)
            z_host[(size_t)c * pixels + p] =
                latents[(size_t)p * QWEN_VAE_Z + c] * std[c] + mean[c];
    }
    ok = h3_gpu_tensor_write_f32(z, z_host, z_count);
    ok = ok && qw_vae_decode(vae, z, gh, gw, image, error, error_size);
    ok = ok && h3_gpu_submit(gpu) && h3_gpu_begin(gpu);
    ok = ok && h3_gpu_tensor_read_f32(image, frame,
                                      (size_t)4 * out_w * out_h);
    if (ok) {
        for (uint32_t y = 0; y < out_h; y++) {
            for (uint32_t x = 0; x < out_w; x++) {
                size_t base = ((size_t)y * out_w + x) * 4;
                uint8_t *pixel = rgba + base;
                for (int c = 0; c < 4; c++) {
                    float value = frame[(size_t)c * out_w * out_h +
                                        (size_t)y * out_w + x];
                    float normalized = value / 2.0f + 0.5f;
                    if (normalized < 0.0f) normalized = 0.0f;
                    if (normalized > 1.0f) normalized = 1.0f;
                    pixel[c] = (uint8_t)(normalized * 255.0f + 0.5f);
                }
            }
        }
        *rgba_out = rgba;
        rgba = NULL;
    }
done:
    h3_gpu_tensor_free(z);
    h3_gpu_tensor_free(image);
    free(z_host);
    free(frame);
    free(rgba);
    return ok;
}

int qw_pipe_generate(qw_dit *dit, qw_vae *vae, h3_gpu *gpu,
                     const h3_gpu_tensor *text_in,
                     uint32_t text_rows, const h3_gpu_tensor *neg_text_in,
                     uint32_t neg_rows, const qw_pipe_params *params,
                     const double *turbo_sigmas, uint32_t turbo_count,
                     const qw_sched_cfg *sched,
                     uint8_t **rgba_out, char *error, size_t error_size) {
    if (!params || !params->width || !params->height ||
        params->width % 32 || params->height % 32) {
        snprintf(error, error_size, "generate: size must be multiples of 32");
        return 0;
    }
    uint32_t gh = params->height / 16;
    uint32_t gw = params->width / 16;
    size_t count = (size_t)gh * gw * 64;
    float *latents = malloc(sizeof(float) * count);
    if (!latents) {
        snprintf(error, error_size, "generate: out of memory");
        return 0;
    }
    qw_pipe_seed(params->seed);
    for (size_t i = 0; i < count; i++) latents[i] = qw_pipe_randn();
    int ok = qw_pipe_denoise(dit, gpu, text_in, text_rows, neg_text_in,
                             neg_rows, latents, gh, gw, params->steps,
                             params->cfg_scale, turbo_sigmas, turbo_count,
                             sched, error, error_size);
    if (ok)
        ok = qw_pipe_decode(vae, gpu, latents, gh, gw, rgba_out, error,
                            error_size);
    free(latents);
    return ok;
}
/* ------------------------------------------------------------------ */
/* ti2i：多块 rope + 参考图条件生成                                     */

#include "qwen_text_encoder.h"
#include "qwen_vision.h"
#include "qwen_imgproc.h"

int qw_pipe_rope_tables_blocks(h3_gpu *gpu, const uint32_t *block_h,
                               const uint32_t *block_w,
                               const uint32_t *text_before,
                               uint32_t n_blocks, h3_gpu_tensor **rope_cos,
                               h3_gpu_tensor **rope_sin, char *error,
                               size_t error_size) {
    uint32_t rows = 0;
    for (uint32_t b = 0; b < n_blocks; b++)
        rows += text_before[b] + block_h[b] * block_w[b];
    if (!rows) {
        snprintf(error, error_size, "rope blocks: empty layout");
        return 0;
    }
    h3_gpu_tensor *cos_t = h3_gpu_tensor_new_f32(gpu, (size_t)rows * 64);
    h3_gpu_tensor *sin_t = h3_gpu_tensor_new_f32(gpu, (size_t)rows * 64);
    float *cos_host = malloc(sizeof(float) * (size_t)rows * 64);
    float *sin_host = malloc(sizeof(float) * (size_t)rows * 64);
    int ok = cos_t && sin_t && cos_host && sin_host;
    static const double axis_dim[3] = {16.0, 56.0, 56.0};
    double position = 0.0;
    uint32_t row = 0;
    for (uint32_t b = 0; ok && b < n_blocks; b++) {
        for (uint32_t t = 0; ok && t < text_before[b]; t++) {
            size_t base = (size_t)row * 64;
            size_t pair = 0;
            for (int axis = 0; axis < 3; axis++) {
                uint32_t pairs = (uint32_t)(axis_dim[axis] / 2.0);
                for (uint32_t p = 0; p < pairs; p++, pair++) {
                    double frequency =
                        pow(10000.0, -2.0 * (double)p / axis_dim[axis]);
                    double angle = position * frequency;
                    cos_host[base + pair] = (float)cos(angle);
                    sin_host[base + pair] = (float)sin(angle);
                }
            }
            position += 1.0;
            row++;
        }
        uint32_t gh = block_h[b], gw = block_w[b];
        long half_h = (long)(gh - gh / 2);
        long half_w = (long)(gw - gw / 2);
        double frame = position;
        for (uint32_t y = 0; ok && y < gh; y++) {
            for (uint32_t x = 0; x < gw; x++) {
                double pos[3] = {frame, (double)((long)y - half_h),
                                 (double)((long)x - half_w)};
                size_t base = (size_t)row * 64;
                size_t pair = 0;
                for (int axis = 0; axis < 3; axis++) {
                    uint32_t pairs = (uint32_t)(axis_dim[axis] / 2.0);
                    for (uint32_t p = 0; p < pairs; p++, pair++) {
                        double frequency = pow(10000.0,
                                               -2.0 * (double)p /
                                                   axis_dim[axis]);
                        double angle = pos[axis] * frequency;
                        cos_host[base + pair] = (float)cos(angle);
                        sin_host[base + pair] = (float)sin(angle);
                    }
                }
                row++;
            }
        }
        position += (gh > gw ? gh : gw);
    }
    if (ok)
        ok = h3_gpu_tensor_write_f32(cos_t, cos_host,
                                     (size_t)rows * 64) &&
             h3_gpu_tensor_write_f32(sin_t, sin_host,
                                     (size_t)rows * 64);
    free(cos_host);
    free(sin_host);
    if (!ok) {
        h3_gpu_tensor_free(cos_t);
        h3_gpu_tensor_free(sin_t);
        snprintf(error, error_size, "rope blocks: build failed");
        return 0;
    }
    *rope_cos = cos_t;
    *rope_sin = sin_t;
    return 1;
}

/* img_mask（merged 槽粒度）→ 段布局：[text0][img0]...[textN][target]。
 * rows = Σpacked 图像行 + Σ文本行 + target。 */
static int ti2i_build_segments(const uint8_t *img_mask, size_t seq,
                               const uint32_t *gh16, const uint32_t *gw16,
                               const uint32_t *cond_arr, uint32_t n_images,
                               uint32_t target_tokens, uint32_t *rows_out,
                               qw_dit_segment *segs, uint32_t *text_lens,
                               uint32_t *text_src, char *error,
                               size_t error_size) {
    uint32_t text_idx = 0, img_idx = 0;
    size_t pos = 0;
    while (pos < seq) {
        if (img_idx < n_images && img_mask[pos]) {
            size_t start = pos;
            while (pos < seq && img_mask[pos]) pos++;
            if (pos - start !=
                (size_t)(gh16[img_idx] / 2) * (gw16[img_idx] / 2)) {
                snprintf(error, error_size,
                         "ti2i: mask layout mismatch (image %u)", img_idx);
                return 0;
            }
            img_idx++;
        } else {
            size_t start = pos;
            while (pos < seq && !img_mask[pos]) pos++;
            /* Defensive bound: the template guarantees exactly
             * n_images+1 text runs, but the mask is produced upstream —
             * write nothing beyond the caller's array. */
            if (text_idx > n_images) {
                snprintf(error, error_size,
                         "ti2i: mask text-run overflow");
                return 0;
            }
            text_lens[text_idx] = (uint32_t)(pos - start);
            text_src[text_idx] = (uint32_t)start;
            text_idx++;
        }
    }
    if (text_idx != n_images + 1 || img_idx != n_images) {
        snprintf(error, error_size, "ti2i: mask segment count mismatch");
        return 0;
    }
    uint32_t packed = 0;
    segs[0].rows = text_lens[0];
    segs[0].is_image = 0;
    segs[0].src_offset = text_src[0];
    for (uint32_t i = 0; i < n_images; i++) {
        segs[2 * i + 1].rows = cond_arr[i];
        segs[2 * i + 1].is_image = 1;
        segs[2 * i + 1].src_offset = packed;
        packed += cond_arr[i];
        segs[2 * i + 2].rows = text_lens[i + 1];
        segs[2 * i + 2].is_image = 0;
        segs[2 * i + 2].src_offset = text_src[i + 1];
    }
    segs[2 * n_images + 1].rows = target_tokens;
    segs[2 * n_images + 1].is_image = 1;
    segs[2 * n_images + 1].src_offset = packed;
    uint32_t text_total = 0;
    for (uint32_t k = 0; k <= n_images; k++) text_total += text_lens[k];
    *rows_out = packed + text_total + target_tokens;
    return 1;
}

static uint16_t pipe_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

int qw_pipe_generate_ti2i(qw_dit *dit, qw_vae *vae, struct qw_text *text,
                          struct qw_vision *vision, h3_gpu *gpu,
                          const char *prompt,
                          const uint8_t *const *rgba_images,
                          const uint32_t *in_ws, const uint32_t *in_hs,
                          uint32_t n_images, const qw_pipe_params *params,
                          const double *turbo_sigmas, uint32_t turbo_count,
                          const qw_sched_cfg *sched, uint8_t **rgba_out,
                          char *error, size_t error_size) {
    /* Mirror qw_pipe_generate's validation: downstream derives latent
     * rows as width/16 * height/16 and reads back decode output at
     * params width×height, so a non-32-multiple size would make the
     * serve caller compute has_alpha/write PNG over a smaller buffer
     * (heap over-read) instead of failing here. */
    if (!params || !params->width || !params->height ||
        params->width % 32 || params->height % 32) {
        snprintf(error, error_size,
                 "ti2i: size must be non-zero multiples of 32");
        return 0;
    }
    /* CFG 语义对齐官方 QwenImage21Pipeline：cfg > 1 且提供 negative
     * 才启用；cfg > 1 而无 negative 降回 1（官方同款告警继续）。 */
    int guided = params->cfg_scale > 1.0f && params->negative &&
                 *params->negative;
    pipe_mark_gpu = gpu; /* stage marks report live/footprint */
    if (params->cfg_scale > 1.0f && !guided)
        fprintf(stderr,
                "ti2i: cfg %.2f without a negative prompt stays at 1.0\n",
                params->cfg_scale);
    pipe_timing_t0 = pipe_now();
    pipe_mark("begin");
    if (!n_images || n_images > QW_TI2I_MAX_IMAGES) {
        snprintf(error, error_size, "ti2i: 1..%d reference images",
                 QW_TI2I_MAX_IMAGES);
        return 0;
    }
    uint32_t output_res = params->output_res ? params->output_res : 1024;

    /* 每图 imgproc 尺寸（--res 面积、32 对齐、比例随参考图） */
    uint32_t *iw_arr = malloc(4u * n_images);
    uint32_t *ih_arr = malloc(4u * n_images);
    uint32_t *gh16_arr = malloc(4u * n_images);
    uint32_t *gw16_arr = malloc(4u * n_images);
    uint32_t *cond_arr = malloc(4u * n_images);
    int ok = iw_arr && ih_arr && gh16_arr && gw16_arr && cond_arr;
    uint32_t slots_total = 0, cond_total = 0;
    size_t patch_total = 0;
    for (uint32_t i = 0; ok && i < n_images; i++) {
        int iw = 0, ih = 0;
        qw_imgproc_output_size((int)in_ws[i], (int)in_hs[i], (int)output_res,
                               &iw, &ih);
        if (iw % 32 || ih % 32) {
            snprintf(error, error_size,
                     "ti2i: image %u encoded size %dx%d not 32-aligned", i, iw,
                     ih);
            ok = 0;
            break;
        }
        iw_arr[i] = (uint32_t)iw;
        ih_arr[i] = (uint32_t)ih;
        gh16_arr[i] = (uint32_t)ih / 16;
        gw16_arr[i] = (uint32_t)iw / 16;
        cond_arr[i] = gh16_arr[i] * gw16_arr[i];
        slots_total += (gh16_arr[i] / 2) * (gw16_arr[i] / 2);
        cond_total += cond_arr[i];
        patch_total += (size_t)(gh16_arr[i] / 2) * (gw16_arr[i] / 2) * 4;
    }
    uint32_t target_tokens = params->width / 16 * (params->height / 16);
    if (!ok) {
        free(iw_arr);
        free(ih_arr);
        free(gh16_arr);
        free(gw16_arr);
        free(cond_arr);
        return 0;
    }

    /* 1. 逐图 resize + 白底合成 + patchify（pv 拼接）+ VAE encode
     * （NHWC [H,W,4]，conv MM 内核的 channels-last 语义），cond packed
     * 拼接为 [img0, img1, ...]（segs 的 src_offset 语义）。 */
    uint8_t **up_rgba_arr = calloc(n_images, sizeof(uint8_t *));
    uint8_t **rgb_arr = calloc(n_images, sizeof(uint8_t *));
    float *pv = malloc(sizeof(float) * patch_total * 1536);
    uint16_t *cond_host = calloc((size_t)cond_total * 64, sizeof(uint16_t));
    ok = ok && up_rgba_arr && rgb_arr && pv && cond_host;
    if (ok && !h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) {
        snprintf(error, error_size, "ti2i: vae alloc/begin");
        ok = 0;
    }
    size_t patch_off = 0, cond_off = 0;
    for (uint32_t i = 0; ok && i < n_images; i++) {
        uint32_t iw = iw_arr[i], ih = ih_arr[i];
        /* 逐图 encode 的 begin/submit 会改命令状态：每张图前恢复。 */
        if (!h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) {
            snprintf(error, error_size, "ti2i: image %u begin failed", i);
            ok = 0;
            break;
        }
        up_rgba_arr[i] = malloc((size_t)iw * ih * 4);
        rgb_arr[i] = malloc((size_t)iw * ih * 3);
        float *vae_in = malloc(sizeof(float) * 4 * (size_t)iw * ih);
        h3_gpu_tensor *vae_t = NULL, *cond_t = NULL;
        ok = ok && up_rgba_arr[i] && rgb_arr[i] && vae_in;
        if (ok)
            ok = qw_imgproc_resize_rgba_u8(rgba_images[i], (int)in_ws[i],
                                           (int)in_hs[i], up_rgba_arr[i],
                                           (int)iw, (int)ih);
        if (ok)
            qw_imgproc_white_composite(up_rgba_arr[i], iw * ih, rgb_arr[i]);
        if (ok)
            qw_imgproc_patchify(rgb_arr[i], (int)ih / 16, (int)iw / 16,
                                pv + patch_off * 1536);
        patch_off += (size_t)(ih / 16 / 2) * (iw / 16 / 2) * 4;
        if (ok) qw_imgproc_vae_input(up_rgba_arr[i], iw, ih, vae_in);
        /* Last use of the upscaled image; release before the long
         * vision/text/denoise phases (iw*ih*4 per reference). */
        free(up_rgba_arr[i]);
        up_rgba_arr[i] = NULL;
        if (ok) {
            vae_t = h3_gpu_tensor_new_f32(gpu, 4 * (size_t)iw * ih);
            cond_t = h3_gpu_tensor_new_bf16(gpu, (size_t)cond_arr[i] * 64);
            ok = vae_t && cond_t;
        }
        ok = ok && h3_gpu_tensor_write_f32(vae_t, vae_in,
                                           4 * (size_t)iw * ih);
        ok = ok && qw_vae_encode(vae, vae_t, ih, iw, cond_t, error,
                                 error_size);
        ok = ok && h3_gpu_submit(gpu);
        ok = ok && h3_gpu_tensor_read_bf16(cond_t,
                                           cond_host + cond_off * 64,
                                           (size_t)cond_arr[i] * 64);
        h3_gpu_tensor_free(vae_t);
        h3_gpu_tensor_free(cond_t);
        free(vae_in);
        cond_off += cond_arr[i];
        pipe_mark("img N imgproc+vae");
        if (!ok) {
            char detail[256];
            snprintf(detail, sizeof(detail), "%s",
                     error[0] ? error : "unknown");
            snprintf(error, error_size, "ti2i: image %u encode failed: %s", i,
                     detail);
        }
    }
    if (rgb_arr) {
        for (uint32_t i = 0; i < n_images; i++) free(rgb_arr[i]);
        free(rgb_arr);
    }
    rgb_arr = NULL;
    pipe_mark("section1 imgproc+vae");
    /* 2. 视觉塔（多图一次前向，图间注意力隔离） */
    float *merged = malloc(sizeof(float) * (size_t)slots_total * 4096);
    float *deepstack =
        malloc(sizeof(float) * 3 * (size_t)slots_total * 4096);
    uint32_t *grid = malloc(12u * n_images);
    ok = ok && merged && deepstack && grid;
    for (uint32_t i = 0; ok && i < n_images; i++) {
        grid[i * 3] = 1;
        grid[i * 3 + 1] = ih_arr[i] / 16;
        grid[i * 3 + 2] = iw_arr[i] / 16;
    }
    if (ok)
        ok = qw_vision_forward(vision, pv, grid, n_images, merged, deepstack,
                               error, error_size);
    free(pv); pv = NULL;
    free(grid); grid = NULL;
    uint16_t *merged_bf =
        malloc(sizeof(uint16_t) * (size_t)slots_total * 4096);
    uint16_t *ds_bf =
        malloc(sizeof(uint16_t) * 3 * (size_t)slots_total * 4096);
    ok = ok && merged_bf && ds_bf;
    for (size_t i = 0; ok && i < (size_t)slots_total * 4096; i++)
        merged_bf[i] = pipe_bf16(merged[i]);
    for (size_t i = 0; ok && i < 3 * (size_t)slots_total * 4096; i++)
        ds_bf[i] = pipe_bf16(deepstack[i]);
    free(merged); merged = NULL;
    free(deepstack); deepstack = NULL;
    pipe_mark("vision(+bf16)");

    /* 3. 文本塔 ti2i 组装（多图 grid 数组，merged/ds 按图序拼接） */
    /* 段容量按上界动态分配：byte-level BPE 的 token 数不超过模板字节
     * 数，因此 slots + 两个 prompt 的字节长度 + 图组标记/模板包装余量
     * 必然覆盖 drop 后的行数；encode_ti2i 内部还有硬校验兜底。 */
    size_t prompt_bytes = strlen(prompt);
    if (guided && params->negative &&
        strlen(params->negative) > prompt_bytes)
        prompt_bytes = strlen(params->negative);
    uint32_t text_cap = slots_total + (uint32_t)prompt_bytes +
                        16u * n_images + 128u;
    h3_gpu_tensor *hidden =
        h3_gpu_tensor_new_bf16(gpu, (size_t)text_cap * 4096);
    uint8_t *img_mask = malloc(text_cap);
    uint32_t *grid_hw = malloc(8u * n_images);
    ok = ok && hidden && img_mask && grid_hw;
    size_t seq = 0;
    for (uint32_t i = 0; ok && i < n_images; i++) {
        grid_hw[i * 2] = gh16_arr[i] / 2;
        grid_hw[i * 2 + 1] = gw16_arr[i] / 2;
    }
    if (ok)
        ok = qw_text_encode_ti2i(text, prompt, n_images, grid_hw, merged_bf,
                                 ds_bf, hidden, img_mask, text_cap, &seq,
                                 error, error_size);
    /* CFG 负向分支：官方 encode_prompt(image=同款参考图, prompt=negative)
     * ——同 vision 注入、同占位符模板，仅用户文本不同（长度可不同）。 */
    h3_gpu_tensor *neg_hidden = NULL;
    uint8_t *neg_img_mask = NULL;
    size_t neg_seq = 0;
    if (ok && guided) {
        neg_hidden = h3_gpu_tensor_new_bf16(gpu, (size_t)text_cap * 4096);
        neg_img_mask = malloc(text_cap);
        ok = neg_hidden && neg_img_mask &&
             qw_text_encode_ti2i(text, params->negative, n_images, grid_hw,
                                 merged_bf, ds_bf, neg_hidden, neg_img_mask,
                                 text_cap, &neg_seq, error, error_size);
    }
    /* encode_core 的命令链在函数内不收尾：提交后才能读 hidden。
     * （序列容量由 encode_ti2i 内部先行校验，此处无需再查。） */
    pipe_mark("text returned");
    ok = ok && h3_gpu_submit(gpu);
    pipe_mark("text final submit");
    free(grid_hw); grid_hw = NULL;
    free(merged_bf); merged_bf = NULL;
    free(ds_bf); ds_bf = NULL;
    pipe_mark("text(+submit)");

    /* 4. 段布局（img_mask → 段）：[text0][img0][text1][img1]...[textN]，
     * joint 行序 = segs scatter 序；图像数据 packed 在前（segs 的
     * src_offset 语义），target 紧随各图之后。 */
    uint32_t n_seg = 2 * n_images + 2;
    qw_dit_segment *segs = malloc(sizeof(qw_dit_segment) * n_seg);
    uint32_t *text_lens = malloc(4u * (n_images + 1));
    uint32_t *text_src = malloc(4u * (n_images + 1));
    ok = ok && segs && text_lens && text_src;
    uint32_t rows = 0;
    if (ok)
        ok = ti2i_build_segments(img_mask, seq, gh16_arr, gw16_arr, cond_arr,
                                 n_images, target_tokens, &rows, segs,
                                 text_lens, text_src, error, error_size);
    qw_dit_segment *neg_segs = NULL;
    uint32_t *neg_text_lens = NULL, *neg_text_src = NULL;
    uint32_t neg_rows = 0;
    if (ok && guided) {
        neg_segs = malloc(sizeof(qw_dit_segment) * n_seg);
        neg_text_lens = malloc(4u * (n_images + 1));
        neg_text_src = malloc(4u * (n_images + 1));
        ok = neg_segs && neg_text_lens && neg_text_src &&
             ti2i_build_segments(neg_img_mask, neg_seq, gh16_arr, gw16_arr,
                                 cond_arr, n_images, target_tokens, &neg_rows,
                                 neg_segs, neg_text_lens, neg_text_src, error,
                                 error_size);
    }
    if (!ok) {
        if (up_rgba_arr) {
            for (uint32_t i = 0; i < n_images; i++) free(up_rgba_arr[i]);
            free(up_rgba_arr);
        }
        free(pv);
        free(cond_host);
        free(img_mask);
        free(neg_img_mask);
        free(segs);
        free(neg_segs);
        free(text_lens);
        free(text_src);
        free(neg_text_lens);
        free(neg_text_src);
        h3_gpu_tensor_free(hidden);
        h3_gpu_tensor_free(neg_hidden);
        free(iw_arr);
        free(ih_arr);
        free(gh16_arr);
        free(gw16_arr);
        free(cond_arr);
        return 0;
    }

    /* 6. 目标噪声 latents */
    /* joint 总行数在段布局后按 latent 粒度重算（见下） */
    float *lat = malloc(sizeof(float) * (size_t)target_tokens * 64);
    ok = ok && lat;
    qw_pipe_seed(params->seed);
    for (uint32_t i = 0; ok && i < target_tokens * 64; i++)
        lat[i] = qw_pipe_randn();

    /* 7. rope 表（N+1 块：每图一块 + target 块，文本段交错） */
    uint32_t *block_h = malloc(4u * (n_images + 1));
    uint32_t *block_w = malloc(4u * (n_images + 1));
    uint32_t *text_before = malloc(4u * (n_images + 1));
    ok = ok && block_h && block_w && text_before;
    for (uint32_t i = 0; ok && i < n_images; i++) {
        block_h[i] = gh16_arr[i];
        block_w[i] = gw16_arr[i];
        text_before[i] = text_lens[i];
    }
    if (ok) {
        block_h[n_images] = params->height / 16;
        block_w[n_images] = params->width / 16;
        text_before[n_images] = text_lens[n_images];
    }
    h3_gpu_tensor *rope_cos = NULL, *rope_sin = NULL;
    ok = ok && qw_pipe_rope_tables_blocks(gpu, block_h, block_w, text_before,
                                          n_images + 1, &rope_cos, &rope_sin,
                                          error, error_size);
    h3_gpu_tensor *neg_rope_cos = NULL, *neg_rope_sin = NULL;
    if (ok && guided) {
        /* neg 文本段长度不同 → 独立的 rope 表。 */
        for (uint32_t i = 0; i < n_images; i++)
            text_before[i] = neg_text_lens[i];
        text_before[n_images] = neg_text_lens[n_images];
        ok = qw_pipe_rope_tables_blocks(gpu, block_h, block_w, text_before,
                                        n_images + 1, &neg_rope_cos,
                                        &neg_rope_sin, error, error_size);
    }
    free(block_h); block_h = NULL;
    free(block_w); block_w = NULL;
    free(text_before); text_before = NULL;
    h3_gpu_tensor *joint =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * 64);
    h3_gpu_tensor *pred =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * 64);
    h3_gpu_tensor *neg_joint = guided
        ? h3_gpu_tensor_new_bf16(gpu, (size_t)neg_rows * 64) : NULL;
    h3_gpu_tensor *neg_pred = guided
        ? h3_gpu_tensor_new_bf16(gpu, (size_t)neg_rows * 64) : NULL;
    ok = ok && joint && pred && (!guided || (neg_joint && neg_pred));

    /* 8. schedule + 去噪循环 */
    double *sigmas = malloc(sizeof(double) * ((size_t)params->steps + 1));
    ok = ok && sigmas &&
         qw_pipe_schedule(params->steps, target_tokens, turbo_sigmas,
                          turbo_count, sched, sigmas, error, error_size);
    /* calloc：文本段行不写入但会被 img_in 全量投影，未初始化垃圾
     * （巨大值）经投影进入 target 段后可在 attention 里传染成 NaN。 */
    uint16_t *joint_host =
        calloc((size_t)(guided && neg_rows > rows ? neg_rows : rows) * 64,
               sizeof(uint16_t));
    ok = ok && joint_host;
    /* Per-step prediction readbacks, allocated once for the whole loop
     * (the malloc/free used to churn every step). */
    uint16_t *pred_host = malloc(sizeof(uint16_t) * (size_t)rows * 64);
    uint16_t *neg_pred_host =
        guided ? malloc(sizeof(uint16_t) * (size_t)neg_rows * 64) : NULL;
    ok = ok && pred_host && (!guided || neg_pred_host);
    pipe_mark("layout(rope+joint)");
    if (ok) {
        memcpy(joint_host, cond_host,
               sizeof(uint16_t) * (size_t)cond_total * 64);
    }
    for (uint32_t step = 0; ok && step < params->steps; step++) {
        double sigma = sigmas[step], next = sigmas[step + 1];
        /* packed latent 布局：各图 cond 在前、target 紧随（segs 的
         * src_offset 语义）。文本段的行保持零（由 scatter 覆盖）。 */
        for (uint32_t i = 0; i < target_tokens * 64; i++) {
            joint_host[(cond_total) * 64 + i] = pipe_bf16((float)lat[i]);
        }
        if (!h3_gpu_tensor_write_bf16(joint, joint_host,
                                      (size_t)rows * 64) ||
            (guided && !h3_gpu_tensor_write_bf16(neg_joint, joint_host,
                                                 (size_t)neg_rows * 64))) {
            ok = 0;
            break;
        }
        if (!h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) {
            snprintf(error, error_size, "ti2i: step begin failed");
            ok = 0;
            break;
        }
        if (!qw_dit_forward_ti2i(dit, hidden, joint, segs, n_seg, 0,
                                 rope_cos, rope_sin, (float)sigma, pred,
                                 error, error_size)) {
            ok = 0;
            break;
        }
        if (!h3_gpu_submit(gpu)) {
            ok = 0;
            break;
        }
        if (!h3_gpu_tensor_read_bf16(pred, pred_host,
                                     (size_t)rows * 64)) {
            ok = 0;
            break;
        }
        /* CFG 负向分支：同 target latents、独立文本/K-V 缓存（slot 1）。 */
        if (guided) {
            if (!qw_dit_forward_ti2i(dit, neg_hidden, neg_joint, neg_segs,
                                     n_seg, 1, neg_rope_cos, neg_rope_sin,
                                     (float)sigma, neg_pred, error,
                                     error_size) ||
                !h3_gpu_submit(gpu) ||
                !h3_gpu_tensor_read_bf16(neg_pred, neg_pred_host,
                                         (size_t)neg_rows * 64)) {
                ok = 0;
                break;
            }
        }
        size_t base = (size_t)rows - target_tokens;
        size_t neg_base =
            guided ? (size_t)neg_rows - target_tokens : 0;
        for (uint32_t i = 0; i < target_tokens * 64; i++) {
            uint32_t b = pred_host[base * 64 + i];
            uint32_t wide = (uint32_t)b << 16;
            float f;
            memcpy(&f, &wide, 4);
            if (guided) {
                uint32_t nb = neg_pred_host[neg_base * 64 + i];
                uint32_t nwide = (uint32_t)nb << 16;
                float nf;
                memcpy(&nf, &nwide, 4);
                /* 官方组合：neg + cfg × (cond − neg)。 */
                f = nf + params->cfg_scale * (f - nf);
            }
            lat[i] += (float)(next - sigma) * f;
        }
        if (pipe_timing_on) {
            char label[32];
            snprintf(label, sizeof(label), "step %u", step);
            pipe_mark(label);
        }
        pipe_progress(step + 1, params->steps);
    }
    free(joint_host); joint_host = NULL;
    free(pred_host); pred_host = NULL;
    free(neg_pred_host); neg_pred_host = NULL;
    free(cond_host); cond_host = NULL;
    free(sigmas); sigmas = NULL;
    h3_gpu_tensor_free(joint);
    h3_gpu_tensor_free(pred);
    h3_gpu_tensor_free(neg_joint);
    h3_gpu_tensor_free(neg_pred);
    h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(neg_rope_cos);
    h3_gpu_tensor_free(neg_rope_sin);
    h3_gpu_tensor_free(neg_hidden);
    free(neg_img_mask); neg_img_mask = NULL;
    free(neg_segs); neg_segs = NULL;
    free(neg_text_lens); neg_text_lens = NULL;
    free(neg_text_src); neg_text_src = NULL;

    /* 9. decode（循环尾部已 submit，恢复命令状态） */
    /* cond KV cache 是单请求结构：缓存的 cond 行编码的是本次请求的
     * 提示词/参考图/权重。布局相同的下一个请求若复用它，会静默沿用上
     * 一个请求的条件（reuse 检查只匹配布局不匹配内容）。请求结束即释放，
     * 也让 ~12GB 回到 Metal 池，供下一个请求同尺寸重分配复用（footprint
     * 不再随 LoRA/布局变更棘轮上涨）。 */
    qw_dit_ti2i_cache_clear(dit);
    if (ok && !h3_gpu_begin(gpu) && !h3_gpu_continue(gpu)) {
        snprintf(error, error_size, "ti2i: decode begin failed");
        ok = 0;
    }
    if (ok)
        ok = qw_pipe_decode(vae, gpu, lat, params->height / 16,
                            params->width / 16, rgba_out, error, error_size);
    free(lat); lat = NULL;
    if (up_rgba_arr) {
        for (uint32_t i = 0; i < n_images; i++) free(up_rgba_arr[i]);
        free(up_rgba_arr);
    }
    up_rgba_arr = NULL;
    free(cond_host);
    free(img_mask); img_mask = NULL;
    free(segs); segs = NULL;
    free(text_lens);
    free(text_src);
    h3_gpu_tensor_free(hidden);
    free(iw_arr);
    free(ih_arr);
    free(gh16_arr);
    free(gw16_arr);
    free(cond_arr);
    return ok;
}
