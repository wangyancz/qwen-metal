#ifndef QWEN_HOST_H
#define QWEN_HOST_H

#include <stddef.h>
#include <stdint.h>

/* Deterministic host-side math for Qwen-Image-2.1: the released
 * FlowMatchEulerDiscreteScheduler schedule, Euler integration, the DiT
 * three-axis RoPE tables, the latent geometry, and the native RNG.
 *
 * Numeric conventions mirror the released diffusers implementation:
 * the schedule is computed element-wise in float32 exactly like numpy,
 * calculate_shift in float64, and RoPE angles in float32. */

/* scheduler/scheduler_config.json */
#define QWEN_TRAIN_TIMESTEPS 1000.0
#define QWEN_SCHED_BASE_SEQ_LEN 256
#define QWEN_SCHED_MAX_SEQ_LEN 8192
#define QWEN_SCHED_BASE_SHIFT 0.5
#define QWEN_SCHED_MAX_SHIFT 0.9
#define QWEN_SCHED_SHIFT_TERMINAL 0.02
#define QWEN_MAX_STEPS 1000

/* transformer/config.json + vae/config.json geometry */
#define QWEN_VAE_SPATIAL_RATIO 16 /* one latent token covers 16x16 pixels */
#define QWEN_Z_DIM 64
#define QWEN_HIDDEN 4096
#define QWEN_HEADS 32
#define QWEN_HEAD_DIM 128
#define QWEN_HEADS_TEXT 32
#define QWEN_HEAD_DIM_TEXT 128
#define QWEN_KV_HEADS_TEXT 8
#define QWEN_TEXT_LAYERS 36
#define QWEN_TEXT_HIDDEN 4096
#define QWEN_TEXT_MLP 12288
#define QWEN_DIT_LAYERS 32
#define QWEN_MLP_RATIO 3
#define QWEN_MLP_HIDDEN (QWEN_HIDDEN * QWEN_MLP_RATIO)
#define QWEN_EPS 1e-6f

/* QwenImage21Rope table geometry. */
#define QWEN_ROPE_AXES 3
#define QWEN_ROPE_THETA 10000.0
#define QWEN_ROPE_POS_ROWS 8192
#define QWEN_ROPE_NEG_ROWS 1024
#define QWEN_ROPE_ROWS (QWEN_ROPE_POS_ROWS + QWEN_ROPE_NEG_ROWS)

/* Latent token grid for an output canvas. Width and height must be
 * multiples of 16. */
int qw_latent_grid(int width, int height, int *latent_w, int *latent_h);

/* Resolution-dependent shift exponent (pipeline calculate_shift): linear
 * interpolation between (256, 0.5) and (8192, 0.9) over the target-image
 * sequence length. */
double qw_calculate_shift(int image_seq_len);

typedef struct {
    int steps;
    float sigma[QWEN_MAX_STEPS + 1];
} qw_sigma_schedule;

/* Port of FlowMatchEulerDiscreteScheduler.set_timesteps with the pipeline
 * arguments: sigmas = linspace(1, 1/steps), exponential dynamic shifting
 * with mu from qw_calculate_shift, terminal stretch to 0.02, appended zero.
 * sigma[steps] is the terminal zero entry. */
int qw_schedule_build(int steps, int image_seq_len, qw_sigma_schedule *schedule);

/* Explicit sigma list (turbo sample_sigmas and friends). Copies steps
 * values and appends the terminal zero; caller keeps ownership. */
int qw_schedule_from_sigmas(const float *sigmas, int steps,
                            qw_sigma_schedule *schedule);

/* sample += (sigma - sigma_next) * velocity, the scheduler.step Euler
 * update on the velocity parameterization. */
void qw_euler_velocity_step(float *sample, const float *velocity,
                            size_t count, float sigma, float sigma_next);

typedef struct {
    int axes;
    int axis_dim[QWEN_ROPE_AXES];
    /* Per-axis cos/sin tables, axis-major: axis r owns
     * [r * rows * pairs, ...], pairs = axis_dim[r] / 2, row-major. */
    float *cos;
    float *sin;
} qw_rope_tables;

/* Build the released tables: rows 0..8191 carry coordinates 0..8191 and
 * rows 8192..9215 carry coordinates -1..-1024 (image blocks center their
 * height/width grids on zero). */
int qw_rope_build(const int *axis_dim, int axes, qw_rope_tables *tables);
void qw_rope_free(qw_rope_tables *tables);

/* Table row for a RoPE coordinate: nonnegative coordinates map identity,
 * coordinate -k maps to QWEN_ROPE_POS_ROWS + QWEN_ROPE_NEG_ROWS - k (the
 * released negative rows run -1024..-1 across rows 8192..9215). Returns -1
 * when the coordinate falls outside the released tables. */
int qw_rope_row(int coordinate);

typedef struct {
    uint64_t state;
    uint64_t increment;
    float spare;
    int has_spare;
} qw_rng;

void qw_rng_seed(qw_rng *rng, uint64_t seed);
uint32_t qw_rng_u32(qw_rng *rng);
float qw_rng_normal(qw_rng *rng);
void qw_rng_fill_normal(qw_rng *rng, float *values, size_t count);

#endif
