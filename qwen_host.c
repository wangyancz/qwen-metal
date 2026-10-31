#include "qwen_host.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int qw_latent_grid(int width, int height, int *latent_w, int *latent_h) {
    if (width < QWEN_VAE_SPATIAL_RATIO || height < QWEN_VAE_SPATIAL_RATIO ||
        width % QWEN_VAE_SPATIAL_RATIO != 0 ||
        height % QWEN_VAE_SPATIAL_RATIO != 0 || !latent_w || !latent_h) {
        return 0;
    }
    *latent_w = width / QWEN_VAE_SPATIAL_RATIO;
    *latent_h = height / QWEN_VAE_SPATIAL_RATIO;
    return 1;
}

double qw_calculate_shift(int image_seq_len) {
    const double base_seq = QWEN_SCHED_BASE_SEQ_LEN;
    const double max_seq = QWEN_SCHED_MAX_SEQ_LEN;
    const double base_shift = QWEN_SCHED_BASE_SHIFT;
    const double max_shift = QWEN_SCHED_MAX_SHIFT;
    double m = (max_shift - base_shift) / (max_seq - base_seq);
    double b = base_shift - m * base_seq;
    return (double)image_seq_len * m + b;
}

/* Element-wise float32 port of
 *   sigmas = np.linspace(1, 1/steps, steps).astype(np.float32)
 *   sigmas = exp(mu) / (exp(mu) + (1/sigma - 1))      # sigma exponent = 1
 *   one_minus = 1 - sigmas
 *   sigmas = 1 - one_minus / (one_minus[-1] / (1 - 0.02))
 * The scalar exponent mu is a python float, so numpy applies it as a weak
 * float32 scalar; every intermediate stays float32 exactly as released. */
static float qw_shifted_sigma(float base, float exp_mu) {
    float inverse = 1.0f / base;
    float shifted_denom = exp_mu + (inverse - 1.0f);
    return exp_mu / shifted_denom;
}

int qw_schedule_build(int steps, int image_seq_len, qw_sigma_schedule *schedule) {
    if (!schedule || steps < 1 || steps > QWEN_MAX_STEPS || image_seq_len < 1) {
        return 0;
    }
    float exp_mu = (float)exp(qw_calculate_shift(image_seq_len));

    /* np.linspace(1.0, 1/steps, steps) in float64, then cast to float32.
     * linspace forces the final element to the exact stop value. */
    double stop = 1.0 / (double)steps;
    double step_delta = (stop - 1.0) / (double)(steps - 1);
    for (int index = 0; index < steps; index++) {
        double value = 1.0 + (double)index * step_delta;
        if (index == steps - 1) value = stop;
        schedule->sigma[index] = qw_shifted_sigma((float)value, exp_mu);
    }

    /* stretch_shift_to_terminal. */
    float one_minus_last = 1.0f - schedule->sigma[steps - 1];
    float scale = one_minus_last / (1.0f - (float)QWEN_SCHED_SHIFT_TERMINAL);
    for (int index = 0; index < steps; index++) {
        float one_minus = 1.0f - schedule->sigma[index];
        schedule->sigma[index] = 1.0f - one_minus / scale;
    }
    schedule->steps = steps;
    schedule->sigma[steps] = 0.0f;
    return 1;
}

int qw_schedule_from_sigmas(const float *sigmas, int steps,
                            qw_sigma_schedule *schedule) {
    if (!sigmas || !schedule || steps < 1 || steps > QWEN_MAX_STEPS) return 0;
    schedule->steps = steps;
    memcpy(schedule->sigma, sigmas, sizeof(float) * (size_t)steps);
    schedule->sigma[steps] = 0.0f;
    return 1;
}

void qw_euler_velocity_step(float *sample, const float *velocity,
                            size_t count, float sigma, float sigma_next) {
    float delta = sigma - sigma_next;
    for (size_t index = 0; index < count; index++) {
        sample[index] += delta * velocity[index];
    }
}

int qw_rope_row(int coordinate) {
    if (coordinate >= 0 && coordinate < QWEN_ROPE_POS_ROWS) return coordinate;
    if (coordinate < 0 && -coordinate <= QWEN_ROPE_NEG_ROWS) {
        return QWEN_ROPE_POS_ROWS + QWEN_ROPE_NEG_ROWS + coordinate;
    }
    return -1;
}

int qw_rope_build(const int *axis_dim, int axes, qw_rope_tables *tables) {
    if (!axis_dim || !tables || axes < 1 || axes > QWEN_ROPE_AXES) return 0;
    size_t total = 0;
    for (int axis = 0; axis < axes; axis++) {
        if (axis_dim[axis] < 2 || axis_dim[axis] % 2 != 0) return 0;
        total += (size_t)QWEN_ROPE_ROWS * (size_t)(axis_dim[axis] / 2);
    }
    tables->axes = axes;
    tables->cos = malloc(sizeof(float) * total);
    tables->sin = malloc(sizeof(float) * total);
    if (!tables->cos || !tables->sin) {
        free(tables->cos);
        free(tables->sin);
        tables->cos = NULL;
        tables->sin = NULL;
        return 0;
    }

    /* Released construction: coordinates 0..8191 then -1024..-1 (the
     * flip()*-1-1 order), with the pair frequency theta^(-2*pair/dim) and
     * float32 angles. Axis blocks are laid out back to back. */
    size_t offset = 0;
    for (int axis = 0; axis < axes; axis++) {
        int pairs = axis_dim[axis] / 2;
        tables->axis_dim[axis] = axis_dim[axis];
        float *cos_rows = tables->cos + offset;
        float *sin_rows = tables->sin + offset;
        for (int pair = 0; pair < pairs; pair++) {
            double exponent = -2.0 * (double)pair / (double)axis_dim[axis];
            float frequency = (float)pow(QWEN_ROPE_THETA, exponent);
            for (int row = 0; row < QWEN_ROPE_ROWS; row++) {
                int coordinate = row < QWEN_ROPE_POS_ROWS
                    ? row
                    : (int)(row - QWEN_ROPE_POS_ROWS) - QWEN_ROPE_NEG_ROWS;
                float angle = (float)coordinate * frequency;
                cos_rows[(size_t)row * pairs + pair] = cosf(angle);
                sin_rows[(size_t)row * pairs + pair] = sinf(angle);
            }
        }
        offset += (size_t)QWEN_ROPE_ROWS * (size_t)pairs;
    }
    return 1;
}

void qw_rope_free(qw_rope_tables *tables) {
    if (!tables) return;
    free(tables->cos);
    free(tables->sin);
    tables->cos = NULL;
    tables->sin = NULL;
}

uint32_t qw_rng_u32(qw_rng *rng) {
    uint64_t old_state = rng->state;
    rng->state = old_state * UINT64_C(6364136223846793005) + rng->increment;
    uint32_t shifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    uint32_t rotation = (uint32_t)(old_state >> 59u);
    return (shifted >> rotation) | (shifted << ((-(int32_t)rotation) & 31));
}

void qw_rng_seed(qw_rng *rng, uint64_t seed) {
    memset(rng, 0, sizeof(*rng));
    rng->increment = (seed << 1u) | 1u;
    (void)qw_rng_u32(rng);
    rng->state += seed ^ UINT64_C(0x9e3779b97f4a7c15);
    (void)qw_rng_u32(rng);
}

float qw_rng_normal(qw_rng *rng) {
    if (rng->has_spare) {
        rng->has_spare = 0;
        return rng->spare;
    }
    double u1 = ((double)qw_rng_u32(rng) + 1.0) / 4294967297.0;
    double u2 = ((double)qw_rng_u32(rng) + 0.5) / 4294967296.0;
    double radius = sqrt(-2.0 * log(u1));
    double angle = 2.0 * 3.14159265358979323846 * u2;
    rng->spare = (float)(radius * sin(angle));
    rng->has_spare = 1;
    return (float)(radius * cos(angle));
}

void qw_rng_fill_normal(qw_rng *rng, float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        values[index] = qw_rng_normal(rng);
    }
}
