#include "qwen_host.h"

#include "host_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            failures++;                                                     \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                     \
            printf(__VA_ARGS__);                                            \
            printf("\n");                                                   \
        }                                                                   \
    } while (0)

static void test_schedule(void) {
    for (size_t c = 0; c < HOST_REF_CASE_COUNT; c++) {
        const host_ref_case *ref = &host_ref_cases[c];
        qw_sigma_schedule schedule;
        CHECK(qw_schedule_build(ref->steps, ref->seq_len, &schedule),
              "schedule_build steps=%d seq=%d", ref->steps, ref->seq_len);
        CHECK(schedule.steps == ref->steps, "steps mismatch");
        for (int i = 0; i <= ref->steps; i++) {
            float expected = ref->sigmas[i];
            float actual = schedule.sigma[i];
            float diff = fabsf(actual - expected);
            float scale = expected > 1.0f ? expected : 1.0f;
            CHECK(diff <= 2e-6f * scale, "sigma[%d] steps=%d: %.9g vs %.9g", i,
                  ref->steps, actual, expected);
        }
        /* Monotone non-increasing, terminal zero. */
        for (int i = 1; i <= ref->steps; i++) {
            CHECK(schedule.sigma[i] <= schedule.sigma[i - 1], "not monotone at %d", i);
        }
        CHECK(schedule.sigma[ref->steps] == 0.0f, "terminal not zero");
    }
    /* calculate_shift sanity: endpoint values. */
    CHECK(fabs(qw_calculate_shift(QWEN_SCHED_BASE_SEQ_LEN) - QWEN_SCHED_BASE_SHIFT) < 1e-12,
          "mu at base seq");
    CHECK(fabs(qw_calculate_shift(QWEN_SCHED_MAX_SEQ_LEN) - QWEN_SCHED_MAX_SHIFT) < 1e-12,
          "mu at max seq");
    CHECK(qw_schedule_build(0, 1024, &(qw_sigma_schedule){0}) == 0, "reject zero steps");
    CHECK(qw_schedule_build(QWEN_MAX_STEPS + 1, 1024, &(qw_sigma_schedule){0}) == 0,
          "reject too many steps");
}

static void test_rope_tables(void) {
    static const int axes[QWEN_ROPE_AXES] = {16, 56, 56};
    qw_rope_tables tables;
    CHECK(qw_rope_build(axes, QWEN_ROPE_AXES, &tables), "rope build");
    CHECK(tables.axes == 3 && tables.axis_dim[0] == 16 && tables.axis_dim[1] == 56,
          "rope dims");

    const float *cos_base[3] = {
        tables.cos,
        tables.cos + (size_t)QWEN_ROPE_ROWS * 8,
        tables.cos + (size_t)QWEN_ROPE_ROWS * 8 + (size_t)QWEN_ROPE_ROWS * 28,
    };
    const float *sin_base[3] = {
        tables.sin,
        tables.sin + (size_t)QWEN_ROPE_ROWS * 8,
        tables.sin + (size_t)QWEN_ROPE_ROWS * 8 + (size_t)QWEN_ROPE_ROWS * 28,
    };
    for (size_t r = 0; r < HOST_REF_ROPE_COUNT; r++) {
        const host_ref_rope *ref = &host_ref_rope_samples[r];
        int pairs = tables.axis_dim[ref->axis] / 2;
        float actual_cos = cos_base[ref->axis][(size_t)ref->table_row * pairs + ref->pair];
        float actual_sin = sin_base[ref->axis][(size_t)ref->table_row * pairs + ref->pair];
        /* Large coordinates hit float32 argument-reduction differences
         * between libm and torch; structural errors (sign, frequency
         * family, row layout) produce errors far above this bound. */
        CHECK(fabsf(actual_cos - ref->cos) < 5e-3f, "rope cos axis=%d row=%d pair=%d: %.7g vs %.7g",
              ref->axis, ref->table_row, ref->pair, actual_cos, ref->cos);
        CHECK(fabsf(actual_sin - ref->sin) < 5e-3f, "rope sin axis=%d row=%d pair=%d: %.7g vs %.7g",
              ref->axis, ref->table_row, ref->pair, actual_sin, ref->sin);
    }
    /* Row mapping: negative grid coordinates reach the negative rows. */
    CHECK(qw_rope_row(0) == 0, "row 0");
    CHECK(qw_rope_row(8191) == 8191, "row max pos");
    CHECK(qw_rope_row(-1024) == QWEN_ROPE_POS_ROWS, "row -1024");
    CHECK(qw_rope_row(-1) == QWEN_ROPE_ROWS - 1, "row -1");
    CHECK(qw_rope_row(-1025) == -1, "row out of range");
    CHECK(qw_rope_row(8192) == -1, "row out of range pos");
    /* Even symmetry of the underlying complex exponential:
     * cos(-k) == cos(k) and sin(-k) == -sin(k). */
    int pairs = 8;
    for (int k = 1; k <= 8; k++) {
        size_t positive = (size_t)qw_rope_row(k) * pairs;
        size_t negative = (size_t)qw_rope_row(-k) * pairs;
        for (int p = 0; p < pairs; p++) {
            CHECK(fabsf(cos_base[0][positive + p] - cos_base[0][negative + p]) < 1e-6f,
                  "cos even symmetry k=%d p=%d", k, p);
            CHECK(fabsf(sin_base[0][positive + p] + sin_base[0][negative + p]) < 1e-6f,
                  "sin odd symmetry k=%d p=%d", k, p);
        }
    }
    qw_rope_free(&tables);
    CHECK(tables.cos == NULL && tables.sin == NULL, "rope free");
}

static void test_euler_step(void) {
    enum { COUNT = 257 };
    float sample[COUNT], velocity[COUNT];
    qw_rng rng;
    qw_rng_seed(&rng, 7);
    qw_rng_fill_normal(&rng, sample, COUNT);
    qw_rng_fill_normal(&rng, velocity, COUNT);
    float reference[COUNT];
    memcpy(reference, sample, sizeof(sample));
    float sigma = 0.37f, sigma_next = 0.11f;
    qw_euler_velocity_step(sample, velocity, COUNT, sigma, sigma_next);
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(sample[i] == (float)((double)reference[i] + (double)(sigma - sigma_next) * velocity[i]),
              "euler element %zu", i);
    }
}

static void test_rng(void) {
    qw_rng a, b;
    qw_rng_seed(&a, 42);
    qw_rng_seed(&b, 42);
    for (int i = 0; i < 64; i++) {
        CHECK(qw_rng_u32(&a) == qw_rng_u32(&b), "rng determinism %d", i);
    }
    qw_rng_seed(&a, 43);
    CHECK(qw_rng_u32(&a) != qw_rng_u32(&b), "rng seed variation");

    qw_rng_seed(&a, 1);
    double sum = 0.0;
    enum { N = 4096 };
    for (int i = 0; i < N; i++) sum += qw_rng_normal(&a);
    double mean = sum / (double)N;
    CHECK(fabs(mean) < 0.08, "normal mean %.4f", mean);
}

static void test_latent_grid(void) {
    int lw = 0, lh = 0;
    CHECK(qw_latent_grid(1024, 1024, &lw, &lh) && lw == 64 && lh == 64, "1024 grid");
    CHECK(qw_latent_grid(832, 1216, &lw, &lh) && lw == 52 && lh == 76, "832x1216 grid");
    CHECK(qw_latent_grid(1344, 768, &lw, &lh) && lw == 84 && lh == 48, "1344x768 grid");
    CHECK(!qw_latent_grid(100, 100, &lw, &lh), "reject non-multiple");
    CHECK(!qw_latent_grid(0, 0, &lw, &lh), "reject zero");
}

int main(void) {
    test_schedule();
    test_rope_tables();
    test_euler_step();
    test_rng();
    test_latent_grid();
    if (failures == 0) {
        printf("host tests passed\n");
        return 0;
    }
    printf("%d failure(s)\n", failures);
    return 1;
}
