#include "h3_gpu.h"
#include "qwen_pipeline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_DIR "tests/pipeline_fixture"

static int failures;

static float *read_f32(const char *name, size_t *count) {
    char path[512];
    snprintf(path, sizeof(path), FIXTURE_DIR "/%s", name);
    FILE *file = fopen(path, "rb");
    if (!file) {
        printf("missing fixture %s\n", path);
        failures++;
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    float *values = malloc((size_t)size);
    if (fread(values, 1, (size_t)size, file) != (size_t)size) {
        fclose(file);
        free(values);
        failures++;
        return NULL;
    }
    fclose(file);
    *count = (size_t)size / 4;
    return values;
}

static void compare_f32(const char *name, const float *got,
                        const float *ref, size_t count, double tolerance) {
    double sum_diff = 0.0, sum_ref = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = fabs((double)got[i] - (double)ref[i]);
        sum_diff += diff * diff;
        sum_ref += (double)ref[i] * ref[i];
        if (diff > max_diff) max_diff = diff;
    }
    double rel = sqrt(sum_diff) / (sqrt(sum_ref) + 1e-30);
    printf("  %-22s relL2 %.3g  maxAbs %.3g\n", name, rel, max_diff);
    if (!(rel <= tolerance)) {
        printf("FAIL %s relL2 %.3g (tolerance %.3g)\n", name, rel,
               tolerance);
        failures++;
    }
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("gpu create failed: %s\n", error);
        return 1;
    }

    if (!h3_gpu_begin(gpu)) {
        printf("begin failed: %s\n", h3_gpu_error(gpu));
        return 1;
    }

    /* RoPE tables vs the released QwenImage21Rope. */
    struct {
        uint32_t text_rows, grid;
        size_t rows;
    } rope_cases[] = {{23, 4, 23 + 16}, {23, 8, 23 + 64}};
    for (size_t c = 0; c < sizeof(rope_cases) / sizeof(rope_cases[0]); c++) {
        uint32_t grid = rope_cases[c].grid;
        h3_gpu_tensor *cos_t = NULL, *sin_t = NULL;
        if (!qw_pipe_rope_tables(gpu, rope_cases[c].text_rows, grid, grid,
                                 &cos_t, &sin_t, error, sizeof(error))) {
            printf("rope build failed: %s\n", error);
            failures++;
            continue;
        }
        if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) {
            printf("rope sync failed: %s\n", h3_gpu_error(gpu));
            failures++;
            continue;
        }
        size_t count = (size_t)rope_cases[c].rows * 64;
        float *cos_host = malloc(sizeof(float) * count);
        float *sin_host = malloc(sizeof(float) * count);
        h3_gpu_tensor_read_f32(cos_t, cos_host, count);
        h3_gpu_tensor_read_f32(sin_t, sin_host, count);
        char name[32];
        size_t ref_count = 0;
        snprintf(name, sizeof(name), "rope_cos_%u.f32", grid);
        float *ref = read_f32(name, &ref_count);
        if (ref) {
            if (ref_count != count) {
                printf("rope %u: size %zu vs %zu\n", grid, ref_count, count);
                failures++;
            } else {
                snprintf(name, sizeof(name), "rope_cos[%u]", grid);
                compare_f32(name, cos_host, ref, count, 1e-6);
            }
            free(ref);
        }
        snprintf(name, sizeof(name), "rope_sin_%u.f32", grid);
        ref = read_f32(name, &ref_count);
        if (ref) {
            if (ref_count != count) {
                printf("rope %u: size %zu vs %zu\n", grid, ref_count, count);
                failures++;
            } else {
                snprintf(name, sizeof(name), "rope_sin[%u]", grid);
                compare_f32(name, sin_host, ref, count, 1e-6);
            }
            free(ref);
        }
        free(cos_host);
        free(sin_host);
        h3_gpu_tensor_free(cos_t);
        h3_gpu_tensor_free(sin_t);
    }

    /* Sigma schedules vs the diffusers scheduler. Each checkpoint is
     * mirrored against its own scheduler config. */
    const qw_sched_cfg base_cfg = {1, 256.0, 8192.0, 0.5, 0.9, 1, 0.02};
    const qw_sched_cfg turbo_cfg = {0, 256.0, 8192.0, 0.5, 0.9, 0, 0.0};
    struct {
        uint32_t steps, seq_len;
        const char *name;
        const char *file;
    } sched_cases[] = {
        {40, 64, "s40", "schedule_s40.f32"},
        {40, 4096, "s40@1024", "schedule_s40_1024.f32"},
    };
    for (size_t c = 0; c < sizeof(sched_cases) / sizeof(sched_cases[0]); c++) {
        double sigmas[65];
        if (!qw_pipe_schedule(sched_cases[c].steps, sched_cases[c].seq_len,
                              NULL, 0, &base_cfg, sigmas, error,
                              sizeof(error))) {
            printf("schedule failed: %s\n", error);
            failures++;
            continue;
        }
        size_t ref_count = 0;
        float *ref = read_f32(sched_cases[c].file, &ref_count);
        if (!ref) continue;
        if (ref_count != sched_cases[c].steps + 1) {
            printf("schedule %s: %zu vs %u\n", sched_cases[c].name,
                   ref_count, sched_cases[c].steps + 1);
            failures++;
            free(ref);
            continue;
        }
        float schedule_f32[65];
        for (size_t i = 0; i < ref_count; i++)
            schedule_f32[i] = (float)sigmas[i];
        compare_f32(sched_cases[c].name, schedule_f32, ref, ref_count, 1e-6);
        free(ref);
    }

    /* Turbo schedule: the turbo scheduler config consumes the released
     * sample_sigmas as-is (no dynamic shift, no terminal stretch). */
    double turbo[8] = {1.0, 0.978453, 0.95418, 0.926626,
                       0.89508, 0.845148, 0.704534, 0.414568};
    double sigmas[9];
    if (!qw_pipe_schedule(8, 64, turbo, 8, &turbo_cfg, sigmas, error,
                          sizeof(error))) {
        printf("turbo schedule failed: %s\n", error);
        failures++;
    } else {
        size_t ref_count = 0;
        float *ref = read_f32("schedule_t8.f32", &ref_count);
        if (ref) {
            float turbo_f32[9];
            for (size_t i = 0; i < ref_count; i++)
                turbo_f32[i] = (float)sigmas[i];
            compare_f32("turbo8", turbo_f32, ref, ref_count, 1e-6);
            free(ref);
        }
    }

    /* RNG smoke: mean/variance over a large sample. */
    qw_pipe_seed(7);
    double sum = 0.0, sum2 = 0.0;
    const int samples = 100000;
    for (int i = 0; i < samples; i++) {
        double value = qw_pipe_randn();
        sum += value;
        sum2 += value * value;
    }
    double mean = sum / samples;
    double variance = sum2 / samples - mean * mean;
    printf("  rng mean %.4f variance %.4f\n", mean, variance);
    if (fabs(mean) > 0.02 || fabs(variance - 1.0) > 0.02) {
        printf("FAIL rng statistics\n");
        failures++;
    }

    if (!failures) printf("pipeline component tests passed\n");
    printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
