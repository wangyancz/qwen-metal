#include "qwen.h"

#include "qwen_dit.h"
#include "qwen_json.h"
#include <mach/mach.h>

#include "qwen_model.h"
#include "qwen_pipeline.h"
#include "qwen_imgproc.h"
#include "qwen_png.h"
#include "qwen_text_encoder.h"
#include "qwen_vae.h"
#include "linenoise.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <stdarg.h>

#define MAX_LORA 8 /* --lora / REPL /lora stack cap */
/* Denoise-loop entry bound (qwen_host.h defines QWEN_STEPS_LIMIT=1000 for
 * the schedule array): without a cap a JSON `steps` of 4294967295 wraps
 * steps+1 to zero in the pipeline's sigmas malloc (malloc(0) passes, the
 * schedule then writes sigmas_out[steps] through a zero-byte block), and
 * mid-sized values simply hang the resident worker for days. */
#define QWEN_STEPS_LIMIT QWEN_MAX_STEPS

static void usage(void) {
    printf("qwen %s - native Qwen-Image-2.1 inference for Apple Silicon\n\n",
           QWEN_VERSION);
    printf("usage:\n");
    printf("  qwen --info -d MODEL_DIR     Inspect a checkpoint without loading weights\n");
    printf("  qwen generate -d MODEL_DIR -p PROMPT [options]\n");
    printf("      --negative TEXT   classifier-free guidance prompt\n");
    printf("      --width N         output width in pixels (multiple of 32)\n");
    printf("      --height N        output height in pixels (multiple of 32)\n");
    printf("      --steps N         denoising steps (default 40; turbo\n");
    printf("                        models always use their official 8-step\n");
    printf("                        fixed schedule, --steps is ignored)\n");
    printf("      --seed N          latent seed (default: time)\n");
    printf("      --cfg F           guidance scale, needs --negative (default 1)\n");
    printf("      --image PATH      ti2i: reference image (PNG)\n");
    printf("      --res N           ti2i: output resolution area (default 1024)\n");
    printf("      --lora PATH       add a LoRA adapter (repeat to stack,\n");
    printf("                        at most %d)\n", MAX_LORA);
    printf("      --lora-scale F    strength of the i-th --lora (repeat to\n");
    printf("                        match; unset default 1)\n");
    printf("      --out PATH        output PNG (default out.png)\n");
    printf("  qwen repl -d MODEL_DIR [options]\n");
    printf("      Interactive session: the model loads once, then every line\n");
    printf("      is a prompt. Directives: /width /height /steps /seed /cfg\n");
    printf("      /neg /image /res /lora /lora-scale /out /show /help /quit\n");
    printf("  qwen serve -d MODEL_DIR\n");
    printf("      JSONL worker for qwen-web (same wire format as the\n");
    printf("      diffusers worker.py): one request per stdin line,\n");
    printf("      progress/done events on stdout; the engine stays\n");
    printf("      resident until stdin closes\n");
    printf("  qwen --help           Show this help\n");
    printf("  qwen --version        Print the version\n");
}

#define MAX_TI2I_IMAGES 8
typedef struct {
    const char *prompt;
    const char *negative;
    const char *out;
    const char *images[MAX_TI2I_IMAGES]; /* --image 可重复 */
    int n_images;
    uint32_t width, height, steps;
    uint64_t seed;
    int seed_set; /* --seed explicitly given (0 is a valid seed) */
    float cfg;
    uint32_t output_res;
    const char *lora[MAX_LORA]; /* --lora 可重复，多适配器叠加 */
    int n_lora;
    float lora_scale[MAX_LORA]; /* --lora-scale 可重复，按序与 lora 配对 */
    int n_scale;
} generate_args;

typedef struct {
    h3_gpu *gpu;
    qw_dit *dit;
    qw_text *text;
    qw_vae *vae;
    qw_vision *vision;
    double turbo_sigmas[64];
    uint32_t turbo_count;
    qw_sched_cfg sched;
} engine;

/* Tear down a possibly half-built engine. The sub-objects normally live
 * as long as the process, so the only callers are load_engine's failure
 * paths — they matter because a load failure is reported and the process
 * then exits with whatever was built still mounted (and because a host
 * that survived a load failure would strand it for good). Towers hold
 * tensors owned by the device, so they go first; every free is
 * NULL-safe. */
static void free_engine(engine *eng) {
    qw_vision_free(eng->vision);
    qw_vae_free(eng->vae);
    qw_text_free(eng->text);
    qw_dit_free(eng->dit);
    h3_gpu_free(eng->gpu);
    memset(eng, 0, sizeof(*eng));
}

static int load_engine(const char *dir, engine *eng, char *error,
                       size_t error_size) {
    eng->gpu = h3_gpu_create("h3_shaders.metal", error, error_size);
    if (!eng->gpu) return 0;
    if (!h3_gpu_begin(eng->gpu)) {
        snprintf(error, error_size, "%s", h3_gpu_error(eng->gpu));
        free_engine(eng);
        return 0;
    }
    eng->dit = qw_dit_create(eng->gpu, dir, 32, error, error_size);
    if (!eng->dit) {
        free_engine(eng);
        return 0;
    }
    eng->text = qw_text_create(eng->gpu, dir, QWEN_TEXT_LAYERS, error,
                               error_size);
    if (!eng->text) {
        free_engine(eng);
        return 0;
    }
    /* vision（视觉塔）只有 ti2i 会话用到：不在这里加载，纯 t2i 的一次性
     * 生成不必为它付 ~1.1GB GPU 权重 + ~2.2GB host fp32 的加载与内存。
     * ti2i 分支按需创建；REPL 在首次 /image 生成时惰性创建并常驻。 */

    eng->vae = qw_vae_create(eng->gpu, dir, error, error_size);
    if (!eng->vae) {
        free_engine(eng);
        return 0;
    }

    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/model_index.json", dir);
    FILE *file = fopen(config_path, "rb");
    if (file) {
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        fseek(file, 0, SEEK_SET);
        char *body = malloc((size_t)size + 1);
        if (body && fread(body, 1, (size_t)size, file) == (size_t)size) {
            body[size] = 0;
            qw_json *index = NULL;
            if (qw_json_parse(body, &index, error, error_size)) {
                const qw_json *list = qw_json_get(index, "sample_sigmas");
                if (list && qw_json_type(list) == QWEN_JSON_ARRAY) {
                    eng->turbo_count = (uint32_t)qw_json_count(list);
                    if (eng->turbo_count > 64) eng->turbo_count = 64;
                    for (uint32_t i = 0; i < eng->turbo_count; i++)
                        eng->turbo_sigmas[i] =
                            qw_json_number(qw_json_at(list, i));
                }
                qw_json_free(index);
            } else {
                error[0] = 0;
            }
            free(body);
        } else {
            free(body);
        }
        fclose(file);
    }

    /* Scheduler behaviour from scheduler/scheduler_config.json. The
     * base model ships dynamic shifting + a 0.02 terminal stretch;
     * the turbo distillation ships neither — its released
     * sample_sigmas are consumed as-is (applying the base rules to
     * them corrupts the sigma trajectory and softens the image). */
    {
        snprintf(config_path, sizeof(config_path),
                 "%s/scheduler/scheduler_config.json", dir);
        FILE *sched_file = fopen(config_path, "rb");
        qw_sched_cfg *sched = &eng->sched;
        memset(sched, 0, sizeof(*sched));
        sched->base_seq = 256.0;
        sched->max_seq = 8192.0;
        sched->base_shift = 0.5;
        sched->max_shift = 0.9;
        sched->dynamic_shifting = 1;
        sched->has_terminal = 1;
        sched->shift_terminal = 0.02;
        if (sched_file) {
            fseek(sched_file, 0, SEEK_END);
            long size = ftell(sched_file);
            fseek(sched_file, 0, SEEK_SET);
            char *body = malloc((size_t)size + 1);
            if (body && fread(body, 1, (size_t)size, sched_file) ==
                            (size_t)size) {
                body[size] = 0;
                qw_json *json = NULL;
                if (qw_json_parse(body, &json, error, error_size)) {
                    sched->dynamic_shifting =
                        qw_json_get_bool(json, "use_dynamic_shifting", 1);
                    sched->base_seq = qw_json_get_number(
                        json, "base_image_seq_len", 256.0);
                    sched->max_seq = qw_json_get_number(
                        json, "max_image_seq_len", 8192.0);
                    sched->base_shift =
                        qw_json_get_number(json, "base_shift", 0.5);
                    sched->max_shift =
                        qw_json_get_number(json, "max_shift", 0.9);
                    const qw_json *terminal =
                        qw_json_get(json, "shift_terminal");
                    sched->has_terminal = terminal && qw_json_type(terminal) !=
                                                                 QWEN_JSON_NULL;
                    sched->shift_terminal =
                        sched->has_terminal
                            ? qw_json_get_number(json, "shift_terminal", 0.02)
                            : 0.0;
                    qw_json_free(json);
                } else {
                    error[0] = 0;  // defaults above already mirror base
                }
                free(body);
            } else {
                free(body);
            }
            fclose(sched_file);
        }
    }
    return 1;
}

/* Encode a prompt through the t2i template; returns the txt_in input. */
static h3_gpu_tensor *encode_prompt(engine *eng, const char *prompt,
                                    uint32_t *rows_out, char *error,
                                    size_t error_size) {
    uint32_t *ids = NULL;
    size_t count = 0, drop = 0;
    if (!qw_text_template_tokens(eng->text, prompt, &ids, &count, &drop,
                                 error, error_size))
        return NULL;
    uint32_t rows = (uint32_t)(count - drop);
    h3_gpu_tensor *text_in =
        h3_gpu_tensor_new_bf16(eng->gpu, (size_t)rows * 4096);
    if (!text_in ||
        !qw_text_encode(eng->text, ids + drop, rows, text_in, error,
                        error_size)) {
        /* The tensor is +1 owned: an encode failure must release it (a
         * serve session would leak one per failed request). */
        h3_gpu_tensor_free(text_in);
        free(ids);
        return NULL;
    }
    free(ids);
    *rows_out = rows;
    return text_in;
}

/* REPL sizes get the same sanity bounds as the serve path: a typo in
 * /width would otherwise start allocating tens of GB (and wrap the
 * 32-bit pixel count the pipeline derives from it) long before any
 * allocation actually fails. Out-of-range input keeps the old value. */
static uint32_t repl_size_arg(const char *text, uint32_t fallback) {
    unsigned long v = strtoul(text, NULL, 10);
    if (v < 32 || v > 8192) {
        fprintf(stderr, "width/height must be in [32, 8192]\n");
        return fallback;
    }
    return (uint32_t)v;
}

/* Strict numeric arg parsing for --width/--height/--steps and their REPL
 * counterparts: strtoul on "abc" silently yields 0, which used to turn
 * /cfg garbage into a cfg=0 (guidance silently off) run. */
static int parse_u32_arg(const char *text, uint32_t *out) {
    char *end = NULL;
    unsigned long v = strtoul(text, &end, 10);
    if (!end || end == text || *end) return 0;
    if (v > 0xfffffffful) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int parse_float_arg(const char *text, float *out) {
    char *end = NULL;
    float v = strtof(text, &end);
    if (!end || end == text || *end) return 0;
    *out = v;
    return 1;
}

static int valid_size(uint32_t v) { return v >= 32 && v <= 8192; }

/* Expand %llu in a user-supplied /out template. The template must never
 * reach snprintf as a format string: "/out %s.png" would pull a pointer
 * out of thin air and crash the session. */
static void format_out_path(char *dst, size_t dst_size, const char *tpl,
                            unsigned long long index) {
    const char *mark = strstr(tpl, "%llu");
    if (!mark) {
        snprintf(dst, dst_size, "%s", tpl);
        return;
    }
    size_t pre = (size_t)(mark - tpl);
    if (pre > dst_size - 1) pre = dst_size - 1;
    memcpy(dst, tpl, pre);
    int n = snprintf(dst + pre, dst_size - pre, "%llu", index);
    if (n < 0) n = 0;
    size_t used = pre + (size_t)n;
    if (used > dst_size - 1) used = dst_size - 1;
    snprintf(dst + used, dst_size - used, "%s", mark + 4);
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* turbo 蒸馏模型只按官方固定 sample_sigmas 轨迹定义（distillation 就是
 * 对这条轨迹学的）：请求的步数被忽略，与 serve/Go 侧/worker.py 的归一
 * 口径一致。此前 steps≠turbo 步数会绕过固定轨迹采样出废图（denoise 的
 * schedule 失败误报成功），改为处处归一。 */
static uint32_t eff_steps(const engine *eng, uint32_t requested) {
    return eng->turbo_count ? eng->turbo_count : requested;
}

static double phys_footprint_gb(void) {
    struct task_vm_info info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info,
                  &count) != KERN_SUCCESS)
        return -1.0;
    return (double)info.phys_footprint / 1e9;
}

/* Apply the --lora stack on a freshly loaded engine (both generate
 * branches and the REPL share this). Each rebuild fuses every stacked
 * adapter in one pass (~3 s), independent of the stack size. */
static int apply_lora_arg(engine *eng, const generate_args *args,
                          char *error, size_t error_size) {
    for (int i = 0; i < args->n_lora; i++) {
        float scale = i < args->n_scale ? args->lora_scale[i] : 1.0f;
        double start = now_seconds();
        if (!qw_dit_apply_lora(eng->dit, args->lora[i], scale, error,
                               error_size))
            return 0;
        printf("lora applied: %s (scale %.2f, %.1fs)\n", args->lora[i],
               scale, now_seconds() - start);
    }
    return 1;
}

static int run_generation(engine *eng, const h3_gpu_tensor *text_in,
                          uint32_t text_rows,
                          const h3_gpu_tensor *neg_text_in, uint32_t neg_rows,
                          const generate_args *args, const char *out_path,
                          int one_shot, char *error, size_t error_size) {
    qw_pipe_params params = {0};
    params.width = args->width ? args->width : 1024;
    params.height = args->height ? args->height : 1024;
    params.steps = eff_steps(eng, args->steps ? args->steps : 40);
    params.seed = args->seed;
    params.cfg_scale = args->cfg;
    if (params.width % 32 || params.height % 32) {
        snprintf(error, error_size, "size must be multiples of 32");
        return 0;
    }
    uint32_t gh = params.height / 16;
    uint32_t gw = params.width / 16;
    size_t count = (size_t)gh * gw * 64;
    float *latents = malloc(sizeof(float) * count);
    if (!latents) {
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    qw_pipe_seed(params.seed);
    for (size_t i = 0; i < count; i++) latents[i] = qw_pipe_randn();
    double start = now_seconds();
    if (!qw_pipe_denoise(eng->dit, eng->gpu, text_in, text_rows,
                         neg_text_in, neg_rows, latents, gh, gw,
                         params.steps, params.cfg_scale,
                         eng->turbo_count ? eng->turbo_sigmas : NULL,
                         eng->turbo_count, &eng->sched, error, error_size)) {
        free(latents);
        return 0;
    }
    if (one_shot) {
        /* One-shot generate: the DiT (13 GB) is dead once the denoise
         * loop is done — decode runs on the VAE weights alone. The
         * REPL keeps it resident for the next line. */
        qw_dit_free(eng->dit);
        eng->dit = NULL;
    }
    uint8_t *rgba = NULL;
    int ok = qw_pipe_decode(eng->vae, eng->gpu, latents, gh, gw, &rgba,
                            error, error_size);
    free(latents);
    if (!ok) return 0;
    double generate_seconds = now_seconds() - start;
    if (qw_png_write_rgba(out_path, params.width, params.height, rgba, error,
                          error_size)) {
        free(rgba);
        return 0;
    }
    free(rgba);
    printf("wrote %s (%ux%u, %u steps, %.1fs)\n", out_path, params.width,
           params.height, (unsigned)params.steps, generate_seconds);
    return 1;
}

static int cmd_generate(const char *dir, const generate_args *args_in) {
    char error[512];
    /* usage documents --seed's default as "time" (the REPL does the
     * same): an unseeded one-shot must not silently replay one fixed
     * trajectory on every run. */
    generate_args args = *args_in;
    if (!args.seed_set) args.seed = (uint64_t)time(NULL);

    /* ti2i 分支：text/vision/vae/dit 全程共存（文本塔在视觉塔之后），
     * 独立于 t2i 的提前释放流程。 */
    if (args.n_images) {
        engine eng;
        memset(&eng, 0, sizeof(eng));
        if (!load_engine(dir, &eng, error, sizeof(error))) {
            fprintf(stderr, "error: %s\n", error);
            return 1;
        }
        if (!apply_lora_arg(&eng, &args, error, sizeof(error))) {
            fprintf(stderr, "error: %s\n", error);
            return 1;
        }
        /* ti2i 专属：视觉塔按需加载（load_engine 不再常驻）。 */
        if (!eng.vision) {
            eng.vision = qw_vision_create(dir, eng.gpu, error, sizeof(error));
            if (!eng.vision) {
                fprintf(stderr, "error: %s\n", error);
                return 1;
            }
        }
        uint8_t *rgba_in[MAX_TI2I_IMAGES] = {0};
        uint32_t iw_arr[MAX_TI2I_IMAGES] = {0}, ih_arr[MAX_TI2I_IMAGES] = {0};
        for (int i = 0; i < args.n_images; i++) {
            if (qw_png_read_rgba(args.images[i], &iw_arr[i], &ih_arr[i],
                                 &rgba_in[i], error, sizeof(error))) {
                for (int j = 0; j < i; j++) free(rgba_in[j]);
                fprintf(stderr, "error: %s\n", error);
                return 1;
            }
        }
        uint32_t ow = 0, oh = 0;
        qw_imgproc_output_size((int)iw_arr[0], (int)ih_arr[0],
                               args.output_res ? (int)args.output_res
                                               : 1024,
                               (int *)&ow, (int *)&oh);
        qw_pipe_params params = {0};
        params.width = args.width ? args.width : ow;
        params.height = args.height ? args.height : oh;
        if (params.width % 32 || params.height % 32) {
            fprintf(stderr,
                    "error: --width/--height must be multiples of 32\n");
            for (int i = 0; i < args.n_images; i++) free(rgba_in[i]);
            return 1;
        }
        params.output_res = args.output_res ? args.output_res : 1024;
        params.steps = eff_steps(&eng, args.steps ? args.steps : 40);
        params.seed = args.seed;
        params.cfg_scale = args.cfg;
        params.negative = args.negative;
        uint8_t *rgba = NULL;
        double start = now_seconds();
        int ok = qw_pipe_generate_ti2i(eng.dit, eng.vae, eng.text, eng.vision,
                                       eng.gpu, args.prompt,
                                       (const uint8_t *const *)rgba_in, iw_arr,
                                       ih_arr, (uint32_t)args.n_images,
                                       &params,
                                       eng.turbo_count ? eng.turbo_sigmas
                                                       : NULL,
                                       eng.turbo_count, &eng.sched, &rgba,
                                       error, sizeof(error));
        for (int i = 0; i < args.n_images; i++) free(rgba_in[i]);
        if (!ok) {
            fprintf(stderr, "error: %s\n", error);
            return 1;
        }
        double generate_seconds = now_seconds() - start;
        const char *out_path =
            args.out && *args.out ? args.out : "out.png";
        if (qw_png_write_rgba(out_path, params.width, params.height, rgba,
                              error, sizeof(error))) {
            free(rgba);
            fprintf(stderr, "error: %s\n", error);
            return 1;
        }
        free(rgba);
        printf("wrote %s (%ux%u from %u refs, %u steps, %.1fs)\n", out_path,
               params.width, params.height, args.n_images,
               (unsigned)params.steps, generate_seconds);
        return 0;
    }

    engine eng;
    memset(&eng, 0, sizeof(eng));
    if (!load_engine(dir, &eng, error, sizeof(error))) {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }
    if (!apply_lora_arg(&eng, &args, error, sizeof(error))) {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }
    if (eng.turbo_count)
        printf("turbo schedule: %u fixed sigmas\n", (unsigned)eng.turbo_count);
    uint32_t text_rows = 0;
    h3_gpu_tensor *text_in =
        encode_prompt(&eng, args.prompt, &text_rows, error, sizeof(error));
    if (!text_in) {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }
    h3_gpu_tensor *neg_text_in = NULL;
    uint32_t neg_rows = 0;
    if (args.cfg > 1.0f && args.negative && *args.negative) {
        neg_text_in =
            encode_prompt(&eng, args.negative, &neg_rows, error,
                          sizeof(error));
        if (!neg_text_in) {
            fprintf(stderr, "error: %s\n", error);
            h3_gpu_tensor_free(text_in);
            return 1;
        }
    }
    /* One-shot generate: the text tower (16 GB) is dead once both
     * prompts are encoded. */
    h3_gpu_stats st_before, st_after;
    h3_gpu_get_stats(eng.gpu, &st_before);
    fprintf(stderr, "[mem] live after encode: %.2f GB (footprint %.2f)\n",
            (double)st_before.live_bytes / 1e9, phys_footprint_gb());
    qw_text_free(eng.text);
    eng.text = NULL;
    if (!h3_gpu_submit(eng.gpu) || !h3_gpu_begin(eng.gpu)) {
        fprintf(stderr, "error: %s\n", h3_gpu_error(eng.gpu));
        h3_gpu_tensor_free(text_in);
        h3_gpu_tensor_free(neg_text_in);
        return 1;
    }
    h3_gpu_get_stats(eng.gpu, &st_after);
    fprintf(stderr, "[mem] live after text release: %.2f GB (footprint %.2f)\n",
            (double)st_after.live_bytes / 1e9, phys_footprint_gb());
    int ok = run_generation(&eng, text_in, text_rows, neg_text_in, neg_rows,
                            &args,
                            args.out && *args.out ? args.out : "out.png",
                            1, error, sizeof(error));
    /* Both conditioning tensors are +1 owned; every exit path frees
     * them (the ~2.6 GB still resident here is the VAE weights, which a
     * one-shot run leaves mounted until the process dies). */
    h3_gpu_tensor_free(text_in);
    h3_gpu_tensor_free(neg_text_in);
    h3_gpu_get_stats(eng.gpu, &st_after);
    fprintf(stderr, "[mem] live at exit: %.2f GB (footprint %.2f)\n",
            (double)st_after.live_bytes / 1e9, phys_footprint_gb());
    if (!ok) fprintf(stderr, "error: %s\n", error);
    return ok ? 0 : 1;
}

/* ------------------------------ REPL ------------------------------ */

typedef struct {
    uint32_t width, height, steps;
    uint64_t seed;
    float cfg;
    char negative[2048];
    char out[1024];
    uint64_t out_index;
    char images[MAX_TI2I_IMAGES][1024]; /* /image 累积的参考图路径 */
    int n_images;
    uint32_t output_res; /* ti2i 输出分辨率档（0 = 默认 1024） */
    int width_set, height_set; /* /width /height 显式覆盖过 ti2i 尺寸 */
    char lora[MAX_LORA][1024]; /* /lora 累积的适配器栈（叠加） */
    float lora_scale[MAX_LORA];
    int n_lora;
} repl_state;

static void repl_help(void) {
    printf("  /width N      output width (multiple of 32)\n");
    printf("  /height N     output height (multiple of 32)\n");
    printf("  /steps N      denoising steps\n");
    printf("  /seed N       latent seed (auto-increments per image)\n");
    printf("  /cfg F        guidance scale (>1 needs /neg)\n");
    printf("  /neg TEXT     negative prompt (empty /neg clears)\n");
    printf("  /image PATH   add a reference image (repeat up to %d;\n",
           MAX_TI2I_IMAGES);
    printf("                bare /image clears; prompts may reference\n");
    printf("                them as <imageN>)\n");
    printf("  /res N        ti2i output resolution area (default 1024)\n");
    printf("  /lora PATH    add a LoRA adapter to the stack (adapters\n");
    printf("                stack additively; bare /lora clears all;\n");
    printf("                each change re-fuses, takes seconds)\n");
    printf("  /lora-scale F strength of the most recent adapter\n");
    printf("  /lora-drop PATH\n");
    printf("                remove one adapter from the stack\n");
    printf("  /out PATH     output PNG path (%%llu -> image index)\n");
    printf("  /show         current settings\n");
    printf("  /help         this help\n");
    printf("  /quit         leave the session\n");
    printf("  anything else is used as the prompt (ti2i when references\n");
    printf("  are set)\n");
}

/* Re-sync the REPL's adapter list from the engine. The engine resets the
 * whole stack when a rebuild fails, so every /lora* directive resyncs
 * after the call — success or failure — keeping the display and the
 * /lora-scale target truthful. */
static void repl_sync_lora(repl_state *st, const qw_dit *dit) {
    int n = qw_dit_lora_count(dit);
    if (n > MAX_LORA) n = MAX_LORA;
    for (int i = 0; i < n; i++) {
        snprintf(st->lora[i], sizeof(st->lora[i]), "%s",
                 qw_dit_lora_path_at(dit, i));
        st->lora_scale[i] = qw_dit_lora_scale_at(dit, i);
    }
    st->n_lora = n;
}

static void repl_show(const repl_state *st) {
    printf("  size %ux%u  steps %u  seed %llu  cfg %.2f  neg %s\n", st->width,
           st->height, st->steps, (unsigned long long)st->seed, st->cfg,
           st->negative[0] ? st->negative : "(none)");
    printf("  refs %d  res %u  lora %d\n", st->n_images,
           st->output_res ? st->output_res : 1024, st->n_lora);
    for (int i = 0; i < st->n_lora; i++)
        printf("    lora[%d] %s (scale %.2f)\n", i + 1, st->lora[i],
               st->lora_scale[i]);
    for (int i = 0; i < st->n_images; i++)
        printf("    <image%d> %s\n", i + 1, st->images[i]);
}

static int cmd_repl(const char *dir, const generate_args *args) {
    char error[512];
    engine eng;
    memset(&eng, 0, sizeof(eng));
    if (!load_engine(dir, &eng, error, sizeof(error))) {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }
    if (!apply_lora_arg(&eng, args, error, sizeof(error))) {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }
    repl_state st;
    memset(&st, 0, sizeof(st));
    st.width = args->width ? args->width : 1024;
    st.height = args->height ? args->height : 1024;
    /* CLI-provided sizes are explicit overrides: without the set flags a
     * ti2i prompt would silently fall back to the reference-derived size
     * while /show still claimed the --width value. */
    st.width_set = args->width ? 1 : 0;
    st.height_set = args->height ? 1 : 0;
    st.steps = args->steps ? args->steps
                           : (eng.turbo_count ? eng.turbo_count : 40);
    st.seed = args->seed_set ? args->seed : (uint64_t)time(NULL);
    st.cfg = args->cfg;
    snprintf(st.out, sizeof(st.out), "%s",
             args->out && *args->out ? args->out : "out-%llu.png");
    for (int i = 0; i < args->n_lora && i < MAX_LORA; i++) {
        snprintf(st.lora[i], sizeof(st.lora[i]), "%s", args->lora[i]);
        st.lora_scale[i] = i < args->n_scale ? args->lora_scale[i] : 1.0f;
        st.n_lora++;
    }
    printf("qwen %s - interactive session (model resident)\n", QWEN_VERSION);
    if (eng.turbo_count)
        printf("turbo schedule: %u fixed sigmas\n", (unsigned)eng.turbo_count);
    repl_show(&st);
    printf("type /help for directives; /quit leaves\n");

    linenoiseHistorySetMaxLen(100);
    uint64_t image_index = 0;
    char *line;
    while ((line = linenoise("qwen> ")) != NULL) {
        const char *text = line;
        while (*text == ' ' || *text == '\t') text++;
        if (!*text) {
            linenoiseFree(line);
            continue;
        }
        linenoiseHistoryAdd(line);
        if (strcmp(text, "/quit") == 0 || strcmp(text, "/exit") == 0) {
            linenoiseFree(line);
            break;
        } else if (strcmp(text, "/help") == 0) {
            repl_help();
        } else if (strcmp(text, "/show") == 0) {
            repl_show(&st);
        } else if (strncmp(text, "/width ", 7) == 0) {
            st.width = repl_size_arg(text + 7, st.width);
            st.width_set = 1;
            repl_show(&st);
        } else if (strncmp(text, "/height ", 8) == 0) {
            st.height = repl_size_arg(text + 8, st.height);
            st.height_set = 1;
            repl_show(&st);
        } else if (strncmp(text, "/steps ", 7) == 0) {
            uint32_t v;
            if (parse_u32_arg(text + 7, &v) && v >= 1 && v <= QWEN_STEPS_LIMIT)
                st.steps = v;
            else
                fprintf(stderr, "steps must be in [1, %u]\n", QWEN_STEPS_LIMIT);
            repl_show(&st);
        } else if (strncmp(text, "/seed ", 6) == 0) {
            st.seed = strtoull(text + 6, NULL, 10);
            repl_show(&st);
        } else if (strncmp(text, "/cfg ", 5) == 0) {
            float v;
            if (parse_float_arg(text + 5, &v))
                st.cfg = v;
            else
                fprintf(stderr, "error: /cfg needs a number\n");
            repl_show(&st);
        } else if (strncmp(text, "/neg ", 5) == 0) {
            snprintf(st.negative, sizeof(st.negative), "%s", text + 5);
            repl_show(&st);
        } else if (strcmp(text, "/neg") == 0) {
            st.negative[0] = 0;
            repl_show(&st);
        } else if (strncmp(text, "/out ", 5) == 0) {
            snprintf(st.out, sizeof(st.out), "%s", text + 5);
            repl_show(&st);
        } else if (strncmp(text, "/image ", 7) == 0) {
            if (st.n_images >= MAX_TI2I_IMAGES) {
                fprintf(stderr, "error: at most %d reference images\n",
                        MAX_TI2I_IMAGES);
            } else {
                snprintf(st.images[st.n_images],
                         sizeof(st.images[st.n_images]), "%s", text + 7);
                st.n_images++;
                repl_show(&st);
            }
        } else if (strcmp(text, "/image") == 0) {
            st.n_images = 0;
            repl_show(&st);
        } else if (strncmp(text, "/res ", 5) == 0) {
            uint32_t v;
            if (parse_u32_arg(text + 5, &v) &&
                (v == 0 || (v >= 128 && v <= 4096)))
                st.output_res = v;
            else
                fprintf(stderr, "res must be in [128, 4096] (0 = default)\n");
            repl_show(&st);
        } else if (strcmp(text, "/res") == 0) {
            st.output_res = 0;
            repl_show(&st);
        } else if (strncmp(text, "/lora ", 6) == 0) {
            double start = now_seconds();
            if (qw_dit_apply_lora(eng.dit, text + 6, 1.0f, error,
                                  sizeof(error)))
                printf("lora applied (%.1fs)\n", now_seconds() - start);
            else
                fprintf(stderr, "error: %s\n", error);
            repl_sync_lora(&st, eng.dit);
            repl_show(&st);
        } else if (strcmp(text, "/lora") == 0) {
            if (qw_dit_clear_lora(eng.dit, error, sizeof(error)))
                printf("lora cleared\n");
            else
                fprintf(stderr, "error: %s\n", error);
            repl_sync_lora(&st, eng.dit);
            repl_show(&st);
        } else if (strncmp(text, "/lora-scale ", 12) == 0) {
            float scale;
            if (!parse_float_arg(text + 12, &scale)) {
                fprintf(stderr, "error: /lora-scale needs a number\n");
            } else if (st.n_lora) {
                double start = now_seconds();
                if (qw_dit_apply_lora(eng.dit, st.lora[st.n_lora - 1],
                                      scale, error, sizeof(error)))
                    printf("lora[%d] scale %.2f applied (%.1fs)\n",
                           st.n_lora, scale, now_seconds() - start);
                else
                    fprintf(stderr, "error: %s\n", error);
            } else {
                fprintf(stderr, "error: no lora loaded\n");
            }
            repl_sync_lora(&st, eng.dit);
            repl_show(&st);
        } else if (strncmp(text, "/lora-drop ", 11) == 0) {
            if (qw_dit_remove_lora(eng.dit, text + 11, error,
                                   sizeof(error)))
                printf("lora dropped\n");
            else
                fprintf(stderr, "error: %s\n", error);
            repl_sync_lora(&st, eng.dit);
            repl_show(&st);
        } else if (text[0] == '/' && !strchr(text, ' ')) {
            /* A bare "/width" (argument left off) used to fall through to
             * the prompt branch and burn a full generation on it. Bare
             * /neg /image /res /lora /show /help /quit are handled
             * above. */
            fprintf(stderr,
                    "error: unknown directive or missing argument: %s"
                    " (see /help)\n", text);
        } else if (st.n_images) {
            /* ti2i 会话：vision 惰性创建（首次 /image 生成时），此后常驻。 */
            if (!eng.vision)
                eng.vision = qw_vision_create(dir, eng.gpu, error,
                                              sizeof(error));
            if (!eng.vision) {
                fprintf(stderr, "error: %s\n", error);
            } else {
                uint8_t *rgba_in[MAX_TI2I_IMAGES] = {0};
                uint32_t iw_arr[MAX_TI2I_IMAGES] = {0};
                uint32_t ih_arr[MAX_TI2I_IMAGES] = {0};
                int read_ok = 1;
                for (int i = 0; read_ok && i < st.n_images; i++) {
                    read_ok =
                        !qw_png_read_rgba(st.images[i], &iw_arr[i], &ih_arr[i],
                                          &rgba_in[i], error, sizeof(error));
                    if (!read_ok) fprintf(stderr, "error: %s\n", error);
                }
                if (read_ok) {
                    uint32_t ow = 0, oh = 0;
                    qw_imgproc_output_size(
                        (int)iw_arr[0], (int)ih_arr[0],
                        (int)(st.output_res ? st.output_res : 1024),
                        (int *)&ow, (int *)&oh);
                    qw_pipe_params params = {0};
                    params.width = st.width_set ? st.width : ow;
                    params.height = st.height_set ? st.height : oh;
                    if (params.width % 32 || params.height % 32) {
                        fprintf(stderr,
                                "error: /width//height must be multiples of"
                                " 32\n");
                    } else {
                        params.output_res =
                            st.output_res ? st.output_res : 1024;
                        params.steps = eff_steps(&eng, st.steps);
                        params.seed = st.seed++;
                        params.cfg_scale = st.cfg;
                        params.negative =
                            st.negative[0] ? st.negative : NULL;
                        char path[1200];
                        format_out_path(path, sizeof(path), st.out,
                                        (unsigned long long)image_index);
                        uint8_t *rgba = NULL;
                        double start = now_seconds();
                        if (!qw_pipe_generate_ti2i(
                                eng.dit, eng.vae, eng.text, eng.vision,
                                eng.gpu, text,
                                (const uint8_t *const *)rgba_in, iw_arr,
                                ih_arr, (uint32_t)st.n_images, &params,
                                eng.turbo_count ? eng.turbo_sigmas : NULL,
                                eng.turbo_count, &eng.sched, &rgba, error,
                                sizeof(error))) {
                            fprintf(stderr, "error: %s\n", error);
                        } else {
                            if (qw_png_write_rgba(path, params.width,
                                                  params.height, rgba, error,
                                                  sizeof(error))) {
                                fprintf(stderr, "error: %s\n", error);
                            } else {
                                printf("wrote %s (%ux%u from %d refs, %u"
                                       " steps, %.1fs)\n",
                                       path, params.width, params.height,
                                       st.n_images, (unsigned)params.steps,
                                       now_seconds() - start);
                                image_index++;
                            }
                            free(rgba);
                        }
                    }
                }
                for (int i = 0; i < st.n_images; i++) free(rgba_in[i]);
            }
        } else {
            generate_args gen;
            memset(&gen, 0, sizeof(gen));
            gen.width = st.width;
            gen.height = st.height;
            gen.steps = eff_steps(&eng, st.steps);
            gen.seed = st.seed++;
            gen.cfg = st.cfg;
            uint32_t text_rows = 0, neg_rows = 0;
            h3_gpu_tensor *text_in =
                encode_prompt(&eng, text, &text_rows, error, sizeof(error));
            h3_gpu_tensor *neg_text_in = NULL;
            if (text_in && st.cfg > 1.0f && st.negative[0])
                neg_text_in = encode_prompt(&eng, st.negative, &neg_rows,
                                            error, sizeof(error));
            if (!text_in || (st.cfg > 1.0f && st.negative[0] &&
                             !neg_text_in)) {
                fprintf(stderr, "error: %s\n", error);
            } else {
                char path[1200];
                format_out_path(path, sizeof(path), st.out,
                                (unsigned long long)image_index);
                if (!run_generation(&eng, text_in, text_rows, neg_text_in,
                                    neg_rows, &gen, path, 0, error,
                                    sizeof(error)))
                    fprintf(stderr, "error: %s\n", error);
                else
                    image_index++;
            }
            h3_gpu_tensor_free(text_in);
            h3_gpu_tensor_free(neg_text_in);
        }
        /* Request boundary, same as the serve loop: keeps the
         * shape-keyed MPSGraph caches from growing for the session. */
        h3_gpu_clear_graph_cache(eng.gpu);
        linenoiseFree(line);
    }
    printf("bye\n");
    return 0;
}

/* ------------------------------ serve ------------------------------ */

/* JSONL worker for qwen-web (wire format identical to the diffusers
 * worker.py, so the Go host drives either engine unchanged): requests
 * on stdin, protocol events on stdout, diagnostics on stderr. stdin EOF
 * exits; the host kills the process group for cancellation and idle
 * unload — the resident weights die with it and the next task reloads.
 *
 * Memory model mirrors the REPL (not the one-shot generate): text/DiT/
 * VAE stay resident across requests, the vision tower is created
 * lazily on the first ti2i request and then kept. */

typedef struct {
    pthread_mutex_t mu;
    char rid[80];   /* active request id ("" = idle: heartbeat silent) */
    double started; /* monotonic request start */
    char stage[16];
    double percent; /* total 0..1 in the worker.py span mapping */
    double last_emit;
} serve_state;

static serve_state g_serve = {
    PTHREAD_MUTEX_INITIALIZER, "", 0.0, "", 0.0, 0.0,
};

/* Total-progress spans per stage, mirroring worker.py exactly. */
static void serve_json_escape(const char *in, char *out, size_t out_size);

static int serve_span(const char *stage, double *lo, double *hi) {
    if (strcmp(stage, "encode") == 0) { *lo = 0.0;  *hi = 0.08; return 1; }
    if (strcmp(stage, "sample") == 0) { *lo = 0.08; *hi = 0.92; return 1; }
    if (strcmp(stage, "decode") == 0) { *lo = 0.92; *hi = 0.98; return 1; }
    if (strcmp(stage, "save") == 0)   { *lo = 0.98; *hi = 1.0;  return 1; }
    return 0;
}

static void serve_emit(const char *fmt, ...) {
    /* escaped error strings can reach ~1KB (each control byte becomes
     * \u00xx); 2048 keeps every protocol line untruncated */
    char line[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    /* vsnprintf returns the untruncated length — clamp or fwrite would
     * read past the stack buffer on an over-long line. */
    if ((size_t)n >= sizeof(line)) n = (int)sizeof(line) - 1;
    pthread_mutex_lock(&g_serve.mu);
    fwrite(line, 1, (size_t)n, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    pthread_mutex_unlock(&g_serve.mu);
}

/* Stage transition / ratio update, rate-limited like worker.py. */
static void serve_stage(const char *stage, double ratio, int force) {
    double lo, hi;
    if (!serve_span(stage, &lo, &hi)) return;
    double now = now_seconds();
    pthread_mutex_lock(&g_serve.mu);
    if (!force && now - g_serve.last_emit < 0.25) {
        pthread_mutex_unlock(&g_serve.mu);
        return;
    }
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;
    double percent = lo + ratio * (hi - lo);
    char rid[80], rid_esc[176];
    double started;
    snprintf(rid, sizeof(rid), "%s", g_serve.rid);
    snprintf(g_serve.stage, sizeof(g_serve.stage), "%s", stage);
    g_serve.percent = percent;
    g_serve.last_emit = now;
    started = g_serve.started;
    pthread_mutex_unlock(&g_serve.mu);
    /* The id is caller-supplied JSON: unescaped quotes or control bytes
     * would produce an unparsable protocol line. */
    serve_json_escape(rid, rid_esc, sizeof(rid_esc));
    serve_emit("{\"type\":\"progress\",\"id\":\"%s\",\"stage\":\"%s\","
               "\"percent\":%.4f,\"elapsed\":%.1f}",
               rid_esc, stage, percent, now - started);
}

/* Every 4 s re-send the current progress while a request is active so
 * the host's quiet window never fires over long LoRA rebuilds, first
 * run page-in of the mmap'd weights, or slow single steps. Mirrors
 * worker.py: heartbeat events do not refresh the host idle timer. */
static void *serve_heartbeat(void *arg) {
    (void)arg;
    for (;;) {
        usleep(4 * 1000 * 1000);
        pthread_mutex_lock(&g_serve.mu);
        int active = g_serve.rid[0] != 0;
        char rid[80], stage[16];
        double percent = 0.0, started = 0.0;
        if (active) {
            snprintf(rid, sizeof(rid), "%s", g_serve.rid);
            snprintf(stage, sizeof(stage), "%s", g_serve.stage);
            percent = g_serve.percent;
            started = g_serve.started;
        }
        pthread_mutex_unlock(&g_serve.mu);
        if (!active) continue;
        char rid_esc[176];
        serve_json_escape(rid, rid_esc, sizeof(rid_esc));
        serve_emit("{\"type\":\"progress\",\"id\":\"%s\",\"stage\":\"%s\","
                   "\"percent\":%.4f,\"elapsed\":%.1f,\"heartbeat\":true}",
                   rid_esc, stage, percent, now_seconds() - started);
    }
    return NULL;
}

static void serve_step_cb(uint32_t steps_done, uint32_t steps, void *ctx) {
    (void)ctx;
    serve_stage("sample",
                steps ? (double)steps_done / (double)steps : 0.0, 0);
}

/* worker.py 口径：alpha<200 的像素超过 0.5% 才算成规模透明（零星低
 * alpha 噪点不触发透明徽章）。 */
static int serve_has_alpha(const uint8_t *rgba, size_t pixels) {
    size_t below = 0;
    for (size_t i = 0; i < pixels; i++)
        if (rgba[i * 4 + 3] < 200) below++;
    return below * 200 > pixels;
}

static void serve_json_escape(const char *in, char *out, size_t out_size) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)in;
         *p && o + 7 < out_size; p++) {
        unsigned char c = *p;
        if (c == '"' || c == '\\') {
            out[o++] = '\\';
            out[o++] = (char)c;
        } else if (c == '\n') {
            out[o++] = '\\';
            out[o++] = 'n';
        } else if (c == '\r') {
            out[o++] = '\\';
            out[o++] = 'r';
        } else if (c == '\t') {
            out[o++] = '\\';
            out[o++] = 't';
        } else if (c < 0x20) {
            o += (size_t)snprintf(out + o, out_size - o, "\\u%04x", c);
        } else {
            out[o++] = (char)c; /* UTF-8 passes through */
        }
    }
    out[o] = 0;
}

/* Reconcile the engine's adapter stack with the request: qwen-web sends
 * at most one adapter (path + scale), the engine may still hold the
 * previous request's stack. Same path+scale is a no-op; a scale change
 * re-fuses IN PLACE (clearing first would restore the base slots and
 * force a full fresh fused allocation — ~14 GB of new pool pages per
 * change); a different adapter clears first, then applies (the free
 * precedes every allocation, so the pool recycles the blocks). */
static int serve_sync_lora(engine *eng, const char *path, float scale,
                           char *error, size_t error_size) {
    int n = qw_dit_lora_count(eng->dit);
    if (!path || !*path) {
        if (n == 0) return 1;
        return qw_dit_clear_lora(eng->dit, error, error_size);
    }
    if (n == 1) {
        /* Same file: apply_lora is a no-op (same scale) or an in-place
         * scale rebuild. A different file: switch in place — clearing
         * first would restore the base slots and force a full fresh
         * fused allocation (~14 GB of new pool pages) per switch. */
        return qw_dit_switch_lora(eng->dit, path, scale, error, error_size);
    }
    /* n > 1 only happens with a REPL-stacked adapter set; a web request
     * carries one adapter, so collapse the stack first. */
    if (n > 1 && !qw_dit_clear_lora(eng->dit, error, error_size)) return 0;
    return qw_dit_apply_lora(eng->dit, path, scale, error, error_size);
}

static int serve_generate_t2i(engine *eng, const char *prompt,
                              const char *negative, uint32_t width,
                              uint32_t height, uint32_t steps, float cfg,
                              uint64_t seed, const char *out,
                              int *has_alpha, char *error,
                              size_t error_size) {
    uint32_t text_rows = 0, neg_rows = 0;
    h3_gpu_tensor *text_in =
        encode_prompt(eng, prompt, &text_rows, error, error_size);
    h3_gpu_tensor *neg_text_in = NULL;
    int failed = text_in == NULL;
    if (!failed && cfg > 1.0f && negative && *negative) {
        neg_text_in =
            encode_prompt(eng, negative, &neg_rows, error, error_size);
        failed = neg_text_in == NULL;
    }
    if (failed) {
        h3_gpu_tensor_free(text_in);
        return 0;
    }
    uint32_t gh = height / 16, gw = width / 16;
    size_t count = (size_t)gh * gw * 64;
    float *latents = malloc(sizeof(float) * count);
    if (!latents) {
        snprintf(error, error_size, "out of memory");
        h3_gpu_tensor_free(text_in);
        h3_gpu_tensor_free(neg_text_in);
        return 0;
    }
    qw_pipe_seed(seed);
    for (size_t i = 0; i < count; i++) latents[i] = qw_pipe_randn();
    qw_pipe_set_progress(serve_step_cb, NULL);
    int ok = qw_pipe_denoise(eng->dit, eng->gpu, text_in, text_rows,
                             neg_text_in, neg_rows, latents, gh, gw, steps,
                             cfg, eng->turbo_count ? eng->turbo_sigmas : NULL,
                             eng->turbo_count, &eng->sched, error, error_size);
    qw_pipe_set_progress(NULL, NULL);
    uint8_t *rgba = NULL;
    if (ok) {
        serve_stage("decode", 1.0, 1);
        ok = qw_pipe_decode(eng->vae, eng->gpu, latents, gh, gw, &rgba,
                            error, error_size);
    }
    free(latents);
    h3_gpu_tensor_free(text_in);
    h3_gpu_tensor_free(neg_text_in);
    if (!ok) return 0;
    *has_alpha = serve_has_alpha(rgba, (size_t)width * height);
    serve_stage("save", 0.0, 1);
    if (qw_png_write_rgba(out, width, height, rgba, error, error_size)) {
        free(rgba);
        return 0;
    }
    free(rgba);
    serve_stage("save", 1.0, 1);
    return 1;
}

static int serve_generate_ti2i(engine *eng, const char *dir,
                               const char *prompt, const char *negative,
                               uint32_t width, uint32_t height,
                               uint32_t steps, float cfg, uint64_t seed,
                               const char *out, const char **images,
                               uint32_t n_images, int *has_alpha,
                               char *error, size_t error_size) {
    /* vision 惰性创建（首次 ti2i 请求），此后常驻（REPL 同款）。 */
    if (!eng->vision) {
        eng->vision = qw_vision_create(dir, eng->gpu, error, error_size);
        if (!eng->vision) return 0;
    }
    uint8_t *rgba_in[MAX_TI2I_IMAGES] = {0};
    uint32_t iw_arr[MAX_TI2I_IMAGES] = {0}, ih_arr[MAX_TI2I_IMAGES] = {0};
    for (uint32_t i = 0; i < n_images; i++) {
        if (qw_png_read_rgba(images[i], &iw_arr[i], &ih_arr[i], &rgba_in[i],
                             error, error_size)) {
            for (uint32_t j = 0; j < i; j++) free(rgba_in[j]);
            return 0;
        }
    }
    qw_pipe_params params = {0};
    params.width = width;
    params.height = height;
    /* 官方 output_resolution 默认（参考图按 1024² 面积 32 对齐重采样；
     * 与显式 width/height 无关，CLI/REPL/官方管线同款）。 */
    params.output_res = 1024;
    params.steps = steps;
    params.seed = seed;
    params.cfg_scale = cfg;
    params.negative = negative;
    uint8_t *rgba = NULL;
    qw_pipe_set_progress(serve_step_cb, NULL);
    int ok = qw_pipe_generate_ti2i(
        eng->dit, eng->vae, eng->text, eng->vision, eng->gpu, prompt,
        (const uint8_t *const *)rgba_in, iw_arr, ih_arr, n_images, &params,
        eng->turbo_count ? eng->turbo_sigmas : NULL, eng->turbo_count,
        &eng->sched, &rgba, error, error_size);
    qw_pipe_set_progress(NULL, NULL);
    for (uint32_t i = 0; i < n_images; i++) free(rgba_in[i]);
    if (!ok) return 0;
    *has_alpha = serve_has_alpha(rgba, (size_t)params.width * params.height);
    serve_stage("decode", 1.0, 1);
    serve_stage("save", 0.0, 1);
    if (qw_png_write_rgba(out, params.width, params.height, rgba, error,
                          error_size)) {
        free(rgba);
        return 0;
    }
    free(rgba);
    serve_stage("save", 1.0, 1);
    return 1;
}

static void serve_handle(engine *eng, const char *dir, const qw_json *req) {
    char error[512] = {0};
    const char *rid = qw_json_get_string(req, "id", "");
    double t0 = now_seconds();

    pthread_mutex_lock(&g_serve.mu);
    snprintf(g_serve.rid, sizeof(g_serve.rid), "%s", rid);
    g_serve.started = t0;
    snprintf(g_serve.stage, sizeof(g_serve.stage), "encode");
    g_serve.percent = 0.0;
    g_serve.last_emit = 0.0;
    pthread_mutex_unlock(&g_serve.mu);

    const char *prompt = qw_json_get_string(req, "prompt", "");
    const char *negative = qw_json_get_string(req, "negativePrompt", "");
    const char *out = qw_json_get_string(req, "output", "");
    const char *lora = qw_json_get_string(req, "lora", "");
    uint32_t width = (uint32_t)qw_json_get_number(req, "width", 1024);
    uint32_t height = (uint32_t)qw_json_get_number(req, "height", 1024);
    uint32_t steps = (uint32_t)qw_json_get_number(req, "steps", 40);
    /* turbo 蒸馏模型只按官方固定 sample_sigmas 轨迹定义：请求里的 steps
     * 与 diffusers worker / Go 侧归一口径一致地被忽略。 */
    if (eng->turbo_count) steps = eng->turbo_count;
    uint64_t seed = (uint64_t)qw_json_get_number(req, "seed", 0);
    float cfg = (float)qw_json_get_number(req, "cfgScale", 1.0);
    float lora_scale = (float)qw_json_get_number(req, "loraScale", 1.0);
    /* Go 侧已限幅 0.05..1.5（同 worker.py 的钳制口径），双保险 */
    if (lora_scale < 0.05f) lora_scale = 0.05f;
    if (lora_scale > 1.5f) lora_scale = 1.5f;

    const qw_json *imgs = qw_json_get(req, "images");
    const char *images[MAX_TI2I_IMAGES] = {0};
    uint32_t n_images = 0;
    int bad = 0;
    if (imgs && qw_json_type(imgs) == QWEN_JSON_ARRAY) {
        size_t count = qw_json_count(imgs);
        if (count > MAX_TI2I_IMAGES) {
            snprintf(error, sizeof(error),
                     "at most %d reference images (got %zu)",
                     MAX_TI2I_IMAGES, count);
            bad = 1;
        } else {
            for (size_t i = 0; i < count; i++) {
                const qw_json *el = qw_json_at(imgs, i);
                if (qw_json_type(el) != QWEN_JSON_STRING) {
                    snprintf(error, sizeof(error),
                             "images[%zu] must be a string", i);
                    bad = 1;
                    break;
                }
                images[n_images++] = qw_json_string(el);
            }
        }
    }

    /* Sanity bounds: 0 slips through the %32 gate and huge values
     * would burn tens of GB in allocations before failing. The largest
     * released aspect bucket is 3520 px; 8192 is a generous ceiling. */
    if (width < 32 || width > 8192 || height < 32 || height > 8192) {
        snprintf(error, sizeof(error),
                 "width/height must be in [32, 8192] (got %ux%u)", width,
                 height);
        bad = 1;
    }
    int gate = !bad && prompt[0] && out[0] && width % 32 == 0 &&
               height % 32 == 0 && steps != 0;
    int sync_ok = 1;
    if (gate) {
        sync_ok = serve_sync_lora(eng, lora[0] ? lora : NULL, lora_scale,
                                  error, sizeof(error));
        if (getenv("QWEN_SERVE_MEM"))
            fprintf(stderr, "[serve][mem] after sync_lora footprint=%.2fGB\n",
                    phys_footprint_gb());
    }

    int has_alpha = 0;
    int ok = 0;
    if (bad) {
        /* error 已设置 */
    } else if (!prompt[0]) {
        snprintf(error, sizeof(error), "prompt is required");
    } else if (!out[0]) {
        snprintf(error, sizeof(error), "output path is required");
    } else if (width % 32 || height % 32) {
        snprintf(error, sizeof(error), "size must be multiples of 32");
    } else if (steps == 0) {
        snprintf(error, sizeof(error), "steps must be >= 1");
    } else if (steps > QWEN_STEPS_LIMIT) {
        snprintf(error, sizeof(error), "steps must be in [1, %u]",
                 QWEN_STEPS_LIMIT);
    } else if (!sync_ok) {
        /* error 已设置 */
    } else if (n_images) {
        serve_stage("encode", 0.0, 1);
        ok = serve_generate_ti2i(eng, dir, prompt, negative, width, height,
                                 steps, cfg, seed, out, images, n_images,
                                 &has_alpha, error, sizeof(error));
    } else {
        serve_stage("encode", 0.0, 1);
        ok = serve_generate_t2i(eng, prompt, negative, width, height, steps,
                                cfg, seed, out, &has_alpha, error,
                                sizeof(error));
    }

    /* rid 清空退场：心跳停发（worker.py finally 同款）。 */
    pthread_mutex_lock(&g_serve.mu);
    g_serve.rid[0] = 0;
    pthread_mutex_unlock(&g_serve.mu);

    if (ok) {
        char rid_esc[512], out_esc[1024];
        serve_json_escape(rid, rid_esc, sizeof(rid_esc));
        serve_json_escape(out, out_esc, sizeof(out_esc));
        serve_emit("{\"type\":\"done\",\"id\":\"%s\",\"path\":\"%s\","
                   "\"width\":%u,\"height\":%u,\"hasAlpha\":%s,"
                   "\"elapsed\":%.1f}",
                   rid_esc, out_esc, width, height,
                   has_alpha ? "true" : "false", now_seconds() - t0);
    } else {
        char msg[1024];
        serve_json_escape(error[0] ? error : "generation failed", msg,
                          sizeof(msg));
        serve_emit("{\"type\":\"error\",\"id\":\"%s\",\"message\":\"%s\"}",
                   rid, msg);
    }
    /* QWEN_SERVE_MEM=1: per-request memory line. live = h3 tensors still
     * referenced (a request-over-request climb means a tensor leak);
     * footprint = phys_footprint (grows monotonically with the Metal
     * allocator pool even without a leak — the 2026-10-28 finding). */
    if (getenv("QWEN_SERVE_MEM")) {
        h3_gpu_stats st;
        h3_gpu_get_stats(eng->gpu, &st);
        fprintf(stderr,
                "[serve][mem] live=%.2fGB alloc=%.2fGB peak=%.2fGB "
                "footprint=%.2fGB lora=%d\n",
                (double)st.live_bytes / 1e9, (double)st.allocated_bytes / 1e9,
                (double)st.peak_live_bytes / 1e9, phys_footprint_gb(),
                qw_dit_lora_count(eng->dit));
    }
    /* Request boundary: every MPSGraph in the shape caches would
     * otherwise be kept for the process lifetime (each new prompt
     * length / output size adds one, ~95 MB each). The next request
     * recompiles the handful of shapes it needs. */
    h3_gpu_clear_graph_cache(eng->gpu);
}

static int cmd_serve(const char *dir) {
    char error[512];
    serve_emit("{\"type\":\"state\",\"state\":\"loading\"}");
    double t0 = now_seconds();
    engine eng;
    memset(&eng, 0, sizeof(eng));
    if (!load_engine(dir, &eng, error, sizeof(error))) {
        fprintf(stderr, "error: %s\n", error);
        char msg[1024];
        serve_json_escape(error, msg, sizeof(msg));
        serve_emit("{\"type\":\"state\",\"state\":\"crashed\","
                   "\"message\":\"%s\"}",
                   msg);
        return 1;
    }
    fprintf(stderr, "serve: engine resident in %.1fs (%s)\n",
            now_seconds() - t0, dir);
    if (eng.turbo_count)
        fprintf(stderr, "serve: turbo schedule: %u fixed sigmas\n",
                (unsigned)eng.turbo_count);
    serve_emit("{\"type\":\"state\",\"state\":\"ready\",\"loadSeconds\":%.1f}",
               now_seconds() - t0);

    pthread_t beat;
    if (pthread_create(&beat, NULL, serve_heartbeat, NULL) != 0) {
        fprintf(stderr, "error: cannot start heartbeat thread\n");
        return 1;
    }

    char line[65536];
    while (fgets(line, sizeof(line), stdin)) {
        size_t len = strlen(line);
        /* fgets silently splits an over-long request into two "lines":
         * the first half fails to parse, the second half gets treated as
         * an independent request. Detect the truncation, drain the
         * remainder so the stream stays line-aligned, and answer with an
         * error event instead of leaving the host waiting on a done. */
        if (len == sizeof(line) - 1 && line[len - 1] != '\n') {
            int ch;
            while ((ch = fgetc(stdin)) != EOF && ch != '\n') {}
            fprintf(stderr, "bad request line: exceeds %zu bytes\n",
                    sizeof(line) - 1);
            serve_emit("{\"type\":\"error\",\"message\":"
                       "\"request line too long\"}");
            continue;
        }
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = 0;
        if (!len) continue;
        qw_json *req = NULL;
        char perr[128];
        if (!qw_json_parse(line, &req, perr, sizeof(perr))) {
            fprintf(stderr, "bad request line: %s\n", perr);
            serve_emit("{\"type\":\"error\",\"message\":"
                       "\"malformed request line\"}");
            continue;
        }
        if (strcmp(qw_json_get_string(req, "cmd", ""), "ping") == 0) {
            serve_emit("{\"type\":\"state\",\"state\":\"ready\"}");
        } else {
            serve_handle(&eng, dir, req);
        }
        qw_json_free(req);
    }
    fprintf(stderr, "serve: stdin closed — bye\n");
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    const char *command = NULL;
    int info = 0;
    generate_args gen;
    memset(&gen, 0, sizeof(gen));
    gen.cfg = 1.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--info") == 0 && !command) {
            info = 1;
            command = "--info";
        } else if (!command && (strcmp(argv[i], "generate") == 0 ||
                                strcmp(argv[i], "repl") == 0 ||
                                strcmp(argv[i], "serve") == 0)) {
            command = argv[i];
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            dir = argv[++i];
        } else if (command && strcmp(command, "generate") == 0) {
            if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
                gen.prompt = argv[++i];
            } else if (strcmp(argv[i], "--negative") == 0 && i + 1 < argc) {
                gen.negative = argv[++i];
            } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
                if (!parse_u32_arg(argv[++i], &gen.width) ||
                    !valid_size(gen.width)) {
                    fprintf(stderr, "error: --width must be in [32, 8192]\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
                if (!parse_u32_arg(argv[++i], &gen.height) ||
                    !valid_size(gen.height)) {
                    fprintf(stderr, "error: --height must be in [32, 8192]\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
                if (!parse_u32_arg(argv[++i], &gen.steps) ||
                    gen.steps < 1 || gen.steps > QWEN_STEPS_LIMIT) {
                    fprintf(stderr, "error: --steps must be in [1, %u]\n",
                            QWEN_STEPS_LIMIT);
                    return 1;
                }
            } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
                gen.seed = strtoull(argv[++i], NULL, 10);
                gen.seed_set = 1;
            } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
                if (gen.n_images < MAX_TI2I_IMAGES)
                    gen.images[gen.n_images++] = argv[++i];
                else {
                    fprintf(stderr,
                            "warning: at most %d --image, ignoring %s\n",
                            MAX_TI2I_IMAGES, argv[++i]);
                }
            } else if (strcmp(argv[i], "--res") == 0 && i + 1 < argc) {
                if (!parse_u32_arg(argv[++i], &gen.output_res) ||
                    (gen.output_res &&
                     (gen.output_res < 128 || gen.output_res > 4096))) {
                    fprintf(stderr,
                            "error: --res must be in [128, 4096]"
                            " (0 = default)\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
                if (gen.n_lora < MAX_LORA)
                    gen.lora[gen.n_lora++] = argv[++i];
                else {
                    fprintf(stderr,
                            "warning: at most %d --lora, ignoring %s\n",
                            MAX_LORA, argv[++i]);
                }
            } else if (strcmp(argv[i], "--lora-scale") == 0 && i + 1 < argc) {
                float scale;
                if (gen.n_scale < MAX_LORA) {
                    if (!parse_float_arg(argv[++i], &scale)) {
                        fprintf(stderr,
                                "error: --lora-scale needs a number\n");
                        return 1;
                    }
                    gen.lora_scale[gen.n_scale++] = scale;
                } else {
                    fprintf(stderr,
                            "warning: at most %d --lora-scale, ignoring"
                            " %s\n", MAX_LORA, argv[++i]);
                }
            } else if (strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) {
                if (!parse_float_arg(argv[++i], &gen.cfg)) {
                    fprintf(stderr, "error: --cfg needs a number\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                gen.out = argv[++i];
            } else {
                fprintf(stderr, "unknown argument: %s\n", argv[i]);
                usage();
                return 1;
            }
        } else if (command && strcmp(command, "repl") == 0) {
            if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
                if (!parse_u32_arg(argv[++i], &gen.width) ||
                    !valid_size(gen.width)) {
                    fprintf(stderr, "error: --width must be in [32, 8192]\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
                if (!parse_u32_arg(argv[++i], &gen.height) ||
                    !valid_size(gen.height)) {
                    fprintf(stderr, "error: --height must be in [32, 8192]\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
                if (!parse_u32_arg(argv[++i], &gen.steps) ||
                    gen.steps < 1 || gen.steps > QWEN_STEPS_LIMIT) {
                    fprintf(stderr, "error: --steps must be in [1, %u]\n",
                            QWEN_STEPS_LIMIT);
                    return 1;
                }
            } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
                gen.seed = strtoull(argv[++i], NULL, 10);
                gen.seed_set = 1;
            } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
                if (gen.n_images < MAX_TI2I_IMAGES)
                    gen.images[gen.n_images++] = argv[++i];
                else {
                    fprintf(stderr,
                            "warning: at most %d --image, ignoring %s\n",
                            MAX_TI2I_IMAGES, argv[++i]);
                }
            } else if (strcmp(argv[i], "--res") == 0 && i + 1 < argc) {
                if (!parse_u32_arg(argv[++i], &gen.output_res) ||
                    (gen.output_res &&
                     (gen.output_res < 128 || gen.output_res > 4096))) {
                    fprintf(stderr,
                            "error: --res must be in [128, 4096]"
                            " (0 = default)\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
                if (gen.n_lora < MAX_LORA)
                    gen.lora[gen.n_lora++] = argv[++i];
                else {
                    fprintf(stderr,
                            "warning: at most %d --lora, ignoring %s\n",
                            MAX_LORA, argv[++i]);
                }
            } else if (strcmp(argv[i], "--lora-scale") == 0 && i + 1 < argc) {
                float scale;
                if (gen.n_scale < MAX_LORA) {
                    if (!parse_float_arg(argv[++i], &scale)) {
                        fprintf(stderr,
                                "error: --lora-scale needs a number\n");
                        return 1;
                    }
                    gen.lora_scale[gen.n_scale++] = scale;
                } else {
                    fprintf(stderr,
                            "warning: at most %d --lora-scale, ignoring"
                            " %s\n", MAX_LORA, argv[++i]);
                }
            } else if (strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) {
                if (!parse_float_arg(argv[++i], &gen.cfg)) {
                    fprintf(stderr, "error: --cfg needs a number\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                gen.out = argv[++i];
            } else {
                fprintf(stderr, "unknown argument: %s\n", argv[i]);
                usage();
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            usage();
            return 0;
        } else if (strcmp(argv[i], "--version") == 0) {
            printf("qwen %s\n", QWEN_VERSION);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage();
            return 1;
        }
    }

    if (info) {
        if (!dir) {
            fprintf(stderr, "--info requires -d MODEL_DIR\n");
            return 1;
        }
        qw_model_config config;
        char error[512];
        if (!qw_model_scan(dir, &config, error, sizeof(error))) {
            fprintf(stderr, "error: %s\n", error);
            return 1;
        }
        qw_model_print(&config);
        return 0;
    }
    if (command && strcmp(command, "generate") == 0) {
        if (!dir) {
            fprintf(stderr, "generate requires -d MODEL_DIR\n");
            return 1;
        }
        if (!gen.prompt || !*gen.prompt) {
            fprintf(stderr, "generate requires -p PROMPT\n");
            return 1;
        }
        return cmd_generate(dir, &gen);
    }
    if (command && strcmp(command, "repl") == 0) {
        if (!dir) {
            fprintf(stderr, "repl requires -d MODEL_DIR\n");
            return 1;
        }
        return cmd_repl(dir, &gen);
    }
    if (command && strcmp(command, "serve") == 0) {
        if (!dir) {
            fprintf(stderr, "serve requires -d MODEL_DIR\n");
            return 1;
        }
        return cmd_serve(dir);
    }
    usage();
    printf("\nStart an interactive session with: qwen repl -d MODEL_DIR\n");
    return 1;
}
