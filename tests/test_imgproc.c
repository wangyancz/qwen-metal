/* ti2i 图像预处理验收（V2）：C 端链对比 PIL 参考链
 * （tests/gen_imgproc_fixture.py）。白底/VAE 变换应逐位一致；
 * Lanczos resize 按 PIL 8bpc 定点复刻；patchify 全链 1e-5 级。 */
#include "qwen_imgproc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_DIR "tests/vision_fixture"

static int failures = 0;

static float *read_f32(const char *path, size_t *count) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long bytes = ftell(file);
    rewind(file);
    *count = (size_t)bytes / 4;
    float *data = malloc((size_t)bytes);
    if (!data || fread(data, 4, *count, file) != *count) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    return data;
}

static void *read_file(const char *path, size_t *bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    void *data = malloc((size_t)size);
    if (!data || fread(data, 1, (size_t)size, file) != (size_t)size) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *bytes = (size_t)size;
    return data;
}

static void check(const char *name, const float *got, const float *ref,
                  size_t count, double gate) {
    double num = 0, den = 0;
    for (size_t i = 0; i < count; i++) {
        double diff = (double)got[i] - ref[i];
        num += diff * diff;
        den += (double)ref[i] * ref[i];
    }
    double rel = den > 0 ? sqrt(num / den) : 0.0;
    int ok = rel <= gate;
    if (!ok) failures++;
    printf("  %-16s relL2 %.3e  %s（门 %.0e）\n", name, rel,
           ok ? "PASS" : "FAIL", gate);
}

static void check_u8(const char *name, const uint8_t *got, const float *ref,
                     size_t count, double gate) {
    double num = 0, den = 0;
    for (size_t i = 0; i < count; i++) {
        double diff = (double)got[i] - ref[i];
        num += diff * diff;
        den += (double)ref[i] * ref[i];
    }
    double rel = den > 0 ? sqrt(num / den) : 0.0;
    int ok = rel <= gate;
    if (!ok) failures++;
    printf("  %-16s relL2 %.3e  %s（门 %.0e）\n", name, rel,
           ok ? "PASS" : "FAIL", gate);
}

int main(void) {
    size_t count = 0;
    float *rgba = read_f32(FIXTURE_DIR "/imgproc_input_rgba.f32", &count);
    size_t sbytes = 0;
    int32_t *size = read_file(FIXTURE_DIR "/imgproc_input_size.i32", &sbytes);
    if (!rgba || !size || sbytes / 4 != 6) {
        printf("imgproc fixture missing/mismatched\n");
        return 1;
    }
    int in_h = size[0], in_w = size[1];
    int up_h = size[2], up_w = size[3];
    int dn_h = size[4], dn_w = size[5];
    printf("input %dx%d -> up %dx%d dn %dx%d\n", in_w, in_h, up_w, up_h,
           dn_w, dn_h);
    int pixels = in_h * in_w;

    uint8_t *rgba_u8 = malloc((size_t)pixels * 4);
    for (int i = 0; i < pixels * 4; i++)
        rgba_u8[i] = (uint8_t)rgba[i];

    /* 1. 白底合成 */
    uint8_t *rgb = malloc((size_t)pixels * 3);
    qw_imgproc_white_composite(rgba_u8, pixels, rgb);
    float *ref_white = read_f32(FIXTURE_DIR "/imgproc_white_rgb.f32", &count);
    {
        double num = 0, den = 0;
        size_t n = (size_t)pixels * 3;
        for (size_t i = 0; i < n; i++) {
            double diff = (double)rgb[i] - ref_white[i];
            num += diff * diff;
            den += (double)ref_white[i] * ref_white[i];
        }
        double rel = den > 0 ? sqrt(num / den) : 0.0;
        int ok = rel == 0.0;
        if (!ok) failures++;
        printf("  %-16s relL2 %.3e  %s（逐位）\n", "white", rel,
               ok ? "PASS" : "FAIL");
    }
    free(ref_white);

    /* 2. Lanczos resize（up 与 dn 两档） */
    uint8_t *up = malloc((size_t)up_w * up_h * 3);
    uint8_t *dn = malloc((size_t)dn_w * dn_h * 3);
    if (!qw_imgproc_resize_lanczos_u8(rgb, in_w, in_h, 3, up, up_w, up_h) ||
        !qw_imgproc_resize_lanczos_u8(rgb, in_w, in_h, 3, dn, dn_w, dn_h)) {
        printf("resize alloc failed\n");
        return 1;
    }
    float *ref_up = read_f32(FIXTURE_DIR "/imgproc_vis_rgb_up.f32", &count);
    check_u8("resize_up", up, ref_up, (size_t)up_w * up_h * 3, 2e-3);
    free(ref_up);
    float *ref_dn = read_f32(FIXTURE_DIR "/imgproc_vis_rgb_dn.f32", &count);
    check_u8("resize_dn", dn, ref_dn, (size_t)dn_w * dn_h * 3, 2e-3);
    free(ref_dn);

    /* 3. VAE 路 resize（RGBA 4 通道，premultiply 链） */
    uint8_t *vae_up = malloc((size_t)up_w * up_h * 4);
    if (!qw_imgproc_resize_rgba_u8(rgba_u8, in_w, in_h, vae_up, up_w,
                                   up_h)) {
        printf("vae resize alloc failed\n");
        return 1;
    }
    float *ref_vae = read_f32(FIXTURE_DIR "/imgproc_vae_rgba_up.f32", &count);
    check_u8("vae_resize", vae_up, ref_vae, (size_t)up_w * up_h * 4, 2e-3);
    /* VAE [-1,1] 变换（qw_imgproc_vae_input 的输出语义是 channels-last
     * NHWC [H][W][4]——conv MM 内核的输入布局；参考同布局构造，应逐位）。 */
    float *vae_in = malloc(sizeof(float) * 4 * up_h * up_w);
    qw_imgproc_vae_input(vae_up, up_w, up_h, vae_in);
    {
        size_t n = (size_t)4 * up_h * up_w;
        float *ref_norm = malloc(sizeof(float) * n);
        for (int y = 0; y < up_h; y++)
            for (int x = 0; x < up_w; x++)
                for (int c = 0; c < 4; c++)
                    ref_norm[((size_t)y * up_w + x) * 4 + c] =
                        (float)((double)ref_vae[((size_t)y * up_w + x) * 4 +
                                                c] /
                                    127.5 -
                                1.0);
        check("vae_norm", vae_in, ref_norm, n, 1e-6);
        free(ref_norm);
    }
    free(ref_vae);

    /* 4. patchify 全链（up 主链） */
    int gh = up_h / 16, gw = up_w / 16;
    float *pv = malloc(sizeof(float) * (size_t)gh * gw * 1536);
    qw_imgproc_patchify(up, gh, gw, pv);
    float *ref_pv = read_f32(FIXTURE_DIR "/imgproc_pixel_values.f32", &count);
    check("pixel_values", pv, ref_pv, (size_t)gh * gw * 1536, 2e-3);
    free(ref_pv);

    /* 5. 尺寸推导 */
    int ow, oh;
    qw_imgproc_output_size(in_w, in_h, 1024, &ow, &oh);
    int ok = ow == up_w && oh == up_h;
    if (!ok) failures++;
    printf("  %-16s %dx%d  %s\n", "output_size", ow, oh,
           ok ? "PASS" : "FAIL");

    free(rgba);
    free(size);
    free(rgba_u8);
    free(rgb);
    free(up);
    free(dn);
    free(vae_up);
    free(vae_in);
    free(pv);
    if (failures == 0) {
        printf("imgproc parity passed\n");
        return 0;
    }
    printf("%d failure(s)\n", failures);
    return 1;
}
