#include "qwen_imgproc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void qw_imgproc_output_size(int in_w, int in_h, int output_res, int *out_w,
                            int *out_h) {
    /* Degenerate inputs used to reach (int)round(inf) (zero side -> inf
     * ratio) or zero outputs (extreme aspect at small res), which the
     * callers' `%32` gates cannot catch (0 % 32 == 0). Clamp instead of
     * producing UB. */
    if (in_w < 1) in_w = 1;
    if (in_h < 1) in_h = 1;
    if (output_res < 64) output_res = 1024; /* 0 = caller's default */
    if (output_res > 8192) output_res = 8192;
    double ratio = (double)in_w / (double)in_h;
    double width = sqrt((double)output_res * (double)output_res * ratio);
    double height = width / ratio;
    long w = lround(width / 32.0) * 32;
    long h = lround(height / 32.0) * 32;
    if (w < 32) w = 32;
    if (h < 32) h = 32;
    if (w > 8192) w = 8192;
    if (h > 8192) h = 8192;
    *out_w = (int)w;
    *out_h = (int)h;
}

/* PIL ImagingPaste 的 MULDIV255：round(a*b/255) 的整数近似 */
static inline int muldiv255(int a, int b) {
    int tmp = a * b + 128;
    return (((tmp >> 8) + tmp) >> 8);
}

void qw_imgproc_white_composite(const uint8_t *rgba, int pixels,
                                uint8_t *rgb) {
    /* PIL paste(mask): BLEND = MULDIV255(in1, 255-mask) + MULDIV255(in2, mask)，
     * in1 = 白底 */
    for (int i = 0; i < pixels; i++) {
        const uint8_t *src = rgba + (size_t)i * 4;
        int a = src[3];
        for (int c = 0; c < 3; c++)
            rgb[(size_t)i * 3 + c] = (uint8_t)(muldiv255(255, 255 - a) +
                                               muldiv255(src[c], a));
    }
}

/* ---- PIL ImagingResample 复刻（Lanczos, 8bpc 定点） ---- */

/* ---- PIL ImagingResample 复刻（Lanczos, 8bpc INT32 定点） ---- */

/* PIL: PRECISION_BITS = 32 - 8 - 2（系数定点 22 位） */
#define PIL_PRECISION_BITS (32 - 8 - 2)

static inline double lanczos_kernel(double x) {
    if (-3.0 <= x && x < 3.0) {
        double sx = x * M_PI;
        double s3 = x / 3.0 * M_PI;
        if (x == 0.0) return 1.0;
        return (sin(sx) / sx) * (sin(s3) / s3);
    }
    return 0.0;
}

/* 每输出位置：xmin 起始源索引、count 个 INT32 定点系数（PIL normalize_coeffs
 * 的 round-to-nearest：负权重向 -0.5、正权重向 +0.5） */
typedef struct {
    int ksize;
    int *xmin;  /* [out] */
    int *count; /* [out] */
    int32_t *coef; /* [out * ksize] */
} resample_coeffs;

static void coeffs_free(resample_coeffs *c) {
    free(c->xmin);
    free(c->count);
    free(c->coef);
    memset(c, 0, sizeof(*c));
}

/* PIL precompute_coeffs（in0=0, in1=inSize）+ normalize_coeffs_8bpc */
static int coeffs_build(int in_size, int out_size, resample_coeffs *c) {
    if (in_size < 1 || out_size < 1 || in_size > 65536) return 0;
    double scale = (double)in_size / (double)out_size; /* PIL: in/out */
    /* PIL: filterscale = scale < 1.0 ? 1.0 : scale（downscale 扩 support） */
    double filterscale = scale < 1.0 ? 1.0 : scale;
    double support = 3.0 * filterscale;
    int ksize = (int)ceil(support) * 2 + 1;
    double inv_filterscale = 1.0 / filterscale;
    c->ksize = ksize;
    c->xmin = calloc((size_t)out_size, sizeof(int));
    c->count = calloc((size_t)out_size, sizeof(int));
    c->coef = calloc((size_t)out_size * (size_t)ksize, sizeof(int32_t));
    /* Heap-allocated tap window: a fixed stack array capped the usable
     * downscale ratio (count > 1024 would smash the stack). */
    double *w = malloc(sizeof(double) * (size_t)ksize);
    if (!c->xmin || !c->count || !c->coef || !w) {
        free(w);
        coeffs_free(c);
        return 0;
    }
    for (int xx = 0; xx < out_size; xx++) {
        double center = (xx + 0.5) * scale;
        /* (int) 向零截断（PIL 原样） */
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        int count = xmax - xmin;
        double ww = 0.0;
        for (int x = 0; x < count; x++) {
            w[x] = lanczos_kernel((x + xmin - center + 0.5) *
                                  inv_filterscale);
            ww += w[x];
        }
        if (ww != 0.0) {
            for (int x = 0; x < count; x++) w[x] /= ww;
        }
        c->xmin[xx] = xmin;
        c->count[xx] = count;
        for (int x = 0; x < count; x++) {
            double q = w[x] * (double)(1 << PIL_PRECISION_BITS);
            int32_t iv = (int32_t)(q < 0 ? q - 0.5 : q + 0.5);
            c->coef[(size_t)xx * ksize + x] = iv;
        }
    }
    free(w);
    return 1;
}

/* PIL 8bpc：acc = 1 << (PRECISION_BITS-1) 起始、算术右移后查表 clamp */
static inline uint8_t clip8(int32_t acc) {
    int32_t v = acc >> PIL_PRECISION_BITS;
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

/* 水平重采样：src [sh][sw][sc] 行内连续 -> dst [sh][dw][sc] */
static void resample_hpass(const uint8_t *src, int sw, int sh, int sc,
                           uint8_t *dst, int dw, const resample_coeffs *c) {
    int32_t bias = 1 << (PIL_PRECISION_BITS - 1);
    for (int y = 0; y < sh; y++) {
        const uint8_t *srow = src + (size_t)y * (size_t)sw * (size_t)sc;
        uint8_t *drow = dst + (size_t)y * (size_t)dw * (size_t)sc;
        for (int xx = 0; xx < dw; xx++) {
            int xmin = c->xmin[xx];
            int count = c->count[xx];
            const int32_t *k = c->coef + (size_t)xx * c->ksize;
            for (int ch = 0; ch < sc; ch++) {
                int32_t acc = bias;
                for (int x = 0; x < count; x++)
                    acc += (int32_t)srow[(size_t)(x + xmin) * sc + ch] *
                           k[x];
                drow[(size_t)xx * sc + ch] = clip8(acc);
            }
        }
    }
}

/* 垂直重采样：src [sh][dw][sc] -> dst [dh][dw][sc]（源列内跨行） */
static void resample_vpass(const uint8_t *src, int sh, int dw, int sc,
                           const resample_coeffs *c, uint8_t *dst, int dh) {
    (void)sh; /* source rows are implied by the coeffs' xmin/count */
    int32_t bias = 1 << (PIL_PRECISION_BITS - 1);
    size_t row = (size_t)dw * (size_t)sc;
    for (int y = 0; y < dh; y++) {
        int ymin = c->xmin[y];
        int count = c->count[y];
        const int32_t *k = c->coef + (size_t)y * c->ksize;
        uint8_t *drow = dst + (size_t)y * row;
        for (int x = 0; x < dw; x++) {
            const uint8_t *col = src + (size_t)x * (size_t)sc;
            for (int ch = 0; ch < sc; ch++) {
                int32_t acc = bias;
                for (int s = 0; s < count; s++)
                    acc += (int32_t)col[(size_t)(s + ymin) * row +
                                        (size_t)ch] *
                           k[s];
                drow[(size_t)x * sc + ch] = clip8(acc);
            }
        }
    }
}

int qw_imgproc_resize_lanczos_u8(const uint8_t *src, int sw, int sh, int sc,
                                 uint8_t *dst, int dw, int dh) {
    resample_coeffs hc, vc;
    memset(&hc, 0, sizeof(hc));
    memset(&vc, 0, sizeof(vc));
    if (!coeffs_build(sw, dw, &hc)) return 0;
    if (!coeffs_build(sh, dh, &vc)) {
        coeffs_free(&hc);
        return 0;
    }
    uint8_t *tmp = malloc((size_t)sh * (size_t)dw * (size_t)sc);
    if (!tmp) {
        coeffs_free(&hc);
        coeffs_free(&vc);
        return 0;
    }
    resample_hpass(src, sw, sh, sc, tmp, dw, &hc);
    resample_vpass(tmp, sh, dw, sc, &vc, dst, dh);
    free(tmp);
    coeffs_free(&hc);
    coeffs_free(&vc);
    return 1;
}

/* RGBA 的 PIL resize：premultiply -> LANCZOS -> unpremultiply
 * （Pillow 对 RGBA/LA 的非 NEAREST 缩放内部转 RGBa/LA 处理） */
int qw_imgproc_resize_rgba_u8(const uint8_t *src, int sw, int sh,
                              uint8_t *dst, int dw, int dh) {
    size_t n = (size_t)sw * (size_t)sh;
    uint8_t *pre = malloc(n * 4);
    uint8_t *tmp = malloc((size_t)dw * (size_t)dh * 4);
    if (!pre || !tmp) {
        free(pre);
        free(tmp);
        return 0;
    }
    /* premultiply: r' = MULDIV255(r, a) */
    for (size_t i = 0; i < n; i++) {
        const uint8_t *p = src + i * 4;
        int a = p[3];
        pre[i * 4 + 0] = (uint8_t)muldiv255(p[0], a);
        pre[i * 4 + 1] = (uint8_t)muldiv255(p[1], a);
        pre[i * 4 + 2] = (uint8_t)muldiv255(p[2], a);
        pre[i * 4 + 3] = p[3];
    }
    if (!qw_imgproc_resize_lanczos_u8(pre, sw, sh, 4, tmp, dw, dh)) {
        free(pre);
        free(tmp);
        return 0;
    }
    /* unpremultiply（PIL rgba2rgbA）：(255*c)/alpha 截断除法，无偏置；
     * alpha==255 || alpha==0 直接拷贝，CLIP8 钳位 */
    size_t m = (size_t)dw * (size_t)dh;
    for (size_t i = 0; i < m; i++) {
        const uint8_t *p = tmp + i * 4;
        int a = p[3];
        uint8_t *o = dst + i * 4;
        o[3] = p[3];
        if (a == 255 || a == 0) {
            o[0] = p[0];
            o[1] = p[1];
            o[2] = p[2];
        } else {
            int r = (255 * (int)p[0]) / a;
            int g = (255 * (int)p[1]) / a;
            int b = (255 * (int)p[2]) / a;
            o[0] = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
            o[1] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
            o[2] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
        }
    }
    free(pre);
    free(tmp);
    return 1;
}

void qw_imgproc_patchify(const uint8_t *rgb, int gh, int gw, float *out) {
    int w = gw * 16;
    int t = 0;
    for (int br = 0; br < gh / 2; br++) {
        for (int bc = 0; bc < gw / 2; bc++) {
            for (int ir = 0; ir < 2; ir++) {
                for (int ic = 0; ic < 2; ic++, t++) {
                    int idx = 0;
                    /* 官方 pixel_values 的段序 [c3][t2][16][16]：temporal
                     * 两帧同为该帧（单图复制），VISION_PATCH_IN=1536 全写。 */
                    for (int c = 0; c < 3; c++)
                        for (int tt = 0; tt < 2; tt++)
                            for (int ty = 0; ty < 16; ty++)
                                for (int tx = 0; tx < 16; tx++, idx++) {
                                    int y = (br * 2 + ir) * 16 + ty;
                                    int x = (bc * 2 + ic) * 16 + tx;
                                    double v = (double)rgb[((size_t)y * w + x) *
                                                               3 + c];
                                    v = (v / 255.0 - 0.5) / 0.5;
                                    out[(size_t)t * 1536 + idx] = (float)v;
                                }
                }
            }
        }
    }
}

void qw_imgproc_vae_input(const uint8_t *rgba, int w, int h, float *out) {
    /* [H][W][4] 0-255 -> [-1,1] -> NHWC [H][W][C_pad]（VAE conv MM 内核
     * 的输入语义是 channels-last；C_pad 由调用方分配，至少 4）。 */
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 4; c++)
                out[((size_t)y * w + x) * 4 + c] =
                    (float)((double)rgba[((size_t)y * w + x) * 4 + c] /
                                127.5 -
                            1.0);
}
