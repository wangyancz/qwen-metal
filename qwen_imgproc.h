#ifndef QWEN_IMGPROC_H
#define QWEN_IMGPROC_H

#include <stdint.h>

/* ti2i 参考图预处理（V2）：复刻 diffusers QwenImage21Pipeline 的图像链。
 * 视觉路 = RGBA -> 白底合成 -> PIL LANCZOS resize -> rescale/normalize ->
 * patchify；VAE 路 = RGBA -> 同尺寸 resize -> [-1,1] CHW。
 * resize 走 PIL ImagingResample 的 8bpc 定点口径（INT16 系数），与
 * uint8 输入的 Pillow 逐位一致。 */

/* calculate_dimensions(output_res^2, w/h)：32 倍数取整 */
void qw_imgproc_output_size(int in_w, int in_h, int output_res, int *out_w,
                            int *out_h);

/* RGBA(0-255) -> RGB 白底合成（PIL paste(mask=A) 的 MULDIV255 语义），
 * 输出 n*3 字节 */
void qw_imgproc_white_composite(const uint8_t *rgba, int pixels,
                                uint8_t *rgb);

/* PIL Image.resize(LANCZOS) 的 8bpc 复刻（分离水平/垂直，支持任意通道数
 * 1-4，行主序像素交错）。失败（分配）返回 0。 */
int qw_imgproc_resize_lanczos_u8(const uint8_t *src, int sw, int sh, int sc,
                                 uint8_t *dst, int dw, int dh);

/* RGBA 的 PIL resize：alpha 模式内部先 premultiply（RGBa），缩放后
 * unpremultiply 回 RGBA。失败（分配）返回 0。 */
int qw_imgproc_resize_rgba_u8(const uint8_t *src, int sw, int sh,
                              uint8_t *dst, int dw, int dh);

/* 视觉塔输入：RGB(0-255) [H][W][3] -> 归一化 (x/255-.5)/.5 -> patchify
 * [gh*gw, 1536] f32；token 序 (gh/2, gw/2, merge_h, merge_w)，1536 维序
 * (C, T, ph, pw)，T 两帧复制。out 需 gh*gw*1536*4 字节。 */
void qw_imgproc_patchify(const uint8_t *rgb, int gh, int gw, float *out);

/* VAE 输入：RGBA [H][W][4] -> [-1,1] -> NHWC [H][W][4] f32（VAE conv MM
 * 内核的 channels-last 语义）。
 * out 需 4*h*w*4 字节。 */
void qw_imgproc_vae_input(const uint8_t *rgba, int w, int h, float *out);

#endif
