#ifndef QWEN_PNG_H
#define QWEN_PNG_H

#include <stddef.h>
#include <stdint.h>

/* Minimal PNG writer for 8-bit RGBA images (filter 0 scanlines, zlib
 * deflate via libz). Returns 0 on success. */
int qw_png_write_rgba(const char *path, uint32_t width, uint32_t height,
                      const uint8_t *rgba, char *error, size_t error_size);

/* Minimal PNG reader: 8-bit non-interlaced RGB / RGBA / grayscale /
 * gray+alpha, expanded to RGBA. Returns 0 on success; caller frees
 * *rgba_out. */
int qw_png_read_rgba(const char *path, uint32_t *width, uint32_t *height,
                     uint8_t **rgba_out, char *error, size_t error_size);

#endif
