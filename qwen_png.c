#include "qwen_png.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

static uint32_t png_crc_table[256];
static int png_crc_ready;

static uint32_t png_crc(uint32_t crc, const uint8_t *data, size_t length) {
    if (!png_crc_ready) {
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            png_crc_table[n] = c;
        }
        png_crc_ready = 1;
    }
    crc ^= 0xffffffffu;
    for (size_t i = 0; i < length; i++)
        crc = png_crc_table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    return crc ^ 0xffffffffu;
}

static void png_be32(uint8_t *out, uint32_t value) {
    out[0] = (uint8_t)(value >> 24);
    out[1] = (uint8_t)(value >> 16);
    out[2] = (uint8_t)(value >> 8);
    out[3] = (uint8_t)value;
}

static int png_paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = p - a >= 0 ? p - a : a - p;
    int pb = p - b >= 0 ? p - b : b - p;
    int pc = p - c >= 0 ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

/* Apply one PNG row filter (0 None, 1 Sub, 2 Up, 3 Average, 4 Paeth).
 * row/prev point at the raw (unfiltered) rows; bpp = bytes per pixel. */
static void png_filter_row(int type, uint8_t *out, const uint8_t *row,
                           const uint8_t *prev, size_t stride, int bpp) {
    for (size_t i = 0; i < stride; i++) {
        int x = row[i];
        int left = i >= (size_t)bpp ? row[i - bpp] : 0;
        int up = prev ? prev[i] : 0;
        int upleft = prev && i >= (size_t)bpp ? prev[i - bpp] : 0;
        int value;
        switch (type) {
        case 1: value = x - left; break;
        case 2: value = x - up; break;
        case 3: value = x - (left + up) / 2; break;
        case 4: value = x - png_paeth(left, up, upleft); break;
        default: value = x; break;
        }
        out[i] = (uint8_t)value;
    }
}

/* Adaptive per-row filtering, the libpng/PIL default heuristic: try all
 * five filters, keep the one with the smallest sum of |signed byte|.
 * Purely an encoding choice — decoded pixels are identical either way.
 * The Up/Average/Paeth predictors reference the previous row's
 * RECONSTRUCTED bytes (the originals), so candidates must be computed
 * against an unfiltered copy, never the already-filtered output —
 * otherwise the file decodes to shifted pixels. */
static void png_filter_select(uint8_t *raw, size_t stride, uint32_t height) {
    uint8_t *tmp = malloc(stride);
    uint8_t *best = malloc(stride);
    uint8_t *orig = malloc(stride * height);
    if (!tmp || !best || !orig) {
        free(tmp);
        free(best);
        free(orig);
        return;  // fallback: rows keep filter 0
    }
    for (uint32_t y = 0; y < height; y++)
        memcpy(orig + (size_t)y * stride,
               raw + 1 + (size_t)y * (stride + 1), stride);
    const uint8_t *prev = NULL;
    for (uint32_t y = 0; y < height; y++) {
        uint8_t *row = raw + 1 + (size_t)y * (stride + 1);
        const uint8_t *src = orig + (size_t)y * stride;
        int best_type = 0;
        unsigned best_sum = ~0u;
        for (int type = 0; type < 5; type++) {
            png_filter_row(type, tmp, src, prev, stride, 4);
            unsigned sum = 0;
            for (size_t i = 0; i < stride; i++) {
                unsigned v = tmp[i];
                sum += v < 128 ? v : 256u - v;
            }
            if (sum < best_sum) {
                best_sum = sum;
                best_type = type;
                memcpy(best, tmp, stride);
            }
        }
        row[-1] = (uint8_t)best_type;
        memcpy(row, best, stride);
        prev = src;
    }
    free(tmp);
    free(best);
    free(orig);
}

static int png_write_chunk(FILE *file, const char *type, const uint8_t *data,
                           uint32_t length) {
    uint8_t header[8];
    png_be32(header, length);
    memcpy(header + 4, type, 4);
    if (fwrite(header, 1, 8, file) != 8) return -1;
    if (length && fwrite(data, 1, length, file) != length) return -1;
    uint32_t crc = png_crc(0, (const uint8_t *)type, 4);
    const uint8_t *cursor = data;
    size_t remaining = length;
    while (remaining) {
        size_t take = remaining < 4096 ? remaining : 4096;
        crc = png_crc(crc, cursor, take);
        cursor += take;
        remaining -= take;
    }
    uint8_t crc_bytes[4];
    png_be32(crc_bytes, crc);
    if (fwrite(crc_bytes, 1, 4, file) != 4) return -1;
    return 0;
}

int qw_png_write_rgba(const char *path, uint32_t width, uint32_t height,
                      const uint8_t *rgba, char *error, size_t error_size) {
    if (!path || !rgba || !width || !height ||
        width > 0x7fffffffu / 4u || height > 0x7fffffffu) {
        snprintf(error, error_size, "png: bad arguments");
        return -1;
    }
    size_t stride = (size_t)width * 4;
    size_t raw_size = (stride + 1) * height;
    uint8_t *raw = malloc(raw_size);
    if (!raw) {
        snprintf(error, error_size, "png: out of memory");
        return -1;
    }
    for (uint32_t y = 0; y < height; y++) {
        raw[y * (stride + 1)] = 0;
        memcpy(raw + y * (stride + 1) + 1, rgba + y * stride, stride);
    }
    png_filter_select(raw, stride, height);
    uLongf compressed_size = compressBound((uLong)raw_size);
    uint8_t *compressed = malloc(compressed_size);
    if (!compressed) {
        free(raw);
        snprintf(error, error_size, "png: out of memory");
        return -1;
    }
    int deflate_result =
        compress2(compressed, &compressed_size, raw, (uLong)raw_size, 6);
    free(raw);
    if (deflate_result != Z_OK) {
        free(compressed);
        snprintf(error, error_size, "png: deflate failed");
        return -1;
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        snprintf(error, error_size, "png: cannot open %s", path);
        free(compressed);
        return -1;
    }
    static const uint8_t signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    int failed = fwrite(signature, 1, 8, file) != 8;
    if (!failed) {
        uint8_t ihdr[13];
        png_be32(ihdr, width);
        png_be32(ihdr + 4, height);
        ihdr[8] = 8;
        ihdr[9] = 6;
        ihdr[10] = 0;
        ihdr[11] = 0;
        ihdr[12] = 0;
        failed = png_write_chunk(file, "IHDR", ihdr, 13) != 0;
    }
    if (!failed)
        failed = png_write_chunk(file, "IDAT", compressed,
                                 (uint32_t)compressed_size) != 0;
    if (!failed) failed = png_write_chunk(file, "IEND", NULL, 0) != 0;
    free(compressed);
    if (fclose(file) != 0) failed = 1;
    if (failed) {
        snprintf(error, error_size, "png: write failed: %s", path);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* PNG reader: 8-bit non-interlaced RGB/RGBA/gray/gray+alpha -> RGBA   */

static uint32_t png_parse_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}

int qw_png_read_rgba(const char *path, uint32_t *width, uint32_t *height,
                     uint8_t **rgba_out, char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        snprintf(error, error_size, "png: cannot open %s", path);
        return 1;
    }
    uint8_t sig[8];
    if (fread(sig, 1, 8, file) != 8 || memcmp(sig, "\x89PNG\r\n\x1a\n", 8)) {
        fclose(file);
        snprintf(error, error_size, "png: bad signature");
        return 1;
    }
    uint32_t w = 0, h = 0;
    int color_type = -1, bit_depth = -1;
    int have_ihdr = 0;
    uint8_t *idat = NULL;
    size_t idat_len = 0, idat_cap = 0;
    for (;;) {
        uint8_t header[8];
        if (fread(header, 1, 8, file) != 8) break;
        uint32_t len = png_parse_be32(header);
        char type[5] = {0};
        memcpy(type, header + 4, 4);
        /* Untrusted input: a crafted chunk length used to drive multi-GB
         * allocations before any validation ran. */
        if (len > 1u << 30) {
            free(idat);
            fclose(file);
            snprintf(error, error_size, "png: chunk too large");
            return 1;
        }
        uint8_t *data = malloc(len ? len : 1);
        if (!data || (len && fread(data, 1, len, file) != len)) {
            free(data);
            break;
        }
        uint8_t crc_bytes[4] = {0};
        if (fread(crc_bytes, 1, 4, file) != 4) {
            free(data);
            break;
        }
        uint32_t crc = png_crc(0, (const uint8_t *)type, 4);
        crc = png_crc(crc, data, len);
        if (crc != png_parse_be32(crc_bytes)) {
            free(data);
            free(idat);
            fclose(file);
            snprintf(error, error_size, "png: chunk CRC mismatch (%.4s)",
                     type);
            return 1;
        }
        if (!memcmp(type, "IHDR", 4)) {
            /* IHDR is exactly 13 bytes; shorter chunks previously read
             * past the allocation for bit depth / color type / interlace. */
            if (have_ihdr || len != 13) {
                free(data);
                free(idat);
                fclose(file);
                snprintf(error, error_size, "png: bad IHDR");
                return 1;
            }
            have_ihdr = 1;
            w = png_parse_be32(data);
            h = png_parse_be32(data + 4);
            bit_depth = data[8];
            color_type = data[9];
            if (data[12] != 0) {
                free(data);
                free(idat);
                fclose(file);
                snprintf(error, error_size, "png: interlaced unsupported");
                return 1;
            }
        } else if (!memcmp(type, "IDAT", 4)) {
            if (idat_len + len > (size_t)1 << 30) {
                free(data);
                free(idat);
                fclose(file);
                snprintf(error, error_size, "png: image data too large");
                return 1;
            }
            if (idat_len + len > idat_cap) {
                idat_cap = (idat_len + len) * 2 + 1024;
                uint8_t *grown = realloc(idat, idat_cap);
                if (!grown) {
                    free(data);
                    free(idat);
                    fclose(file);
                    snprintf(error, error_size, "png: out of memory");
                    return 1;
                }
                idat = grown;
            }
            memcpy(idat + idat_len, data, len);
            idat_len += len;
        } else if (!memcmp(type, "IEND", 4)) {
            free(data);
            break;
        }
        free(data);
    }
    fclose(file);
    if (!w || !h || color_type < 0 || bit_depth != 8 || !idat) {
        free(idat);
        snprintf(error, error_size,
                 "png: unsupported format (need 8-bit non-interlaced)");
        return 1;
    }
    /* Dimension cap: h * (w*channels+1) is a 64-bit product, so an IHDR
     * near 2^31 per side used to wrap the size (malloc(0) passed, the
     * unfilter loop below then walked the full stride and wrote far out
     * of bounds). 32768 per side is far above any real use and keeps
     * every later product comfortably inside 64 bits. */
    if (w > 32768u || h > 32768u) {
        free(idat);
        snprintf(error, error_size, "png: image too large (%ux%u)", w, h);
        return 1;
    }
    int channels = color_type == 0 ? 1
                 : color_type == 2 ? 3
                 : color_type == 4 ? 2
                 : color_type == 6 ? 4 : -1;
    if (channels < 0) {
        free(idat);
        snprintf(error, error_size, "png: unsupported color type %d",
                 color_type);
        return 1;
    }
    /* 64-bit or nothing: w * channels used to be a 32-bit product, so a
     * crafted IHDR with w >= 2^30 wrapped the size to a few bytes while
     * the unfilter loop below still walked the full stride. */
    uLongf raw_len = (uLongf)h * ((size_t)w * channels + 1);
    uLongf expected_len = raw_len;
    uint8_t *raw = malloc(raw_len);
    if (!raw) {
        free(idat);
        snprintf(error, error_size, "png: out of memory");
        return 1;
    }
    int zrc = uncompress(raw, &raw_len, idat, (uLong)idat_len);
    free(idat);
    if (zrc != Z_OK) {
        free(raw);
        snprintf(error, error_size, "png: zlib inflate failed (%d)", zrc);
        return 1;
    }
    /* A complete-but-short zlib stream returns Z_OK with raw_len shrunk
     * to the actual output; the unfilter walk below would then read the
     * uninitialized tail. */
    if (raw_len != expected_len) {
        free(raw);
        snprintf(error, error_size, "png: truncated image data");
        return 1;
    }
    /* unfilter in place */
    size_t stride = (size_t)w * channels;
    for (uint32_t y = 0; y < h; y++) {
        uint8_t filter = raw[y * (stride + 1)];
        uint8_t *line = raw + y * (stride + 1) + 1;
        for (size_t x = 0; x < stride; x++) {
            int a = x >= (size_t)channels ? line[x - channels] : 0;
            int b = y ? raw[(y - 1) * (stride + 1) + 1 + x] : 0;
            int c = (x >= (size_t)channels && y)
                        ? raw[(y - 1) * (stride + 1) + 1 + x - channels]
                        : 0;
            switch (filter) {
            case 1: line[x] = (uint8_t)(line[x] + a); break;
            case 2: line[x] = (uint8_t)(line[x] + b); break;
            case 3: line[x] = (uint8_t)(line[x] + (a + b) / 2); break;
            case 4: line[x] = (uint8_t)(line[x] + png_paeth(a, b, c)); break;
            default: break;
            }
        }
    }
    uint8_t *rgba = malloc((size_t)w * h * 4);
    if (!rgba) {
        free(raw);
        snprintf(error, error_size, "png: out of memory");
        return 1;
    }
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            const uint8_t *src =
                raw + y * (stride + 1) + 1 + (size_t)x * channels;
            uint8_t *dst = rgba + ((size_t)y * w + x) * 4;
            switch (channels) {
            case 1: dst[0] = dst[1] = dst[2] = src[0]; dst[3] = 255; break;
            case 2: dst[0] = dst[1] = dst[2] = src[0]; dst[3] = src[1]; break;
            case 3: dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
                    dst[3] = 255; break;
            default: memcpy(dst, src, 4); break;
            }
        }
    free(raw);
    *width = w;
    *height = h;
    *rgba_out = rgba;
    return 0;
}
