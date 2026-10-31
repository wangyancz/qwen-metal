/* Public API for the qwen-metal Qwen-Image-2.1 inference engine. */
#ifndef QWEN_H
#define QWEN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QWEN_VERSION "0.1.0-dev"
#define QWEN_DEFAULT_WIDTH 1024
#define QWEN_DEFAULT_HEIGHT 1024
#define QWEN_DEFAULT_STEPS 40

typedef struct qw_ctx qw_ctx;

#ifdef __cplusplus
}
#endif

#endif
