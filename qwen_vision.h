#ifndef QWEN_VISION_H
#define QWEN_VISION_H

#include <stddef.h>
#include <stdint.h>

#include "h3_gpu.h"

/* Qwen3-VL 视觉塔（ti2i 的参考图视觉上下文编码器）。
 * GPU bf16 路径：权重 bf16 上传、激活 bf16 常驻，GEMM 走 h3 linear
 *（MPSGraph 快路）、attention 走 MPSGraph SDPA（head_dim 参数化），
 * 逐点算（LN/GELU/RoPE/merge）fp32 数学 + bf16 单次舍入，与官方 bf16
 * 语义对齐。QWEN_VISION_CPU=1 回退 CPU fp32 参考实现。 */

typedef struct qw_vision qw_vision;

qw_vision *qw_vision_create(const char *text_encoder_dir, h3_gpu *gpu,
                            char *error, size_t error_size);
void qw_vision_free(qw_vision *v);

/* 逐段调试落盘目录（QWEN_VISION_DUMP 同款机制），供 fixture 验收。 */
void qw_vision_set_dump(qw_vision *v, const char *dir);

/* hidden: [seq, 1536] fp32（processor 预 patch 化，(c,t,kh,kw) 布局）；
 * grid_thw: [n_images, 3]（t,h,w，patch 网格）；merged_seq = Σ h*w/4。
 * merged: [merged_seq, 4096]；deepstack: [3][merged_seq, 4096] 连续
 * （block 8/16/24 输出经三级 deepstack merger）。 */
int qw_vision_forward(qw_vision *v, const float *hidden,
                      const uint32_t *grid_thw, uint32_t n_images,
                      float *merged, float *deepstack,
                      char *error, size_t error_size);

#endif
