#ifndef QWEN_TEXT_ENCODER_H
#define QWEN_TEXT_ENCODER_H

#include "h3_gpu.h"
#include "qwen_host.h" /* text geometry constants */

#include <stddef.h>
#include <stdint.h>

#define QWEN_TEXT_HEADS QWEN_HEADS_TEXT
#define QWEN_TEXT_HEAD_DIM QWEN_HEAD_DIM_TEXT

/* Qwen3-VL language tower of the released Qwen-Image-2.1 text encoder:
 * 36 causal layers, hidden 4096, GQA 32/8 heads of 128, interleaved RoPE
 * (theta 5e6). The pipeline consumes the last decoder layer's output
 * BEFORE the final RMSNorm, so encode() never applies it.
 *
 * Prompt assembly mirrors the released pipeline: the raw template string
 * is tokenized directly (never through a chat-template renderer, which
 * tokenizes differently), and the system-role prefix tokens are dropped
 * from the resulting hidden states. */

#define QWEN_TEXT_SYS_PROMPT "Comprehend and analyze the provided prompt."
/* 官方 2.1 的 ti2i（edit_plus）管线沿用与 t2i 相同的 system 段——
 * fixture 对拍（tests/test_ti2i_fixture.c，drop==14、逐 id 一致）已证实。
 * 老版 EDIT prompt（"Describe the key features..."）是 Qwen-Image-Edit
 * 1.0 的口径，本引擎不使用。 */

typedef struct qw_text qw_text;

qw_text *qw_text_create(h3_gpu *gpu, const char *model_dir, int layers,
                        char *error, size_t error_size);
void qw_text_free(qw_text *text);

/* Causal encode of a token sequence. Output carries every position,
 * [seq, 4096] BF16, pre-final-norm. */
int qw_text_encode(qw_text *text, const uint32_t *token_ids, uint32_t seq,
                   h3_gpu_tensor *hidden, char *error, size_t error_size);

/* VLM (ti2i) encode: the embedding sequence already carries visual tokens
 * (image_pad slots scattered with the vision tower's merged output).
 * pos_ids is the interleaved mrope table [3, seq] row-major (T/H/W axes;
 * pure-text callers pass NULL for the degenerate 1-D rope). After decoder
 * layers 0/1/2 the deepstack features are added at visual positions
 * (deepstack_bf16: [3 * n_vis * 4096] host BF16 in sequence order,
 * visual_mask[s] != 0 marks a visual slot). Output [seq, 4096] BF16,
 * pre-final-norm, same as qw_text_encode. */
int qw_text_encode_vlm(qw_text *text, const uint16_t *embeds_bf16,
                       const int32_t *pos_ids, const uint8_t *visual_mask,
                       const uint16_t *deepstack_bf16, uint32_t seq,
                       uint32_t n_vis, h3_gpu_tensor *hidden, char *error,
                       size_t error_size);

/* Builds the t2i prompt template around the user text and tokenizes it
 * with the checkpoint tokenizer. Returns the token ids (caller frees with
 * h3_tokenizer_ids_free) and the number of leading system-prefix tokens
 * that must be dropped from the encoded hidden states. */
int qw_text_template_tokens(qw_text *text, const char *prompt,
                            uint32_t **ids, size_t *count, size_t *drop,
                            char *error, size_t error_size);

/* ti2i 模板（官方 QwenImage21Pipeline.prompt_template_ti2i）：system 段 +
 * user(<image1><|vision_start|><|image_pad|>×slots[0][ 空格 <image{i}>...]
 * <|vision_end|>prompt)<|im_end|>assistant\n。slots[i] = 第 i 图的合并
 * 视觉槽数。返回 token ids 与 drop（system 段 token 数，同 t2i）。 */
int qw_text_template_tokens_ti2i(qw_text *text, const char *prompt,
                                 uint32_t n_images, const uint32_t *grid_hw,
                                 uint32_t **ids, size_t *count, size_t *drop,
                                 uint32_t *image_pad_id, char *error,
                                 size_t error_size);

/* get_rope_index（单 batch、无 padding、N 图）：文本段三轴共用递增位置；
 * image_pad 连续段用 T=start、H=start+row、W=start+col 的 merged 网格
 * 行主序展开；位置推进 max(gh, gw)。pos_out [3, count]（T/H/W 三行）。 */
void qw_text_pos_ids(const uint32_t *ids, size_t count, uint32_t image_pad_id,
                     uint32_t n_images, const uint32_t *grid_hw,
                     int32_t *pos_out);

/* ti2i 全组装：tokenize → embed_tokens → scatter 视觉 merged embeds →
 * encode_core（mrope + deepstack 注入）。merged_bf16 [Σslots, 4096] 与
 * deepstack_bf16 [3, Σslots, 4096] 按图序拼接；grid_hw [n_images][2] 为
 * merged 网格。输出 hidden [seq, 4096] BF16（pre-final-norm）、
 * img_mask_out [seq]（1=视觉槽）。out_rows 是 hidden 的行数容量（同时
 * 也是 img_mask_out 的字节容量，容纳的是 drop 后的行）；序列超容量立即
 * 报错返回（调用方按 slots+prompt 字节上界分配，见 qwen_pipeline.m）。 */
int qw_text_encode_ti2i(qw_text *text, const char *prompt,
                        uint32_t n_images, const uint32_t *grid_hw,
                        const uint16_t *merged_bf16,
                        const uint16_t *deepstack_bf16,
                        h3_gpu_tensor *hidden, uint8_t *img_mask_out,
                        uint32_t out_rows, size_t *seq_out, char *error,
                        size_t error_size);

#endif
