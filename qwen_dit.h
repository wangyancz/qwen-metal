#ifndef QWEN_DIT_H
#define QWEN_DIT_H

#include "h3_gpu.h"
#include "qwen_host.h"

#include <stddef.h>
#include <stdint.h>

/* Qwen-Image-2.1 DiT on the vendored Metal runtime. Weights stream from the
 * released checkpoint shards; every activation is BF16 with F32
 * accumulation, matching the released model's numerics.
 *
 * Modulation column layout (shared, [mod_rows, 4 * hidden]):
 *   slot 0: mod1.scale   slot 1: mod2.scale
 *   slot 2: mod1.gate (tanh applied)   slot 3: mod2.gate (tanh applied)
 * The tanh pre-pass makes the raw h3 gate kernel exact. */

typedef struct qw_dit qw_dit;

/* ti2i joint-sequence run: a text run (txt_in rows) or an image run
 * (img_in rows, condition blocks first and the target last). src_offset
 * is the run's starting row in text_in / latents (text runs skip the
 * VLM image-slot rows, which the joint overwrite replaces anyway). */
typedef struct {
    uint32_t rows;
    int is_image;
    uint32_t src_offset;
} qw_dit_segment;

qw_dit *qw_dit_create(h3_gpu *gpu, const char *model_dir, int layers,
                      char *error, size_t error_size);
void qw_dit_free(qw_dit *dit);

/* Drop the ti2i cond-KV caches. The cached rows encode ONE request's
 * prompt/images/weights — they must not survive the request even when
 * the next one has a matching run layout (the reuse check only matches
 * the layout, so a stale cache would silently serve the previous
 * request's conditioning). Call at the end of every ti2i generation. */
void qw_dit_ti2i_cache_clear(qw_dit *dit);

/* LoRA adapters (single safetensors files, rank-r A/B pairs named
 * "transformer_blocks.N.attn.to_q|.to_k|.to_v|.to_out.0|.img_mlp.gate_layer|
 * .img_mlp.proj|.img_mlp.out" + ".lora_A/.lora_B.default.weight" — the same
 * module names the checkpoint uses). Multiple adapters stack additively:
 * W' = RNE_bf16(fp32(W) + Σ_a scale_a · B_a@A_a) over every matched target;
 * missing targets are skipped with a note. Any change (apply / remove /
 * scale update) rebuilds the fused copies from the untouched base weights
 * in one pass — order-independent, no rounding accumulation. The base
 * weights stay referenced (mmap pages remain OS page cache), so rebuilds
 * need no weight I/O beyond the adapter payloads. Applying or removing
 * invalidates the ti2i step caches (K/V were produced by the old weights).
 * No lora_alpha is honoured (PEFT's default scaling = 1): scale is exactly
 * the caller's strength factor. A failed rebuild (OOM) resets the adapter
 * list to none and reports the cause. */

#define QW_DIT_MAX_LORA 8

/* Add an adapter, or update the scale of one already present (same
 * path): a new path appends to the stack and rebuilds; an existing
 * path just updates its scale and rebuilds. Fails without state
 * change when the file is unreadable, matches no checkpoint tensor,
 * or the stack is full. */
int qw_dit_apply_lora(qw_dit *dit, const char *path, float scale,
                      char *error, size_t error_size);
/* Remove the adapter with this path; NULL or empty removes every
 * adapter. Removing an unknown path is a no-op. */
int qw_dit_remove_lora(qw_dit *dit, const char *path,
                       char *error, size_t error_size);

/* Single-adapter switch: replaces the one active adapter with a new
 * file/scale, rebuilding the fused weights IN PLACE (the fused buffers
 * survive — no fresh allocation, unlike clear+apply). The new file is
 * validated first; a broken file leaves the old adapter active. */
int qw_dit_switch_lora(qw_dit *dit, const char *path, float scale,
                       char *error, size_t error_size);
/* Remove every adapter. */
int qw_dit_clear_lora(qw_dit *dit, char *error, size_t error_size);
/* Current adapter stack (empty when none). */
int qw_dit_lora_count(const qw_dit *dit);
const char *qw_dit_lora_path_at(const qw_dit *dit, int index);
float qw_dit_lora_scale_at(const qw_dit *dit, int index);
/* Diagnostic/test accessor: the live weight tensor of one block field
 * (0=to_q 1=to_k 2=to_v 3=to_out 4=fc1 5=fc2) — the fused copy while
 * adapters are applied, the base otherwise. */
const h3_gpu_tensor *qw_dit_block_tensor(const qw_dit *dit, int block,
                                         int field);

/* Joint conditioning: txt_in(text) prefixed to img_in(latents). */
int qw_dit_embed(qw_dit *dit, const h3_gpu_tensor *text_in,
                 uint32_t text_rows, const h3_gpu_tensor *latents,
                 uint32_t image_rows, h3_gpu_tensor *joint, char *error,
                 size_t error_size);

/* Host-side timestep features: each row of row_sigmas (a sigma value; the
 * causal_condition trailing row passes 0) expands to
 * [cos(1000 t f) | sin(1000 t f)] with f = 10000^(-i/128), i in 0..127. */
void qw_dit_timestep_features(const float *row_sigmas, size_t count,
                              float *features);

/* temb = TimestepEmbedder(features); then modulation slots (with the tanh
 * pre-pass) and the norm_out scale rows, both indexed by modulation row. */
int qw_dit_conditioning(qw_dit *dit, const h3_gpu_tensor *features,
                        uint32_t mod_rows, h3_gpu_tensor *temb,
                        h3_gpu_tensor *modulation,
                        h3_gpu_tensor *norm_out_scale, char *error,
                        size_t error_size);

/* One transformer block: modulation slots as documented above, row_map
 * selects the modulation row per token (target rows pick the timestep row,
 * the causal_condition text prefix picks the trailing t=0 row). Rows
 * [0, text_rows) form the causal text segment; the rest is one image
 * block attending to every key. debug_attn (optional) receives the merged
 * attention result before the output projection, for parity diagnostics.
 * kv_limit (ti2i): per-row prefix-visible key counts (U32, rows elements,
 * NULL keeps the t2i topology above). segs/seg_count (ti2i, required with
 * kv_limit on the fused path): the joint run layout — image runs go
 * through the MPSGraph fused SDPA as the exact rectangle [0, run_end),
 * text runs keep the flash prefix kernel. NULL falls back to the single
 * prefix kernel over all rows (QWEN_FLASH_ATTN=1 forces the fallback).
 * k_out/v_out: caller-owned [rows, hidden] K/V buffers to write instead
 * of the internal workspace (the ti2i step cache keeps them alive across
 * sampling steps); NULL uses the internal workspace. */
int qw_dit_block(qw_dit *dit, int layer, h3_gpu_tensor *hidden,
                 const h3_gpu_tensor *modulation,
                 const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
                 const h3_gpu_tensor *row_map, uint32_t rows,
                 uint32_t text_rows, const h3_gpu_tensor *kv_limit,
                 const qw_dit_segment *segs, uint32_t seg_count,
                 h3_gpu_tensor *k_out, h3_gpu_tensor *v_out,
                 h3_gpu_tensor *debug_attn, char *error, size_t error_size);

/* ti2i joint embed: text runs pull from text_in in order, image runs
 * from latents (packed row-major). segs give the joint row layout. */
int qw_dit_embed_segments(qw_dit *dit, const h3_gpu_tensor *text_in,
                          const h3_gpu_tensor *latents,
                          const qw_dit_segment *segs, uint32_t seg_count,
                          h3_gpu_tensor *joint, char *error,
                          size_t error_size);

/* Host-side block-causal-as-prefix table: text rows carry causal limits
 * (run start + offset + 1), image rows the run end (intra-block
 * bidirectional). limits has one entry per joint row. */
void qw_dit_prefix_limits(const qw_dit_segment *segs, uint32_t seg_count,
                          uint32_t *limits);

/* Final scale-only AdaLayerNorm on every row plus the 64-channel output
 * projection. Output carries all rows; callers slice the target rows. */
int qw_dit_final(qw_dit *dit, h3_gpu_tensor *hidden,
                 const h3_gpu_tensor *norm_out_scale,
                 const h3_gpu_tensor *row_map, uint32_t rows,
                 h3_gpu_tensor *output, char *error, size_t error_size);

/* Full denoising forward over all loaded layers: embeds the joint
 * sequence, builds the conditioning, runs every block, and applies the
 * final head. sigma selects the timestep row (the trailing modulation row
 * stays t = 0 under causal_condition). target_rows counts the image rows
 * appended after the text rows; output carries all rows. */
int qw_dit_forward(qw_dit *dit, const h3_gpu_tensor *text_in,
                   uint32_t text_rows, const h3_gpu_tensor *latents,
                   uint32_t image_rows, float sigma,
                   const h3_gpu_tensor *rope_cos,
                   const h3_gpu_tensor *rope_sin, h3_gpu_tensor *output,
                   char *error, size_t error_size);

/* ti2i forward: joint rows follow segs (text runs interleaved with
 * condition latent blocks and the target block last); kv_limit carries
 * the block-causal-as-prefix attention table; rope tables cover the
 * whole joint sequence. Only the target rows (the last image run) read
 * the sampled timestep; every other row modulates from t = 0. Output
 * carries all rows. cache_slot picks the cross-step K/V cache (0 = the
 * guided branch, 1 = the CFG negative branch — each keeps its own run
 * layout and K/V set). */
int qw_dit_forward_ti2i(qw_dit *dit, const h3_gpu_tensor *text_in,
                        const h3_gpu_tensor *latents,
                        const qw_dit_segment *segs, uint32_t seg_count,
                        uint32_t cache_slot,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, float sigma,
                        h3_gpu_tensor *output, char *error,
                        size_t error_size);

#endif
