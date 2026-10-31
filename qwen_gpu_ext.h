#ifndef QWEN_GPU_EXT_H
#define QWEN_GPU_EXT_H

/* Dispatch wrappers for the qw_* kernels appended to h3_shaders.metal.
 * These are qwen-metal additions; the h3_* API in h3_gpu.h is
 * unchanged vendored code. */

#include "h3_gpu.h"

/* Affine-free LayerNorm then per-row (1 + scale) modulation. */
int qw_gpu_layer_norm_mod_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *input,
                               const h3_gpu_tensor *modulation,
                               const h3_gpu_tensor *row_map, uint32_t rows,
                               uint32_t width, uint32_t slots,
                               uint32_t scale_slot, float epsilon);

/* RMSNorm whose effective weight is (weight + 1), F32 math. */
int qw_gpu_rms_norm_one_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight, uint32_t rows,
                             uint32_t width, float epsilon);

/* Elementwise tanh over BF16. */
int qw_gpu_tanh_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements);

/* Interleaved complex RoPE on [row, heads, head_dim] BF16 Q/K with F32
 * cos/sin tables of [row, head_dim / 2]. In-place safe. */
int qw_gpu_qk_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output,
                        h3_gpu_tensor *key_output,
                        const h3_gpu_tensor *query_input,
                        const h3_gpu_tensor *key_input,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim);

/* Half-split (rotate_half) RoPE: head element i pairs with element
 * i + head_dim / 2 (transformers text-tower convention), cos/sin tables
 * of [row, head_dim / 2]. In-place safe. */
int qw_gpu_qk_rope_half_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin, uint32_t sequence,
                             uint32_t heads, uint32_t head_dim);

/* RMSNorm with PyTorch CPU bf16 rounding points: normalized fp32 value
 * quantized to bf16 BEFORE the weight multiply (double rounding). */
int qw_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon);

/* Per-head RMSNorm with the same double rounding (Qwen3 q_norm/k_norm). */
int qw_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon);

/* SwiGLU over the fused fc1 gemm output with PyTorch CPU bf16 rounding
 * points (silu quantized to bf16, product quantized again); width is
 * the per-half MLP width. */
int qw_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fc1, uint32_t count,
                       uint32_t width);

/* Qwen VAE decoder (fp32, channels-last [H, W, C] activations). */

/* 3D convolution with explicit zero padding on all six NDHWC sides. */
int h3_gpu_conv3d_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t batch,
                          uint32_t depth, uint32_t height, uint32_t width,
                          uint32_t input_channels, uint32_t output_channels,
                          uint32_t kernel_depth, uint32_t kernel_height,
                          uint32_t kernel_width, uint32_t stride_depth,
                          uint32_t stride_height, uint32_t stride_width,
                          uint32_t pad_front, uint32_t pad_back,
                          uint32_t pad_top, uint32_t pad_bottom,
                          uint32_t pad_left, uint32_t pad_right);

/* Channel RMSNorm: x / max(||x||2, eps) * sqrt(dim) * gamma, one
 * threadgroup per row. */
int qw_gpu_vae_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *gamma, uint32_t rows,
                            uint32_t width, float scale, float epsilon);

/* Nearest-exact 2x spatial upsample. */
int qw_gpu_vae_nearest2x_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input, uint32_t height,
                             uint32_t width, uint32_t channels);

/* DupUp3D spatial shortcut for the image path (single frame,
 * first_chunk drops the first duplicated temporal frame). */
int qw_gpu_vae_dupup_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input, uint32_t height,
                         uint32_t width, uint32_t in_channels,
                         uint32_t out_channels, uint32_t factor_t,
                         uint32_t factor_s, uint32_t frame_index);

/* Split the fused [rows, 3C] qkv activation: parts 0/1 -> [rows, C],
 * part 2 -> transposed [C, rows]. */
int qw_gpu_vae_split3_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input, uint32_t rows,
                          uint32_t width, uint32_t part);

/* Row softmax over the full width; scale applied before max/sum (SDPA). */
int qw_gpu_vae_softmax_rows_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t width, float scale);

/* Flash-style online-softmax SDPA (bf16, fp32 accumulation): image
 * queries against every key, 64-query tiles. */
int qw_gpu_flash_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t q_rows,
                           uint32_t kv_rows, uint32_t heads,
                           uint32_t head_dim, float scale);

/* Prefix-visible variant: kv_limit[q_row] caps the key range per query
 * row (block-causal mask as a prefix; U32 tensor, q_rows elements,
 * values are non-negative key counts). kv_rows is the full key length
 * (buffer extent); limits <= kv_rows. Grouped-query attention:
 * heads % kv_heads == 0, q head h reads kv head h * kv_heads / heads
 * (pass heads for plain MHA). */
int qw_gpu_flash_sdpa_prefix_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *query,
                                  const h3_gpu_tensor *key,
                                  const h3_gpu_tensor *value,
                                  const h3_gpu_tensor *kv_limit,
                                  uint32_t q_rows, uint32_t kv_rows,
                                  uint32_t heads, uint32_t kv_heads,
                                  uint32_t head_dim, float scale);

/* [C, H, W] <-> [H, W, C] layout transposes. */
int qw_gpu_vae_chw_to_hwc_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input, uint32_t channels,
                              uint32_t height, uint32_t width);
int qw_gpu_vae_hwc_to_chw_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input, uint32_t channels,
                              uint32_t height, uint32_t width);

/* Direct fp32 3x3 convolution, zero padding, stride 1, channels-last. */
int qw_gpu_vae_conv3x3_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t height,
                           uint32_t width, uint32_t in_channels,
                           uint32_t out_channels);

/* GEMM-based fp32 3x3 convolution: 9-tap accumulating matmul2d over a
 * ring-padded copy of the activation. `w9` is the repacked
 * [9, CO_pad, CI] weight from qw_gpu_vae_conv_repack_f32. `pad_arena`
 * is a caller-owned scratch of >= pad_rows*in_channels floats
 * (vae_pad_elems), reused across convs. */
int qw_gpu_vae_conv_mm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *w9,
                           const h3_gpu_tensor *bias, uint32_t height,
                           uint32_t width, uint32_t in_channels,
                           uint32_t out_channels, h3_gpu_tensor *pad_arena);

/* One-time conv weight repack [CO, CI, 9] -> [9, CO_pad, CI]
 * (CO_pad = CO rounded up to 64). */
int qw_gpu_vae_conv_repack_f32(h3_gpu *gpu, h3_gpu_tensor *w9,
                               const h3_gpu_tensor *w, uint32_t in_channels,
                               uint32_t out_channels, uint32_t out_pad);

/* Elementwise add. */
int qw_gpu_vae_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                       uint32_t count);

/* 3x3 conv stride 2 (encoder downsampler), channels-last activations,
 * weight [oc, ic, 9]. */
int qw_gpu_vae_downsample_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input,
                              const h3_gpu_tensor *weight,
                              const h3_gpu_tensor *bias, uint32_t height,
                              uint32_t width, uint32_t in_channels,
                              uint32_t out_channels);

/* AvgDown3D single-frame spatial mean with channel remap
 * (factor_t semantics identical to the CPU reference). */
int qw_gpu_vae_avgdown_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, uint32_t height,
                           uint32_t width, uint32_t in_channels,
                           uint32_t out_channels, uint32_t factor_t);

/* Linear with the weight stored transposed [input_dim, output_dim]. */
int qw_gpu_vae_linear_t_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *bias, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim);

/* Fused q/k/v projection: one MPSGraph encode runs three matmuls that
 * share the input; weights keep their native [out, in] buffers. */
int qw_gpu_qkv_bf16(h3_gpu *gpu, h3_gpu_tensor *query_out,
                    h3_gpu_tensor *key_out, h3_gpu_tensor *value_out,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *query_weight,
                    const h3_gpu_tensor *key_weight,
                    const h3_gpu_tensor *value_weight, uint32_t rows,
                    uint32_t dim);

/* Causal GQA with the SDPA score-scaling convention (scale applied to
 * the fp32 scores after the dot product). */
int qw_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale);

/* Full-bidirectional SDPA whose query length may differ from the
 * key/value length (image-block rows against the full text+image prefix). */
int qw_gpu_sdpa_rect_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *query,
                          const h3_gpu_tensor *key,
                          const h3_gpu_tensor *value,
                          uint32_t query_sequence, uint32_t kv_sequence,
                          uint32_t heads, uint32_t head_dim, float scale);

/* ---- Qwen3-VL vision tower (bf16 activations, fp32 elementwise math) ---- */

/* Affine LayerNorm, one threadgroup per row (width <= 4608). */
int qw_gpu_vision_ln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon);

/* GELU (tanh approximation). */
int qw_gpu_vision_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);

/* Elementwise add (residual). */
int qw_gpu_vision_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *left,
                           const h3_gpu_tensor *right, uint32_t elements);

/* Split the row-fused [rows, 3C] qkv into one [rows, C] part
 * (part_index: 0=q 1=k 2=v). */
int qw_gpu_vision_split3_bf16(h3_gpu *gpu, h3_gpu_tensor *part,
                              const h3_gpu_tensor *qkv, uint32_t rows,
                              uint32_t channels, uint32_t part_index);

/* Concatenate every 4 consecutive tokens: [rows, C] -> [rows/4, 4C]. */
int qw_gpu_vision_merge2x2_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t channels);

/* Axial half-split RoPE with F32 cos/sin tables ([rows, head_dim/2]),
 * fp32 math + single bf16 output rounding (vision semantics, unlike the
 * text tower's quantized-table double rounding). */
int qw_gpu_vision_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *rope_cos,
                            const h3_gpu_tensor *rope_sin, uint32_t rows,
                            uint32_t heads, uint32_t head_dim);

#endif
