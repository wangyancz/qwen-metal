#!/Volumes/Work/diffusers/.venv/bin/python
"""Qwen3-VL 视觉塔 fixture 生成器（ti2i V1 验收真值）。

以 patch 化后的像素 hidden_states [seq, 3*2*16*16] + grid_thw 直接喂参考
Qwen3VLVisionModel（绕开 processor/图像预处理），逐段落盘 .f32：
  vision_input.f32            输入 [seq, 1536]
  vision_patch_embed.f32      Conv3d 输出 [seq, 1152]
  vision_block{NN}.f32        27 个 block 的输出 [seq, 1152]
  vision_merger.f32           merger 输出 [seq/4, 4096]
  vision_deepstack{i}.f32     block 8/16/24 经三级 merger 的输出 [seq/4, 4096]
  vision_rope_cos/sin.f32     轴向 rope [seq, 72]
manifest.json 记录全部形状。
"""
import json
import os

import numpy as np
import torch

MODEL = "/Volumes/Work/models/Qwen-Image-2.1.models"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_fixture")


def save_f32(path, arr):
    np.asarray(arr, dtype=np.float32).tofile(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder")
    vision = Qwen3VLVisionModel(cfg.vision_config)

    # 从 text_encoder 分片收集 model.visual.* 权重（全部在 shard 1）
    import glob
    import safetensors.torch
    shards = sorted(glob.glob(f"{MODEL}/text_encoder/*.safetensors"))
    state = {}
    for shard in shards:
        loaded = safetensors.torch.load_file(shard)
        for k, v in loaded.items():
            if k.startswith("model.visual."):
                state[k[len("model.visual."):]] = v
    missing, unexpected = vision.load_state_dict(state, strict=False)
    print("missing:", list(missing)[:4], "unexpected:", list(unexpected)[:4])
    vision = vision.to(torch.bfloat16).eval()

    # 合成输入：单图 grid (1, 22, 32) → seq = 704（merge 后 176）
    t, h, w = 1, 22, 32
    seq = h * w
    gen = torch.Generator().manual_seed(20261029)
    hidden = torch.randn(seq, 3 * 2 * 16 * 16, generator=gen).to(torch.bfloat16)
    grid = torch.tensor([[t, h, w]], dtype=torch.long)

    save_f32(os.path.join(OUT, "vision_input.f32"),
             hidden.float().numpy().reshape(-1))

    captured = {}

    def hook(name):
        def grab(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            captured[name] = tensor.detach().float().cpu().numpy().reshape(-1)
        return grab

    vision.patch_embed.register_forward_hook(hook("vision_patch_embed"))
    for i, blk in enumerate(vision.blocks):
        blk.register_forward_hook(hook(f"vision_block{i:02d}"))
    vision.merger.register_forward_hook(hook("vision_merger"))
    for i, m in enumerate(vision.deepstack_merger_list):
        m.register_forward_hook(hook(f"vision_deepstack{i}"))

    with torch.no_grad():
        out = vision(hidden_states=hidden, grid_thw=grid, return_dict=True)

    save_f32(os.path.join(OUT, "vision_merger.f32"),
             out.pooler_output.float().numpy().reshape(-1))
    for i, feature in enumerate(out.deepstack_features):
        save_f32(os.path.join(OUT, f"vision_deepstack{i}.f32"),
                 feature.float().numpy().reshape(-1))
    for name, arr in captured.items():
        save_f32(os.path.join(OUT, f"{name}.f32"), arr)

    # axial rope（未合并 704 token，block-major 坐标，与 vision_utils 一致）
    hpos, wpos = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    bh, bw = h // 2, w // 2
    hpos = hpos.reshape(bh, 2, bw, 2).transpose(0, 2, 1, 3).flatten()
    wpos = wpos.reshape(bh, 2, bw, 2).transpose(0, 2, 1, 3).flatten()
    inv = vision.rotary_pos_emb.inv_freq.float().numpy()  # [18]，theta=10000
    freq_h = hpos.astype(np.float64)[:, None] * inv[None, :]
    freq_w = wpos.astype(np.float64)[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(freq_h), np.cos(freq_w)], axis=-1)
    cos = np.concatenate([cos, cos], axis=-1)
    sin = np.concatenate([np.sin(freq_h), np.sin(freq_w)], axis=-1)
    sin = np.concatenate([sin, sin], axis=-1)
    save_f32(os.path.join(OUT, "vision_rope_cos.f32"), cos.reshape(-1))
    save_f32(os.path.join(OUT, "vision_rope_sin.f32"), sin.reshape(-1))

    # pos_embed 双线性插值的 indices/weights（参考实现预计算，C 端 parity 用）
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
    )
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, vision.num_grid_per_side, mode="bilinear", align_corners=True,
        spatial_merge_size=vision.spatial_merge_size)
    np.asarray(indices.numpy(), dtype=np.int64).tofile(
        os.path.join(OUT, "vision_interp_indices.i64"))
    save_f32(os.path.join(OUT, "vision_interp_weights.f32"),
             weights.float().numpy().reshape(-1))

    manifest = {
        "grid": [t, h, w], "seq": seq, "merged_seq": seq // 4,
        "hidden": 1152, "out": 4096,
        "deepstack_indexes": cfg.vision_config.deepstack_visual_indexes,
        "interp_taps": 4, "pos_side": 48,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("fixtures:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
