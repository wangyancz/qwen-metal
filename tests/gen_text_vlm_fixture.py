#!/Volumes/Work/diffusers/.venv/bin/python
"""ti2i 文本塔 fixture 生成器（V1b 验收真值）。

构造一条带单图（grid 1×22×32，合并后 176 个视觉槽）的 ti2i 序列：
system + user（vision_start + 176×image_pad + vision_end + 提示词）+ assistant
骨架。视觉 embeds 与三级 deepstack 特征用合成 bf16（视觉塔本身已由
gen_vision_fixture 单独验收，这里只需给文本塔确定输入）。逐段落盘：
  text_vlm_input_embeds.bf16   scatter 后的 embeds [seq, 4096]
  text_vlm_pos_ids.i32         get_rope_index 的 [3, seq]（T/H/W 行主序）
  text_vlm_visual_mask.u8      [seq]（image_pad 位置为 1）
  text_vlm_deepstack.bf16      [3, 176, 4096]
  text_vlm_rope_cos/sin.f32    rotary 输出 [seq, 128]（bf16 量化后）
  text_vlm_layer{N}_after.bf16 层 0/1/2 的 DeepStack 注入后输出
  text_vlm_layer{6,16,35}.bf16 深层输出（层 35 = 管线消费的 pre-final-norm）
"""
import glob
import os

import numpy as np
import torch

MODEL = "/Volumes/Work/models/Qwen-Image-2.1.models"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_fixture")
GRID = (1, 22, 32)  # patch 化网格，spatial_merge 2 → 176 个视觉槽
PROMPT = "a red apple on a wooden table"


def main():
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLTextModel,
    )

    cfg = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder")
    tok = AutoTokenizer.from_pretrained(os.path.join(MODEL, "processor"))

    n_img = (GRID[1] // 2) * (GRID[2] // 2)
    user = ("<|vision_start|>" + "<|image_pad|>" * n_img + "<|vision_end|>" +
            PROMPT)
    full = ("<|im_start|>system\nComprehend and analyze the provided prompt."
            f"<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")
    ids = tok(full, add_special_tokens=False).input_ids
    n_pads = sum(1 for t in ids if t == cfg.image_token_id)
    assert n_pads == n_img, f"image_pad count {n_pads} != {n_img}"
    seq = len(ids)
    print(f"seq={seq}, visual slots={n_img}")

    # 文本塔加载（text encoder 分片的 model.language_model.* 键）
    model = Qwen3VLTextModel(cfg.text_config)
    state = {}
    for shard in sorted(glob.glob(f"{MODEL}/text_encoder/*.safetensors")):
        for k, v in safetensors_load(shard).items():
            if k.startswith("model.language_model."):
                state[k[len("model.language_model."):]] = v
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", list(missing)[:4], "unexpected:", list(unexpected)[:4])
    assert not missing, f"missing weights: {list(missing)[:8]}"
    model = model.to(torch.bfloat16).eval()

    # 输入构造：embed → scatter 合成视觉 embeds
    gen = torch.Generator().manual_seed(20261029)
    input_ids = torch.tensor([ids], dtype=torch.long)
    embeds = model.embed_tokens(input_ids)  # [1, seq, 4096]
    image_embeds = torch.randn(n_img, cfg.text_config.hidden_size,
                               generator=gen).to(torch.bfloat16)
    img_mask = input_ids[0] == cfg.image_token_id  # [seq]
    inputs_embeds = embeds.masked_scatter(
        img_mask[None, :, None], image_embeds[None])

    # get_rope_index（跳过 Qwen3VLModel 权重加载，方法只用 config）
    qm = Qwen3VLModel.__new__(Qwen3VLModel)
    qm.config = cfg
    mm_ids = img_mask.long()  # image=1 / text=0
    position_ids, deltas = qm.get_rope_index(
        input_ids, mm_ids[None],
        image_grid_thw=torch.tensor([GRID], dtype=torch.long))
    print("position_ids:", tuple(position_ids.shape),
          "delta:", deltas.item())
    assert position_ids.shape == (3, 1, seq)

    # 三级 deepstack 合成特征
    deepstack = [torch.randn(n_img, cfg.text_config.hidden_size,
                             generator=gen).to(torch.bfloat16)
                 for _ in range(3)]

    # forward + hook
    captured = {}

    def grab_rot(module, args, out):
        captured["rope_cos"] = out[0].detach().float().cpu().numpy()
        captured["rope_sin"] = out[1].detach().float().cpu().numpy()

    model.rotary_emb.register_forward_hook(grab_rot)
    watch = (0, 1, 2, 6, 16, 35)
    layer_raw = {}

    def make_grab(i):
        def grab(module, args, out):
            layer_raw[i] = out.detach().float().cpu().numpy()[0]
        return grab

    for i in watch:
        model.layers[i].register_forward_hook(make_grab(i))

    with torch.no_grad():
        out = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=img_mask[None],
            deepstack_visual_embeds=deepstack,
        )
    final = out.last_hidden_state[0].float().cpu().numpy()  # post-norm
    pre_final = layer_raw[35]  # 管线消费的 pre-final-norm（层 35 无注入）

    # 层 0/1/2 hook 是注入前，补上 deepstack（与 _deepstack_process 同式）
    vmask = img_mask.numpy()
    for i in (0, 1, 2):
        h = layer_raw[i].copy()
        h[vmask] += deepstack[i].float().cpu().numpy()
        np.asarray(h, dtype=np.float32).tofile(
            f"{OUT}/text_vlm_layer{i}_after.f32")

    def save_bf16(path, tensor):
        t = torch.as_tensor(tensor).detach().to(torch.bfloat16).contiguous()
        t.view(torch.uint16).numpy().tofile(path)

    save_bf16(f"{OUT}/text_vlm_input_embeds.bf16", inputs_embeds[0])
    np.asarray(position_ids[:, 0, :].numpy(), dtype=np.int32).tofile(
        f"{OUT}/text_vlm_pos_ids.i32")
    np.asarray(vmask, dtype=np.uint8).tofile(f"{OUT}/text_vlm_visual_mask.u8")
    save_bf16(f"{OUT}/text_vlm_deepstack.bf16", torch.stack(deepstack))
    np.asarray(captured["rope_cos"][0], dtype=np.float32).tofile(
        f"{OUT}/text_vlm_rope_cos.f32")
    np.asarray(captured["rope_sin"][0], dtype=np.float32).tofile(
        f"{OUT}/text_vlm_rope_sin.f32")
    for i in (6, 16, 35):
        np.asarray(layer_raw[i], dtype=np.float32).tofile(
            f"{OUT}/text_vlm_layer{i}.f32")
    np.asarray(pre_final, dtype=np.float32).tofile(
        f"{OUT}/text_vlm_final.f32")
    np.asarray(final, dtype=np.float32).tofile(
        f"{OUT}/text_vlm_final_postnorm.f32")

    manifest = {"seq": seq, "n_vis": n_img, "grid": list(GRID),
                "watch_layers": list(watch)}
    import json
    with open(f"{OUT}/text_vlm_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print("fixtures written to", OUT)


def safetensors_load(path):
    import safetensors.torch
    return safetensors.torch.load_file(path)


if __name__ == "__main__":
    main()
