"""ti2i fixture 生成器：官方 QwenImage21Pipeline 单步对拍量 → tests/fixtures/ti2i/。

覆盖本轮修复的全部回归点：模板 ids、文本塔 hidden（含视觉槽）、
imgproc_vae_input 的 NHWC 布局、VAE encode cond latents、DiT 单步 pred。
规模：输入/输出均 256x256，t=0.9140625（bf16(0.916)），单步（无采样）。

用法：
  /Volumes/Work/diffusers/.venv/bin/python tests/gen_ti2i_fixture.py
"""
import json
import os

import numpy as np
import torch
from PIL import Image

MODEL = "/Volumes/Work/models/Qwen-Image-2.1.models"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ti2i")
PROMPT = "把这只猫放到雪地里"
SIDE = 256  # 输入与输出边长
T_SCALE = 914.0625  # 官方 timestep 尺度（/1000 = 0.9140625 = bf16(0.916)）

os.makedirs(OUT, exist_ok=True)

from diffusers import QwenImage21Pipeline  # noqa: E402

# ---------- 合成图（确定性）：水平渐变 + 圆 ----------
yy, xx = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
r = np.sqrt((xx - 168) ** 2 + (yy - 100) ** 2)
circle = r < 56.0
img = np.zeros((SIDE, SIDE, 4), np.uint8)
img[..., 0] = (40 + xx * 0.6).astype(np.uint8)
img[..., 1] = (30 + yy * 0.5).astype(np.uint8)
img[..., 2] = (200 - xx * 0.3).astype(np.uint8)
img[..., 0][circle] = 220
img[..., 1][circle] = 120
img[..., 2][circle] = 60
img[..., 3] = 255
pil = Image.fromarray(img, "RGBA")
pil.save(os.path.join(OUT, "input.png"))

# ---------- 官方管线 ----------
pipe = QwenImage21Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.to("mps")

# ids + hidden（官方 _get_qwen_prompt_embeds 的 ti2i 路径复刻）
template = pipe.prompt_template_ti2i.format(PROMPT)
white = Image.new("RGB", pil.size, (255, 255, 255))
white.paste(pil, mask=pil.getchannel("A"))
model_inputs = pipe.processor(
    text=[template], images=[white], padding=True, padding_side="left",
    return_tensors="pt",
).to("mps")
drop = pipe._drop_idx
ids = model_inputs.input_ids[0].cpu().numpy().astype(np.uint32)
img_token_id = pipe._img_token_id

text_model = getattr(pipe.text_encoder.model, "language_model", pipe.text_encoder.model)
handle = text_model.norm.register_forward_hook(lambda m, a, o: a[0])
merged_cap = {}
def merger_hook(module, args, output):
    if "merged" not in merged_cap:
        t = output[0] if isinstance(output, tuple) else output
        merged_cap["merged"] = t.detach()
        if args and hasattr(args[0], "shape"):
            merged_cap["merger_in"] = args[0].detach()
    return output
mh = pipe.text_encoder.model.visual.merger.register_forward_hook(merger_hook)
vis_cap = {}
def pe_hook(module, args, output):
    if "pe" not in vis_cap:
        vis_cap["pe"] = output[0] if isinstance(output, tuple) else output
        vis_cap["pe"] = vis_cap["pe"].detach()
    return output
peh = pipe.text_encoder.model.visual.patch_embed.register_forward_hook(pe_hook)
def b0_hook(module, args, output):
    if "b0" not in vis_cap:
        vis_cap["b0"] = output[0] if isinstance(output, tuple) else output
        vis_cap["b0"] = vis_cap["b0"].detach()
    return output
b0h = pipe.text_encoder.model.visual.blocks[0].register_forward_hook(b0_hook)
for _bi in (8, 16, 24, 26):
    def _mk(idx):
        def _h(module, args, output):
            if idx not in vis_cap:
                t = output[0] if isinstance(output, tuple) else output
                vis_cap[idx] = t.detach()
            return output
        return _h
    pipe.text_encoder.model.visual.blocks[_bi].register_forward_hook(_mk(_bi))
forward_kwargs = dict(
    input_ids=model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    output_hidden_states=True,
    pixel_values=model_inputs.pixel_values,
    image_grid_thw=model_inputs.image_grid_thw,
)
if hasattr(model_inputs, "mm_token_type_ids"):
    forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids
with torch.no_grad():
    outputs = pipe.text_encoder(**forward_kwargs)
handle.remove()
mh.remove()
peh.remove()
b0h.remove()
merged_ref = merged_cap.get("merged")  # [slots, 4096]
if merged_ref is not None:
    print("merger out", tuple(merged_ref.shape))
hidden_full = outputs.hidden_states[-1][0]
hidden = hidden_full[model_inputs.attention_mask[0].bool()][drop:]  # [seq,4096]
seq = hidden.shape[0]
image_pad_row = (model_inputs.input_ids[0][drop:] == img_token_id)  # [seq]
print(f"ids={len(ids)} drop={drop} hidden={tuple(hidden.shape)} slots={int(image_pad_row.sum())}")

# vae_in：官方 preprocess（NHWC 与 C 端 imgproc_vae_input 对拍）
pre = pipe.image_processor.preprocess(pil, width=SIDE, height=SIDE)[0]  # [4,256,256]
vae_in = pre.float().cpu().numpy()

# cond：官方 _encode_vae_image（argmax=mode）+ packed
vae_images = pre[None].unsqueeze(2).to(pipe.vae.dtype).to("mps")  # [1,4,1,256,256]
with torch.no_grad():
    enc = pipe._encode_vae_image(vae_images, generator=None)  # [1,64,1,16,16] 已反归一化
cond = enc[0, :, 0].reshape(64, -1).T.float().cpu().numpy()  # [256,64] 行主 [h][w]
print("cond", cond.shape, "rms", round(float(np.sqrt((cond**2).mean())), 3))

# 初始 target 噪声（固定；不测 C 端 RNG）
rng = np.random.default_rng(20261029)
lat_init = rng.standard_normal((SIDE // 16 * SIDE // 16, 64)).astype(np.float32)

# ---------- DiT 单步 ----------
cond_packed = torch.from_numpy(cond).to("mps", torch.bfloat16)[None]
lat_packed = torch.from_numpy(lat_init).to("mps", torch.bfloat16)[None]
hidden_states_in = torch.cat([cond_packed, lat_packed], dim=1)  # [1,512,64] cond 前
img_shapes = [[(1, SIDE // 16, SIDE // 16), (1, SIDE // 16, SIDE // 16)]]
img_mask = torch.cat(
    [image_pad_row, image_pad_row.new_ones(lat_packed.shape[1] // 4)], dim=0
)[None]
timestep = torch.tensor([T_SCALE / 1000.0], device="mps",
                        dtype=torch.bfloat16)  # forward 语义是 [0,1] 的 sigma

captured = {}
joint_cap = {}
def blk0_hook(module, args, output):
    if "joint" not in joint_cap and args:
        joint_cap["joint"] = args[0].detach()
    return output
b0jh = pipe.transformer.transformer_blocks[0].register_forward_hook(blk0_hook)
def mod_hook(module, args, output):
    if "mod" not in captured:
        captured["mod"] = output.detach()
    return output

h1 = pipe.transformer.modulation.register_forward_hook(mod_hook)
with torch.no_grad():
    pred = pipe.transformer(
        hidden_states=hidden_states_in,
        timestep=timestep,
        encoder_hidden_states=hidden[None],
        encoder_hidden_states_mask=None,
        img_shapes=img_shapes,
        img_mask=img_mask,
        return_dict=False,
    )[0]
h1.remove()
b0jh.remove()
if "joint" in joint_cap:
    save_bf16(os.path.join(OUT, "joint_in.bin"), joint_cap["joint"][0])
    print("joint_in", tuple(joint_cap["joint"][0].shape))
print("pred", tuple(pred.shape), "rms", round(float(pred.float().pow(2).mean().sqrt()), 3))

# ---------- 落盘（bf16 以 uint16 位模式存，C 端直接读） ----------
def save_bf16(path, t):
    t = t.detach().to(torch.bfloat16).contiguous().cpu()
    t.view(torch.uint16).numpy().tofile(path)

def save_f32(path, a):
    np.ascontiguousarray(a, dtype="<f4").tofile(path)

save_bf16(os.path.join(OUT, "hidden.bin"), hidden)
if merged_ref is not None:
    save_bf16(os.path.join(OUT, "merged.bin"), merged_ref)
if "merger_in" in merged_cap:
    save_bf16(os.path.join(OUT, "merger_in.bin"), merged_cap["merger_in"])
for key, name in [("pe", "vis_patch_embed.bin"), ("b0", "vis_block0.bin")]:
    if key in vis_cap:
        t = vis_cap[key]
        t = t[0] if t.dim() == 3 else t
        save_bf16(os.path.join(OUT, name), t)
        print("vis", name, tuple(t.shape))
for _bi in (8, 16, 24, 26):
    if _bi in vis_cap:
        t = vis_cap[_bi]
        t = t[0] if t.dim() == 3 else t
        save_bf16(os.path.join(OUT, f"vis_block{_bi}.bin"), t)
        print("vis", f"block{_bi}", tuple(t.shape))
if hasattr(model_inputs, "pixel_values"):
    save_f32(os.path.join(OUT, "pixel_values.bin"),
             model_inputs.pixel_values.float().cpu().numpy())
    print("pixel_values", tuple(model_inputs.pixel_values.shape))
save_bf16(os.path.join(OUT, "vae_in.bin"), pre)
save_bf16(os.path.join(OUT, "modulation.bin"), captured["mod"][0])
save_f32(os.path.join(OUT, "cond.bin"), cond)
save_f32(os.path.join(OUT, "lat_init.bin"), lat_init)
save_f32(os.path.join(OUT, "pred.bin"), pred[0].float().cpu().numpy())
ids.tofile(os.path.join(OUT, "ids.bin"))

gh = SIDE // 16 // 2
meta = {
    "side": SIDE,
    "grid_hw": [gh, gh],
    "drop": int(drop),
    "seq": int(seq),
    "slots": int(image_pad_row.sum()),
    "cond_tokens": SIDE // 16 * SIDE // 16,
    "target_tokens": SIDE // 16 * SIDE // 16,
    "rows": int(seq - image_pad_row.sum().item() + 2 * int(image_pad_row.sum().item())),
    "sigma": T_SCALE / 1000.0,
    "prompt": PROMPT,
}
n_target = meta["target_tokens"]
meta["rows"] = meta["seq"] - meta["slots"] + meta["cond_tokens"] + n_target
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("rows", meta["rows"], "| fixture written to", OUT)
