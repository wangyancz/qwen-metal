#!/Volumes/Work/diffusers/.venv/bin/python
"""ti2i 图像预处理 fixture 生成器（V2 验收真值）。

复刻 diffusers QwenImage21Pipeline 的两条图像链：
  视觉路: RGBA -> 白底合成 RGB -> VaeImageProcessor.resize (PIL LANCZOS)
          -> Qwen3VL processor（rescale 1/255 + normalize (x-.5)/.5 + patchify
             [seq, 3*2*16*16]，token 序 (gh/2, gw/2, merge_h, merge_w)，
             1536 维序 (C, T, ph, pw)，T 两帧复制）
  VAE 路: RGBA 原样 -> 同尺寸 resize -> [-1,1] -> [1,4,1,H,W]
尺寸: calculate_dimensions(output_res^2, w/h) -> 32 倍数。

落盘（tests/vision_fixture/imgproc_*）：
  imgproc_input_rgba.f32      原图 RGBA [H*W*4]（0-255 浮点）
  imgproc_input_size.i32      [H, W, out_H_up, out_W_up, out_H_dn, out_W_dn]
  imgproc_white_rgb.f32       白底合成后 [H*W*3]
  imgproc_vis_rgb_up.f32      上采样链 resize 后 RGB
  imgproc_vis_rgb_dn.f32      下采样链 resize 后 RGB
  imgproc_vae_rgba_up.f32     VAE 路 resize 后 RGBA
  imgproc_pixel_values.f32    视觉塔最终输入 [seq, 1536]（主链，up 尺寸）
  imgproc_grid.i32            [1, gh, gw]
"""
import json
import math
import os

import numpy as np
from PIL import Image

MODEL = "/Volumes/Work/models/Qwen-Image-2.1.models"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_fixture")
PROCESSOR_DIR = os.path.join(MODEL, "processor")
OUTPUT_RES = 1024


def make_test_image():
    """合成 RGBA 测试图 512x384：渐变 + 形状 + 半透明区域。"""
    w, h = 512, 384
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = (xx / w * 255)
    g = (yy / h * 255)
    b = (np.sin(xx / 37.0) * 0.5 + 0.5) * 255
    a = np.full((h, w), 255.0)
    # 中央半透明圆
    dist = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    circle = np.clip((110 - dist) / 40.0, 0, 1)
    a = a * (1 - 0.7 * circle) + 128 * 0.7 * circle
    # 左上角全透明块
    a[:60, :80] = 0
    img = np.stack([r, g, b, a], axis=-1).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGBA")


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def qwen_patchify(rgb_f32, gh, gw):
    """归一化后的 RGB [H, W, 3] -> [gh*gw, 1536]，官方 pixel_values 段序
    [C=3][T=2][16][16]（temporal 两帧=同帧复制），token 序
    (gh/2, gw/2, merge_h, merge_w)。与 qwen_imgproc.c 的
    qw_imgproc_patchify（idx=(c*2+t)*256+ty*16+tx）逐位对齐。"""
    h, w = gh * 16, gw * 16
    norm = rgb_f32.reshape(h, w, 3).astype(np.float64)
    norm = (norm / 255.0 - 0.5) / 0.5
    seq = gh * gw
    out = np.zeros((seq, 3 * 2 * 16 * 16), dtype=np.float64)
    t = 0
    for br in range(gh // 2):
        for bc in range(gw // 2):
            for ir in range(2):
                for ic in range(2):
                    idx = 0
                    for c in range(3):
                        for tt in range(2):
                            for ty in range(16):
                                for tx in range(16):
                                    out[t, idx] = norm[
                                        (br * 2 + ir) * 16 + ty,
                                        (bc * 2 + ic) * 16 + tx, c]
                                    idx += 1
                    t += 1
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    img = make_test_image()
    w, h = img.size
    out_w_up, out_h_up = calculate_dimensions(OUTPUT_RES ** 2, w / h)
    out_w_dn, out_h_dn = calculate_dimensions(256 ** 2, w / h)
    print(f"input {w}x{h} -> up {out_w_up}x{out_h_up}, dn {out_w_dn}x{out_h_dn}")

    def save_f32(path, arr):
        np.asarray(arr, dtype=np.float32).tofile(path)

    save_f32(f"{OUT}/imgproc_input_rgba.f32",
             np.asarray(img, dtype=np.float32).reshape(-1))

    # 白底合成（PIL paste 语义：dst = dst*(1-a) + src*a）
    white = Image.new("RGB", img.size, (255, 255, 255))
    white.paste(img, mask=img.getchannel("A"))
    save_f32(f"{OUT}/imgproc_white_rgb.f32",
             np.asarray(white, dtype=np.float32).reshape(-1))

    # VaeImageProcessor.resize（default 模式，resample=config.resample=lanczos）
    up = white.resize((out_w_up, out_h_up), Image.Resampling.LANCZOS)
    dn = white.resize((out_w_dn, out_h_dn), Image.Resampling.LANCZOS)
    save_f32(f"{OUT}/imgproc_vis_rgb_up.f32",
             np.asarray(up, dtype=np.float32).reshape(-1))
    save_f32(f"{OUT}/imgproc_vis_rgb_dn.f32",
             np.asarray(dn, dtype=np.float32).reshape(-1))
    # VAE 路：RGBA resize（do_convert_rgb 默认 False，保 4 通道）
    vae_up = img.resize((out_w_up, out_h_up), Image.Resampling.LANCZOS)
    save_f32(f"{OUT}/imgproc_vae_rgba_up.f32",
             np.asarray(vae_up, dtype=np.float32).reshape(-1))

    # 视觉塔最终输入（up 主链）
    gh, gw = out_h_up // 16, out_w_up // 16
    pv = qwen_patchify(np.asarray(up, dtype=np.float32).reshape(-1), gh, gw)
    save_f32(f"{OUT}/imgproc_pixel_values.f32", pv)
    np.asarray([1, gh, gw], dtype=np.int32).tofile(f"{OUT}/imgproc_grid.i32")

    size_meta = np.asarray([h, w, out_h_up, out_w_up, out_h_dn, out_w_dn],
                           dtype=np.int32)
    size_meta.tofile(f"{OUT}/imgproc_input_size.i32")
    manifest = {"in": [h, w], "up": [out_h_up, out_w_up],
                "dn": [out_h_dn, out_w_dn], "grid": [1, gh, gw],
                "seq": gh * gw}
    with open(f"{OUT}/imgproc_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print("imgproc fixtures written, seq =", gh * gw)


if __name__ == "__main__":
    main()
