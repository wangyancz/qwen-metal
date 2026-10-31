"""Generate the VAE-decoder parity fixture from the released checkpoint.

Run with the diffusers venv python:
  /Volumes/Work/diffusers/.venv/bin/python tests/gen_vae_fixture.py [outdir]

Mirrors the production decode exactly (qwen-web worker):
  - VAE weights and compute in fp32 (pipe.vae.to(torch.float32))
  - decode(z) with the streaming feature cache active and first_chunk=True
    for the single latent frame (AutoencoderKLQwenImage21._decode)
  - output clamped to [-1, 1]; frame 0 is the RGBA image

Dumps per module (fp32 raw .bin + shapes.json manifest) so test_vae.c can
compare stage by stage, plus the final frame-0 RGBA image.
"""

import json
import os
import sys

import numpy as np
import torch

MODEL_DIR = "/Volumes/Work/models/Qwen-Image-2.1.models/vae"


def save_f32(name, tensor):
    """Dump an fp32 tensor as raw little-endian .bin."""
    array = np.ascontiguousarray(tensor.float().numpy(), dtype="<f4")
    os.makedirs(os.path.dirname(name), exist_ok=True)
    array.tofile(name)


def build_manifest(module):
    """Name every leaf module uniquely in execution order."""
    names = {}
    counter = [0]
    seen = set()
    for name, mod in module.named_modules():
        if name == "" or name in seen:
            continue
        seen.add(name)
        if len(list(mod.children())) == 0:
            key = f"m{counter[0]:03d}_{name.replace('.', '_')}"
            names[name] = key
            counter[0] += 1
    return names


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "tests/vae_fixture"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(MODEL_DIR)
    vae = vae.to(dtype=torch.float32, device=device)
    vae.eval()

    manifest = build_manifest(vae.decoder)
    manifest["post_quant_conv"] = "m900_post_quant_conv"

    captures = {}

    def mk(key):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captures[key] = t.detach()
        return hook

    for name, key in manifest.items():
        if name == "post_quant_conv":
            vae.post_quant_conv.register_forward_hook(mk(key))
        else:
            dict(vae.decoder.named_modules())[name].register_forward_hook(
                mk(key))

    shapes = {}

    def record(key, tensor):
        shapes[key] = list(tensor.shape)
        save_f32(f"{outdir}/{key}.bin", tensor.detach().to("cpu"))

    cases = {
        "a": (16, 16),   # 256 px image
        "b": (32, 32),   # 512 px image
    }
    for case, (lh, lw) in cases.items():
        torch.manual_seed(42 + hash(case) % 1000)
        z = torch.randn(1, 64, 1, lh, lw, dtype=torch.float32, device=device)

        # Production denorm before decode.
        mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
        z_denorm = (z * std.to(device) + mean.to(device)).to(
            device=device, dtype=torch.float32)

        captures.clear()
        with torch.no_grad():
            out = vae.decode(z_denorm, return_dict=False)[0]
        frame0 = out[:, :, 0]

        record(f"z_{case}", z_denorm)
        for key, tensor in captures.items():
            record(f"{case}_{key}", tensor)
        record(f"out_{case}", out)
        record(f"frame0_{case}", frame0)

    print(f"wrote fixture to {outdir}")
    with open(f"{outdir}/shapes.json", "w") as f:
        json.dump(shapes, f, indent=1, sort_keys=True)

    # ---- encoder case (ti2i): RGBA input in [-1, 1]. ----
    torch.manual_seed(4242)
    x = torch.randn(1, 4, 1, 128, 128, dtype=torch.float32, device=device)
    x = x.clamp(-1, 1)
    captures.clear()
    for name, mod in vae.encoder.named_modules():
        if name in ("encoder", "") or len(list(mod.children())) != 0:
            continue
        dict(vae.encoder.named_modules())[name].register_forward_hook(
            mk(f"enc_{name.replace('.', '_')}"))
    with torch.no_grad():
        dist = vae.encode(x, return_dict=False)[0]
    mu = dist.parameters[:, :64]
    for key, tensor in captures.items():
        record(f"{key}", tensor)
    record("enc_input", x)
    record("enc_mu", mu)
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1).to(device)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1).to(device)
    record("enc_latents", (mu - mean) / std)

    # Shape trace for the first case: resolves the temporal semantics.
    trace = {k: v for k, v in shapes.items() if k.startswith("a_")}
    for key in sorted(trace):
        print(key, trace[key])


if __name__ == "__main__":
    main()
