#!/usr/bin/env python3
"""Generate the DiT parity fixture from the released checkpoint.

Run with the diffusers venv python:
  /Volumes/Work/diffusers/.venv/bin/python tests/gen_dit_fixture.py [outdir]

Builds the real QwenImage21 submodules (txt_in, img_in, timestep embedding,
shared modulation, one transformer block, norm_out, proj_out, Rope), loads
their weights from the released shards, and runs a single-layer t2i-shaped
forward. Every intermediate is dumped as raw binary for the C Metal replay
in tests/test_dit_block.c. The stage values come from the real module code,
so rounding boundaries match the reference exactly.

Case A: one pure image block (every token is target; plain full attention).
Case B: 8 text tokens + a 3x4 latent grid; text rows modulate from the t=0
        row and attend causally (block-causal prefill decomposition).
"""
import json
import os
import sys

import torch
import torch.nn as nn

from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AdaLayerNormContinuous,
    QwenImage21Rope,
    QwenImage21TextProjection,
    QwenImage21TimestepProjEmbeddings,
    QwenImage21TransformerBlock,
    _qwenimage21_prefix_segments,
    QwenImage21Transformer2DModel,
)

MODEL = "/Volumes/Work/models/Qwen-Image-2.1.models"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dit_fixture")

HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
SEQ_A = 16
TEXT_B = 8
GRID_B = (3, 4)
SIGMA = 0.37


def save(name, tensor):
    path = os.path.join(OUT, name)
    if tensor.dtype == torch.bfloat16:
        data = tensor.contiguous().view(torch.int16).cpu().numpy().astype("<i2").tobytes()
    elif tensor.dtype == torch.float32:
        data = tensor.contiguous().cpu().numpy().astype("<f4").tobytes()
    else:
        raise ValueError(f"unsupported dtype for {name}: {tensor.dtype}")
    with open(path, "wb") as handle:
        handle.write(data)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cpu")

    from safetensors import safe_open
    index = json.load(open(os.path.join(
        MODEL, "transformer", "diffusion_pytorch_model.safetensors.index.json")))
    weight_map = index["weight_map"]
    handles = {}

    def weight(name):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(
                os.path.join(MODEL, "transformer", shard),
                framework="pt", device="cpu")
        return handles[shard].get_tensor(name).to(torch.bfloat16)

    # ---- real reference modules ----
    block = QwenImage21TransformerBlock(
        dim=HIDDEN, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
        mlp_ratio=3, eps=1e-6).to(torch.bfloat16)
    txt_in = QwenImage21TextProjection(4096, HIDDEN, eps=1e-6).to(torch.bfloat16)
    time_embed = QwenImage21TimestepProjEmbeddings(HIDDEN).to(torch.bfloat16)
    modulation_proj = nn.Sequential(
        nn.SiLU(), nn.Linear(HIDDEN, 4 * HIDDEN, bias=False)).to(torch.bfloat16)
    norm_out = QwenImage21AdaLayerNormContinuous(
        HIDDEN, HIDDEN, eps=1e-6).to(torch.bfloat16)
    img_in = nn.Linear(64, HIDDEN, bias=False).to(torch.bfloat16)
    proj_out = nn.Linear(HIDDEN, 64, bias=False).to(torch.bfloat16)

    block.load_state_dict({
        "attn.to_q.weight": weight("transformer_blocks.0.attn.to_q.weight"),
        "attn.to_k.weight": weight("transformer_blocks.0.attn.to_k.weight"),
        "attn.to_v.weight": weight("transformer_blocks.0.attn.to_v.weight"),
        "attn.to_out.0.weight": weight("transformer_blocks.0.attn.to_out.0.weight"),
        "attn.norm_q.weight": weight("transformer_blocks.0.attn.norm_q.weight"),
        "attn.norm_k.weight": weight("transformer_blocks.0.attn.norm_k.weight"),
        "img_mlp.gate_layer.weight": weight("transformer_blocks.0.img_mlp.gate_layer.weight"),
        "img_mlp.proj.weight": weight("transformer_blocks.0.img_mlp.proj.weight"),
        "img_mlp.out.weight": weight("transformer_blocks.0.img_mlp.out.weight"),
    }, strict=True)
    txt_in.load_state_dict({
        "text_norm.weight": weight("txt_in.text_norm.weight"),
        "in_layer.weight": weight("txt_in.in_layer.weight"),
        "out_layer.weight": weight("txt_in.out_layer.weight"),
    }, strict=True)
    time_embed.load_state_dict({
        "timestep_embedder.linear_1.weight":
            weight("time_text_embed.timestep_embedder.linear_1.weight"),
        "timestep_embedder.linear_2.weight":
            weight("time_text_embed.timestep_embedder.linear_2.weight"),
    }, strict=True)
    modulation_proj.load_state_dict(
        {"1.weight": weight("modulation.1.weight")}, strict=True)
    norm_out.load_state_dict({"linear.weight": weight("norm_out.linear.weight")},
                             strict=True)
    img_in.load_state_dict({"weight": weight("img_in.weight")}, strict=True)
    proj_out.load_state_dict({"weight": weight("proj_out.weight")}, strict=True)

    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])

    def block_layer(index):
        module = QwenImage21TransformerBlock(
            dim=HIDDEN, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
            mlp_ratio=3, eps=1e-6).to(torch.bfloat16)
        prefix = f"transformer_blocks.{index}."
        module.load_state_dict({
            "attn.to_q.weight": weight(prefix + "attn.to_q.weight"),
            "attn.to_k.weight": weight(prefix + "attn.to_k.weight"),
            "attn.to_v.weight": weight(prefix + "attn.to_v.weight"),
            "attn.to_out.0.weight": weight(prefix + "attn.to_out.0.weight"),
            "attn.norm_q.weight": weight(prefix + "attn.norm_q.weight"),
            "attn.norm_k.weight": weight(prefix + "attn.norm_k.weight"),
            "img_mlp.gate_layer.weight": weight(prefix + "img_mlp.gate_layer.weight"),
            "img_mlp.proj.weight": weight(prefix + "img_mlp.proj.weight"),
            "img_mlp.out.weight": weight(prefix + "img_mlp.out.weight"),
        }, strict=True)
        return module

    def run_full_case(case, spec, suffix=""):
        """All 32 layers through the real block modules; only the final
        output is dumped (single-block stages are already parity-tested).
        suffix="" runs the released bf16 path; suffix="_f32" runs the same
        wiring fully in fp32 as an intrinsic-sensitivity yardstick."""
        compute_dtype = torch.float32 if suffix else torch.bfloat16
        text_rows = spec["text"]
        gh, gw = spec["grid"]
        image_rows = gh * gw
        seq = text_rows + image_rows
        gen = torch.Generator().manual_seed(23 if case == "a" else 31)
        text_in_values = (torch.randn(text_rows, HIDDEN, generator=gen)
                          .to(torch.bfloat16) if text_rows
                          else torch.zeros(0, HIDDEN).to(torch.bfloat16))
        latents = (torch.randn(image_rows, 64, generator=gen)
                   .to(torch.bfloat16))
        image_pad_mask = torch.cat([
            torch.zeros(text_rows, dtype=torch.bool),
            torch.ones(image_rows, dtype=torch.bool)])
        target = image_pad_mask.clone()
        freqs_cis = rope([(1, gh, gw)], image_pad_mask, torch.device("cpu"))
        # The released model always feeds complex64 rope tables regardless
        # of compute dtype; never round them to bf16.
        if suffix:
            freqs_cis = freqs_cis.to(torch.float32)

        cast = (lambda module: module.to(torch.float32)) if suffix \
            else (lambda module: module)
        text_embeds = cast(txt_in)(text_in_values.to(compute_dtype))
        joint = torch.cat([text_embeds,
                           cast(img_in)(latents.to(compute_dtype))], 0)
        timestep = torch.tensor([SIGMA, 0.0])
        temb = cast(time_embed)(timestep, joint)
        modulation_model = cast(modulation_proj)(temb)
        segments = [(0, text_rows, True)] if text_rows else []

        hidden = joint
        for index in range(32):
            hidden = cast(block_layer(index))(
                hidden_states=hidden,
                modulation=modulation_model,
                rotary_emb=freqs_cis,
                attention_mask=None,
                target_token_mask=target,
                segments=segments,
            )
        final = cast(norm_out)(hidden, temb, target)
        output = cast(proj_out)(final)
        save(f"{case}_full_output{suffix}.bin", output.to(torch.bfloat16))
        return output

    def run_ti2i_case(suffix=""):
        """ti2i joint sequence: text runs interleaved with a condition
        latent block (the VLM image slots, expanded four-fold) and the
        target block last. Exercises the prefix-visible attention table,
        multi-block rope and the t=0 modulation split."""
        compute_dtype = torch.float32 if suffix else torch.bfloat16
        t0, t1 = 20, 17
        cond_grid, target_grid = (6, 8), (8, 8)
        cond_tokens = cond_grid[0] * cond_grid[1]
        target_tokens = target_grid[0] * target_grid[1]
        cond_slots, target_slots = cond_tokens // 4, target_tokens // 4
        text_rows = t0 + t1
        latents_rows = cond_tokens + target_tokens
        seq = text_rows + latents_rows
        gen = torch.Generator().manual_seed(47)
        # VLM full sequence: text0 | image slots | text1 (the slot rows
        # are placeholders — the joint overwrite replaces them).
        vlm_rows = t0 + cond_slots + t1
        text_in_values = (torch.randn(vlm_rows, HIDDEN, generator=gen)
                          .to(torch.bfloat16))
        latents = (torch.randn(latents_rows, 64, generator=gen)
                   .to(torch.bfloat16))
        img_mask = torch.cat([
            torch.zeros(t0), torch.ones(cond_slots), torch.zeros(t1),
            torch.ones(target_slots)]).bool()
        repeats = torch.where(img_mask, 4, 1)
        image_pad_mask = torch.repeat_interleave(img_mask, repeats)
        img_shapes = [(1, *cond_grid), (1, *target_grid)]
        freqs_cis = rope(img_shapes, image_pad_mask, torch.device("cpu"))
        if suffix:
            freqs_cis = freqs_cis.to(torch.float32)

        cast = (lambda module: module.to(torch.float32)) if suffix \
            else (lambda module: module)
        modules = [txt_in, time_embed, modulation_proj, norm_out, img_in,
                   proj_out]
        for module in modules:
            module.to(compute_dtype)
        text_embeds = cast(txt_in)(text_in_values.to(compute_dtype))
        joint = torch.cat([text_embeds,
                           torch.zeros(target_slots, HIDDEN).to(
                               text_embeds.dtype)], 0)
        joint = torch.repeat_interleave(joint, repeats, 0)
        joint[image_pad_mask] = cast(img_in)(latents.to(compute_dtype))
        image_ids, target_token_mask = (
            QwenImage21Transformer2DModel.build_token_metadata(
                image_pad_mask, img_shapes))
        prefix_len = int((~target_token_mask).sum())
        segments = _qwenimage21_prefix_segments(image_ids, prefix_len)

        timestep = torch.tensor([SIGMA, 0.0])
        temb = cast(time_embed)(timestep, joint)
        modulation_model = cast(modulation_proj)(temb)

        hidden = joint
        for index in range(32):
            hidden = cast(block_layer(index))(
                hidden_states=hidden,
                modulation=modulation_model,
                rotary_emb=freqs_cis,
                attention_mask=None,
                target_token_mask=target_token_mask,
                segments=segments,
            )
        final = cast(norm_out)(hidden, temb, target_token_mask)
        output = cast(proj_out)(final)
        if suffix:
            for module in modules:
                module.to(torch.bfloat16)
        if not suffix:
            save("c_text_in.bin", text_in_values)
            save("c_latents.bin", latents)
            save("c_rope_cos.f32", freqs_cis.real.float())
            save("c_rope_sin.f32", freqs_cis.imag.float())
        save(f"c_full_output{suffix}.bin", output.to(torch.bfloat16))
        meta_cases["c"] = {
            "seq": seq,
            "text_rows": text_rows,
            "segments": [[0, t0, True], [t0, t0 + cond_tokens, False],
                         [t0 + cond_tokens, seq - target_tokens, True],
                         [seq - target_tokens, seq, False]],
        }

    cases = {
        "a": {"text": 0, "grid": (4, 4)},
        "b": {"text": TEXT_B, "grid": GRID_B},
    }
    meta_cases = {}
    for case, spec in cases.items():
        text_rows = spec["text"]
        gh, gw = spec["grid"]
        image_rows = gh * gw
        seq = text_rows + image_rows

        gen = torch.Generator().manual_seed(23 if case == "a" else 31)
        text_in_values = (torch.randn(text_rows, HIDDEN, generator=gen)
                          .to(torch.bfloat16) if text_rows
                          else torch.zeros(0, HIDDEN).to(torch.bfloat16))
        latents = (torch.randn(image_rows, 64, generator=gen)
                   .to(torch.bfloat16))

        image_pad_mask = torch.cat([
            torch.zeros(text_rows, dtype=torch.bool),
            torch.ones(image_rows, dtype=torch.bool)])
        target = image_pad_mask.clone()
        img_shapes = [(1, gh, gw)]
        save(f"{case}_text_in.bin", text_in_values)
        save(f"{case}_latents.bin", latents)

        freqs_cis = rope(img_shapes, image_pad_mask, device)
        save(f"{case}_rope_cos.f32", freqs_cis.real.float())
        save(f"{case}_rope_sin.f32", freqs_cis.imag.float())

        text_embeds = txt_in(text_in_values)
        image_embeds = img_in(latents)
        joint = torch.cat([text_embeds, image_embeds], 0)
        save(f"{case}_joint_in.bin", joint)

        # causal_condition: the real timestep row first, the t=0 row last.
        timestep = torch.tensor([SIGMA, 0.0])
        temb = time_embed(timestep, joint)
        save(f"{case}_temb.bin", temb)

        modulation_model = modulation_proj(temb)
        # Reorder into the C-side column layout: [mod1.scale, mod2.scale,
        # tanh(mod1.gate), tanh(mod2.gate)].
        mod1, mod2 = modulation_model.chunk(2, dim=-1)
        modulation_c = torch.cat([
            mod1[:, :HIDDEN], mod2[:, :HIDDEN],
            torch.tanh(mod1[:, HIDDEN:].float()).to(torch.bfloat16),
            torch.tanh(mod2[:, HIDDEN:].float()).to(torch.bfloat16),
        ], dim=-1)
        save(f"{case}_modulation.bin", modulation_c)

        stages = {}

        def pre_hook(name):
            def hook(module, args, kwargs=None):
                value = None
                if kwargs and "hidden_states" in kwargs:
                    value = kwargs["hidden_states"]
                elif isinstance(args, tuple) and len(args) > 0:
                    value = args[0]
                stages[name] = value.detach().clone()
            return hook

        def post_hook(name):
            def hook(module, args, output):
                stages[name] = output.detach().clone()
            return hook

        handle1 = block.attn.register_forward_pre_hook(
            pre_hook("attn_in"), with_kwargs=True)
        handle2 = block.attn.register_forward_hook(post_hook("attn_out"))
        handle3 = block.img_mlp.register_forward_pre_hook(pre_hook("mlp_in"))
        handle4 = block.img_mlp.register_forward_hook(post_hook("mlp_out"))
        handle5 = block.attn.to_out[0].register_forward_pre_hook(
            pre_hook("attn_merged"))
        try:
            block_out = block(
                hidden_states=joint,
                modulation=modulation_model,
                rotary_emb=freqs_cis,
                attention_mask=None,
                target_token_mask=target,
                # Real t2i pipeline: the causal text prefix is the only
                # conditioning segment; the target image rows attend every
                # key through the processor's final call.
                segments=[(0, text_rows, True)] if text_rows else [],
            )
        finally:
            handle1.remove()
            handle2.remove()
            handle3.remove()
            handle4.remove()
            handle5.remove()

        save(f"{case}_attn_in.bin", stages["attn_in"])
        save(f"{case}_attn_out.bin", stages["attn_out"])
        save(f"{case}_mlp_in.bin", stages["mlp_in"])
        save(f"{case}_mlp_out.bin", stages["mlp_out"])
        save(f"{case}_block_out.bin", block_out)
        save(f"{case}_attn_merged.bin", stages["attn_merged"])

        final = norm_out(block_out, temb, target)
        save(f"{case}_norm_out.bin", final)
        output = proj_out(final)
        save(f"{case}_output.bin", output)

        meta_cases[case] = {
            "seq": seq,
            "text_rows": text_rows,
            "segments": [[0, text_rows, True]] if text_rows else [],
        }

    meta = {
        "hidden": HIDDEN,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "eps": 1e-6,
        "sigma": SIGMA,
        "cases": meta_cases,
    }
    run_full_case("a", cases["a"])
    run_full_case("b", cases["b"])
    run_full_case("a", cases["a"], "_f32")
    run_full_case("b", cases["b"], "_f32")
    run_ti2i_case()
    run_ti2i_case("_f32")
    with open(os.path.join(OUT, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=1)
    print(f"wrote fixture to {OUT}")
    print(json.dumps(meta_cases))


if __name__ == "__main__":
    main()
