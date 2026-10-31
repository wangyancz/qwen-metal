#!/usr/bin/env python3
"""Generate the text-encoder parity fixture from the released checkpoint.

Run with the diffusers venv python:
  /Volumes/Work/diffusers/.venv/bin/python tests/gen_text_fixture.py [outdir]

Dumps (for one prompt):
  - the raw-template token ids from the checkpoint tokenizer
  - the system-prefix drop count (chat-template derived, as released)
  - the 36-layer causal encode, last-decoder-layer output BEFORE the final
    RMSNorm (final norm neutralized exactly like the released
    QwenImage21Pipeline), system prefix dropped
  - the same wiring in fp32 as an intrinsic-sensitivity yardstick
"""
import json
import os
import sys

import torch

MODEL = "/Volumes/Work/models/Qwen-Image-2.1.models"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "text_fixture")

PROMPT = "A red apple on a wooden table, morning light, 4k photo"
SYS_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE = (f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n")


def save(name, tensor):
    path = os.path.join(OUT, name)
    if tensor.dtype == torch.bfloat16:
        data = tensor.contiguous().view(torch.int16).cpu().numpy().astype("<i2").tobytes()
    else:
        data = tensor.contiguous().cpu().numpy().astype("<i4").tobytes()
    with open(path, "wb") as handle:
        handle.write(data)


def dump_layer_stages(model, input_ids, positions, layer_indices=(0, 6, 16)):
    """Hook-captured per-layer intermediates for the staged parity harness.

    Layer 0 proves the wiring; layer 16 is where the sink token's massive
    activations emerge. The C encoder dumps the same stages
    (QWEN_TEXT_DUMP=tests/text_fixture/cdump); test_text_encoder.c compares
    the pairs. Layouts match the C row-major [seq, heads, head_dim] convention.
    """
    captures = {}

    def mk(name):
        def hook(module, inp, out):
            captures[name] = out
        return hook

    for idx in layer_indices:
        layer = model.layers[idx]
        p = f"l{idx}_"
        layer.input_layernorm.register_forward_hook(mk(p + "input_ln"))
        layer.self_attn.q_proj.register_forward_hook(mk(p + "q_proj"))
        layer.self_attn.k_proj.register_forward_hook(mk(p + "k_proj"))
        layer.self_attn.v_proj.register_forward_hook(mk(p + "v_proj"))
        layer.self_attn.q_norm.register_forward_hook(mk(p + "q_norm"))
        layer.self_attn.k_norm.register_forward_hook(mk(p + "k_norm"))
        # o_proj input tuple = the pre-projection attention output.
        layer.self_attn.o_proj.register_forward_hook(
            lambda m, i, o, p=p: captures.__setitem__(p + "attn", i[0]))
        layer.self_attn.o_proj.register_forward_hook(mk(p + "o_proj"))
        layer.self_attn.register_forward_hook(mk(p + "self_attn"))
        layer.post_attention_layernorm.register_forward_hook(
            mk(p + "post_ln"))
        layer.mlp.gate_proj.register_forward_hook(mk(p + "gate"))
        layer.mlp.up_proj.register_forward_hook(mk(p + "up"))
        layer.mlp.down_proj.register_forward_hook(
            lambda m, i, o, p=p: captures.__setitem__(p + "act", i[0]))
        layer.mlp.register_forward_hook(mk(p + "mlp"))
        # Layer forward hook: input[0] is the residual, output the result.
        layer.register_forward_hook(
            lambda m, i, o, p=p: captures.__setitem__(p + "residual", i[0]))
        layer.register_forward_hook(mk(p + "out"))
    model.embed_tokens.register_forward_hook(mk("l0_embed"))
    model.rotary_emb.register_forward_hook(mk("rotary"))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=None,
                        position_ids=positions, output_hidden_states=True)

    # Per-layer outputs (pre-final-norm) for drift localization.
    for idx in range(1, len(outputs.hidden_states)):
        save(f"layer_out_{idx - 1}.bin", outputs.hidden_states[idx][0])

    def flat(name):
        t = captures[name]
        if isinstance(t, tuple):
            t = t[0]
        return t.detach()[0].contiguous()

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    cos, sin = captures["rotary"]  # [1, seq, head_dim] each
    save("l0_rope_cos.bin", cos[0])
    save("l0_rope_sin.bin", sin[0])

    for idx in layer_indices:
        p = f"l{idx}_"
        save(p + "input_ln.bin", flat(p + "input_ln"))
        save(p + "q_proj.bin", flat(p + "q_proj"))
        save(p + "k_proj.bin", flat(p + "k_proj"))
        save(p + "v_proj.bin", flat(p + "v_proj"))
        save(p + "q_norm.bin", flat(p + "q_norm"))
        save(p + "k_norm.bin", flat(p + "k_norm"))

        # Replicate the module's rope to dump post-rope Q/K (half-split
        # pairing on the head dim, seq-major view saved).
        q4 = flat(p + "q_norm").transpose(0, 1).unsqueeze(0)  # [1,32,seq,dim]
        k4 = flat(p + "k_norm").transpose(0, 1).unsqueeze(0)  # [1,8,seq,dim]
        cos4 = cos[0].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin4 = sin[0].unsqueeze(0).unsqueeze(0)
        q_embed = q4 * cos4 + rotate_half(q4) * sin4
        k_embed = k4 * cos4 + rotate_half(k4) * sin4
        save(p + "q_embed.bin", q_embed[0].transpose(0, 1))
        save(p + "k_embed.bin", k_embed[0].transpose(0, 1))

        save(p + "attn.bin", flat(p + "attn"))
        save(p + "o_proj.bin", flat(p + "o_proj"))
        save(p + "res1.bin", flat(p + "residual") + flat(p + "self_attn"))
        save(p + "post_ln.bin", flat(p + "post_ln"))
        save(p + "gate.bin", flat(p + "gate"))
        save(p + "up.bin", flat(p + "up"))
        save(p + "act.bin", flat(p + "act"))
        save(p + "mlp.bin", flat(p + "mlp"))
        save(p + "out.bin", flat(p + "out"))
    print(f"stage tensors dumped for layers {list(layer_indices)}")


def main():
    os.makedirs(OUT, exist_ok=True)

    from transformers import AutoTokenizer, Qwen3VLTextConfig, Qwen3VLTextModel
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL, "processor"))
    payload = TEMPLATE.format(PROMPT)
    encoded = tokenizer(payload, padding=True, padding_side="left",
                        return_tensors="pt")
    input_ids = encoded.input_ids  # [1, seq]
    chat_message = [{"role": "system",
                     "content": [{"type": "text", "text": SYS_PROMPT}]}]
    sys_tokens = tokenizer.apply_chat_template(chat_message, tokenize=True,
                                               return_dict=False)
    if isinstance(sys_tokens, int):
        sys_tokens = [sys_tokens]
    if sys_tokens and isinstance(sys_tokens[0], list):
        drop = len(sys_tokens[0])
    else:
        drop = len(sys_tokens)
    print(f"template tokens: {int(input_ids.shape[-1])}, drop: {drop}")

    config = Qwen3VLTextConfig(
        vocab_size=151936, num_hidden_layers=36, hidden_size=4096,
        intermediate_size=12288, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
        rope_theta=5000000.0, max_position_embeddings=262144,
        rope_scaling={"mrope_interleaved": True,
                      "mrope_section": [24, 20, 20],
                      "rope_type": "default"})
    model = Qwen3VLTextModel(config)

    index = json.load(open(os.path.join(
        MODEL, "text_encoder", "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    handles = {}

    def weight(name):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(
                os.path.join(MODEL, "text_encoder", shard),
                framework="pt", device="cpu")
        return handles[shard].get_tensor(name)

    state = {}
    for name in weight_map:
        short = name
        if short.startswith("model.language_model."):
            short = short[len("model.language_model."):]
        if (short.startswith("lm_head") or short.startswith("visual.") or
                short.startswith("model.visual.")):
            continue
        state[short] = weight(name).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not [m for m in missing], f"missing: {missing[:4]}"
    assert not unexpected, f"unexpected: {unexpected[:4]}"

    # Pure-text mrope: all three axes carry the same 1-D positions.
    seq_len = int(input_ids.shape[-1])
    positions = torch.arange(seq_len).expand(3, 1, -1)
    model = model.to(torch.bfloat16)
    # Match the released QwenImage21Pipeline: a forward hook returning the
    # norm's INPUT neutralizes the final RMSNorm, so hidden_states[-1] is
    # the pre-norm last decoder layer output (transformers 5.x otherwise
    # ties that entry to the normalized last_hidden_state).
    model.norm.register_forward_hook(lambda module, args, output: args[0])
    dump_layer_stages(model, input_ids, positions)

    with torch.no_grad():
        for dtype, suffix in ((torch.bfloat16, ""), (torch.float32, "_f32")):
            run_model = model.to(dtype)
            ids = input_ids if dtype == torch.bfloat16 else input_ids
            outputs = run_model(input_ids=ids, attention_mask=None,
                                position_ids=positions,
                                output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0]
            # Left padding is a no-op for a single prompt; drop the system
            # prefix exactly like the released pipeline.
            hidden = hidden[drop:]
            if suffix:
                save(f"hidden{suffix}.bin", hidden.to(torch.bfloat16))
            else:
                save("hidden.bin", hidden)

    save("input_ids.bin", input_ids.to(torch.int32))
    meta = {"prompt": PROMPT, "sys_prompt": SYS_PROMPT, "drop": drop,
            "seq": int(input_ids.shape[-1]), "hidden": 4096}
    with open(os.path.join(OUT, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=1)
    print(f"wrote fixture to {OUT}: seq={meta['seq']} drop={drop}")


if __name__ == "__main__":
    main()
