"""Reference values for the pipeline component tests: RoPE tables from
the released QwenImage21Rope and the FlowMatchEuler sigma schedule.

  /Volumes/Work/diffusers/.venv/bin/python tests/gen_pipeline_fixture.py
"""

import json
import os

import numpy as np
import torch

from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope

OUT = os.path.join(os.path.dirname(__file__), "pipeline_fixture")
os.makedirs(OUT, exist_ok=True)


def save_f32(name, values):
    np.asarray(values, dtype=np.float32).tofile(os.path.join(OUT, name))


# --- RoPE tables for (text_rows, gh, gw) grids ---------------------------
def reference_rope(text_rows, gh, gw):
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    image_pad_mask = torch.cat(
        [torch.zeros(text_rows, dtype=torch.bool),
         torch.ones(gh * gw, dtype=torch.bool)])
    freqs = rope([(1, gh, gw)], image_pad_mask, torch.device("cpu"))
    return freqs.real.numpy(), freqs.imag.numpy()


for text_rows, gh, gw in [(23, 4, 4), (23, 8, 8)]:
    real, imag = reference_rope(text_rows, gh, gw)
    save_f32(f"rope_cos_{gh}.f32", real.reshape(-1))
    save_f32(f"rope_sin_{gh}.f32", imag.reshape(-1))
    print(f"rope {gh}x{gw}: {real.shape}")

# --- Sigma schedules (mirror FlowMatchEulerDiscreteScheduler) ------------
# Each checkpoint is mirrored against its OWN scheduler config: the base
# model ships use_dynamic_shifting=true + shift_terminal 0.02, the turbo
# distillation ships false/null (sample_sigmas consumed as-is).
BASE_SCHED = ("/Volumes/Work/models/Qwen-Image-2.1.models"
              "/scheduler/scheduler_config.json")
TURBO_SCHED = ("/Volumes/Work/models/Qwen-Image-2.1-turbo.models"
               "/scheduler/scheduler_config.json")


def reference_schedule(steps, seq_len, config_path, turbo=None):
    config = json.load(open(config_path))
    base_seq = config["base_image_seq_len"]
    max_seq = config["max_image_seq_len"]
    base_shift = config["base_shift"]
    max_shift = config["max_shift"]
    dynamic = config.get("use_dynamic_shifting", False)
    static_shift = np.float32(config.get("shift", 1.0))
    terminal = config.get("shift_terminal")  # None → no stretch

    raw = (np.asarray(turbo, dtype=np.float32) if turbo is not None
           else np.linspace(1.0, 1.0 / steps, steps).astype(np.float32))
    if dynamic:
        m = (max_shift - base_shift) / (max_seq - base_seq)
        b = base_shift - m * base_seq
        mu = seq_len * m + b
        exp_mu = np.float32(np.exp(mu))
        # _time_shift_exponential(mu, 1.0, t) in fp32 tensor arithmetic
        inverse = (np.float32(1.0) / raw).astype(np.float32)
        base = (inverse - np.float32(1.0)).astype(np.float32)
        sigmas = (exp_mu / (exp_mu + base)).astype(np.float32)
    else:
        sigmas = (static_shift * raw /
                  (np.float32(1.0) +
                   (static_shift - np.float32(1.0)) * raw)).astype(np.float32)

    if terminal is not None:
        one_minus_z = (np.float32(1.0) - sigmas).astype(np.float32)
        scale = np.float32(one_minus_z[-1] / (1.0 - terminal))
        sigmas = (np.float32(1.0) - one_minus_z / scale).astype(np.float32)
    return np.concatenate([sigmas, [0.0]]).astype(np.float64)


TURBO_SIGMAS = json.load(open(
    "/Volumes/Work/models/Qwen-Image-2.1-turbo.models/model_index.json")
)["sample_sigmas"]

for steps, seq_len, name, config_path, turbo in [
        (40, 64, "s40", BASE_SCHED, None),
        (40, 4096, "s40_1024", BASE_SCHED, None),
        (8, 64, "t8", TURBO_SCHED, TURBO_SIGMAS)]:
    schedule = reference_schedule(steps, seq_len, config_path, turbo)
    save_f32(f"schedule_{name}.f32", schedule)
    print(f"schedule {name}: {schedule[:3]} ... {schedule[-2:]}")
