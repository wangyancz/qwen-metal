# qwen-metal

A native **Qwen-Image-2.1** text-to-image inference engine for Apple Silicon (Metal).

Pure C11 + Objective-C with zero Python runtime dependency: the text encoder (Qwen3-VL text tower), the 7B DiT, the VAE decoder, and the FlowMatch scheduler all run on hand-written Metal kernels and MPSGraph — one command takes you from prompt to PNG.

```
qwen generate -d /path/to/Qwen-Image-2.1.models \
    -p "a photorealistic red apple on a rustic wooden table, soft morning light" \
    --width 512 --height 512 --steps 40 --seed 7 --out apple.png
```

## Features

- **All-Metal inference**: DiT and text encoder in bf16 (fp32 accumulation), VAE in fp32; M5 TensorOps linear-layer acceleration with automatic fallback on older chips
- **Complete pipeline**: template tokenization → text encoding → dynamic-shifting scheduler → 32-layer single-stream DiT (block-causal attention + causal_condition) → VAE decode → PNG
- **Both checkpoints**: base (40 steps) and turbo (official 8-step distilled `sample_sigmas`) auto-detected
- **Classifier-free guidance (CFG)**: `--cfg` + `--negative`
- **LoRA adapters**: single-file safetensors (PEFT naming); on load, `scale · B@A` is fused into a copy of the weights; attach, re-scale, and detach at any time from the CLI/REPL
- **Equivalence-driven development**: every component (tokenization, text encoding, DiT segment by segment, VAE segment by segment, scheduler, RoPE) has bit-exact / noise-band alignment tests against the diffusers/transformers reference implementations

## Requirements

| Item | Requirement |
|---|---|
| Chip | Apple Silicon (validated on M5 Max; the TensorOps path needs M5, other chips fall back to plain kernels automatically) |
| OS | macOS (Metal / MPSGraph runtime required) |
| Toolchain | Xcode command line tools (clang + ObjC + Metal frameworks); zlib ships with the OS |
| Unified memory | **16 GB minimum** (~17 GB peak at 1024²); the largest official size tops out at ≤ 47 GB, comfortable on 64 GB |
| Disk | Model weights 31 GB (base) / 30 GB (turbo) |

## Getting the model

Download a diffusers-layout weights directory from the official Qwen-Image-2.1 release channels
(`model_index.json` + `transformer/` + `text_encoder/` + `vae/` + `scheduler/` + `processor/`), e.g.:

```
/path/to/Qwen-Image-2.1.models          # base
/path/to/Qwen-Image-2.1-turbo.models    # turbo (optional)
```

Note: use of the model weights is subject to their official license.

## Building

Prerequisites: Xcode command line tools (`xcode-select --install`, providing clang,
Objective-C, and the Metal frameworks); zlib and Accelerate ship with macOS.

```sh
cd qwen-metal
make clean && make     # full build, ~2 seconds
```

Artifacts:

- `./qwen` — command-line inference program (`--info` / `generate`)
- `libqwen.a` — static library (full engine + Metal runtime, embeddable in other programs)

All make targets:

| Target | Description | Needs model weights |
|---|---|---|
| `make` / `make all` | Build the CLI and static library | No |
| `make host-tests` | Host-side math / JSON / weight-tool tests | No |
| `make pipeline-tests` | Scheduler / RoPE / RNG component tests | No (fixtures ship with the repo) |
| `make dit-tests` | DiT per-segment blocks + full 32-layer forward | Yes |
| `make text-tests` | Text encoder tokenization + 36-layer hidden states | Yes |
| `make vae-tests` | VAE ~50-segment comparison | Yes |
| `make lora-tests` | LoRA fusion numeric parity + attach/detach lifecycle | Yes + LoRA files |
| `make dit-fixture` etc. | Regenerate equivalence fixtures | Yes + diffusers environment |
| `make clean` | Remove build artifacts | — |

Quick verification that does not need model weights:

```sh
make host-tests pipeline-tests
```

## Usage

### Inspect checkpoint info (without loading weights)

```sh
qwen --info -d /path/to/Qwen-Image-2.1.models
```

### Text-to-image

```sh
qwen generate -d MODEL_DIR -p "PROMPT" [options]
```

| Option | Description | Default |
|---|---|---|
| `-p TEXT` | Prompt (required) | — |
| `--negative TEXT` | Negative prompt (used with `--cfg > 1`) | — |
| `--width N` / `--height N` | Output size (multiple of 32, [32, 8192]) | 1024 × 1024 |
| `--steps N` | Denoising steps ([1, 1000]; turbo always takes 8) | base: 40 |
| `--seed N` | Random seed (same seed reproduces the same image) | 0 |
| `--cfg F` | Guidance strength (>1 enables CFG) | 1.0 |
| `--lora PATH` | LoRA adapter (safetensors; repeatable, up to 8 stacked — see section below) | — |
| `--lora-scale F` | Strength for the i-th `--lora` (repeat to pair in order; default 1.0, not clamped) | 1.0 |
| `--out PATH` | Output PNG path | out.png |

Examples:

```sh
# base, 1024x1024
qwen generate -d $MODELS/Qwen-Image-2.1.models \
    -p "a shiba inu wearing an astronaut helmet, pixar style" --width 1024 --height 1024 --seed 42

# turbo, 8-step quick generation (official fixed sigma trajectory; --steps is ignored on turbo)
qwen generate -d $MODELS/Qwen-Image-2.1-turbo.models \
    -p "cyberpunk skyline at dusk, neon reflections" --width 512 --height 512

# CFG guidance
qwen generate -d $MODELS/Qwen-Image-2.1.models \
    -p "a cozy log cabin in snowy forest, warm window light" \
    --negative "blurry, low quality, watermark" \
    --cfg 4 --width 768 --height 512 --seed 3

# Multiple reference images: use <imageN> in the prompt to address the N-th --image (N = order of appearance)
qwen generate -d $MODELS/Qwen-Image-2.1.models \
    -p "replace the apple in <image2> with a green pear, keep <image1> unchanged" \
    --image apple.png --image blue_apple.png --res 512 --steps 8 --seed 7
```

### Resolution and aspect ratios

Output side lengths must be a **multiple of 32** (pipeline `multiple_of = vae_scale_factor × 2`).
The official aspect-ratio bucket table is recommended: three fixed pixel sizes per ratio, with
cost determined by the token count (= pixels / 256) regardless of the width/height combination.
**The "max" tier is the largest resolution for each ratio** (≈ 7.0-7.1M pixels, 27.0-27.7k tokens):

| Ratio | Standard | HD | Max (largest) | Max pixels |
|---|---|---|---|---|
| 1:1 | 1024 × 1024 | 2048 × 2048 | **2656 × 2656** | 7.05M |
| 3:2 | 1248 × 832 | 2528 × 1696 | **3264 × 2176** | 7.10M |
| 2:3 | 832 × 1248 | 1696 × 2528 | **2176 × 3264** | 7.10M |
| 4:3 | 1248 × 928 | 2400 × 1792 | **3040 × 2272** | 6.91M |
| 3:4 | 928 × 1248 | 1792 × 2400 | **2272 × 3040** | 6.91M |
| 16:9 | 1376 × 768 | 2752 × 1536 | **3520 × 1984** | 6.98M |
| 9:16 | 768 × 1376 | 1536 × 2752 | **1984 × 3520** | 6.98M |

Notes:

- This engine has no hard 8M-pixel conv2d limit of the Python/MPS reference stack; every size in
  the table (including the 3:2 max at 3264 × 2176) has been tested and works. Larger side lengths
  are syntactically allowed but fall outside the training distribution — quality is not guaranteed.
- Peak memory (phys_footprint, with zero-copy weight mapping): ~17 GB at 1024²,
  ~10 GB at 512², ~34 GB at 1536 × 2752, ~47 GB at 3264 × 2176 —
  **the full official size range runs comfortably on a 64 GB machine**.
- Any multiple-of-32 combination outside these ratios also works, but prefer the buckets in the
  table — those are the ratios the model saw during training.

### Reference-image editing (ti2i)

```sh
qwen generate -d MODEL_DIR -p "change the apple's color to green" \
    --image /path/to/ref.png [options]
```

| Option | Description | Default |
|---|---|---|
| `--image PATH` | Reference PNG (RGBA; repeatable, ≤8; the decoder validates size/CRC) | — |
| `--res N` | Reference-encoding resolution area ([128, 4096], 32-aligned, ratio follows the reference; 0 = default) | 1024 |
| `--width N` / `--height N` | Output size (multiple of 32, [32, 8192]) | follows the reference encoding size |

Notes:

- The prompt describes "what to change into" (recolor / restyle / add-remove elements); the reference
  image provides the composition and content base.
- **`<imageN>` in the prompt addresses a specific reference image** (N = order of `--image`
  appearance, starting at 1). The token sequence is plain text (the tokenizer has no special token
  for it); the model associates it with the N-th image group preceding the prompt — same mechanism
  and same notation as the official stack.
  Example: `change the apple in <image2> to the color of <image1>`. Nonexistent indices are mostly
  ignored by the model (the CLI does not validate them — same pass-through as the official stack).
- `--negative` + `--cfg > 1` are supported (classifier-free guidance with the same semantics as the
  official QwenImage21Pipeline: the neg branch carries the same reference images, independent text
  encoding, and its own K-V cache; if cfg > 1 but no negative is provided, cfg falls back to 1 with a warning).
- When the output aspect differs from the reference encoding, the composition adapts automatically;
  keeping the two ratios close is recommended.
- Same-initial-noise cross-check against the official QwenImage21Pipeline: average pixel difference
  0.42/255 (1024²); with CFG enabled, 1.27/255 (512² — the guidance delta amplifies floating-point
  ordering noise).

### LoRA adapters

```sh
qwen generate -d MODEL_DIR -p "chibi a shiba inu" \
    --lora /path/to/lora.safetensors \
    --lora-scale 1.0 [options]

# Stacking multiple adapters (--lora is repeatable; --lora-scale pairs in order, default 1.0)
qwen generate -d MODEL_DIR -p "chibi 3D a shiba inu" \
    --lora qingban.safetensors --lora three_d.safetensors \
    --lora-scale 0.7 --lora-scale 0.5 [options]
```

- **Coverage**: single-file adapters with PEFT naming (`transformer_blocks.N.<module>
  .lora_A/lora_B.default.weight`); all 7 linear layers of each block
  (attn.to_q/k/v/out.0 + img_mlp.gate_layer/proj/out) are fused. Partial adapters with missing
  keys are allowed (skipped with a notice).
- **Stacking**: the adapter stack holds up to 8; `W' = bf16RNE(fp32(W) + Σ_a
  scale_a·B_a@A_a)` — any change (append / detach / re-scale) re-fuses all adapters from base in
  one pass, order-independent with no rounding accumulation; one rebuild takes ~3-5 s
  (independent of stack size); detaching everything restores the original weights (output is
  bit-identical to never having attached).
- **The implementation is weight-domain fusion** (fp32 accumulation + round-to-nearest-even,
  within ≤2 bf16 ulp of the independent hand-computed reference in `tests/test_lora`).
  The inference path is unchanged and the CFG branches share it naturally; applying the same
  adapter twice just updates its scale.
- **No `lora_alpha` semantics** (PEFT default scaling=1): `--lora-scale` is the entire strength.
  Training trigger words (e.g. "chibi") are ordinary tokens — just write them into the prompt.
- **Cost**: while an adapter is attached, the DiT weights leave the zero-copy mapping and become a
  private copy, adding +14 GB to the footprint (released on detach; the base weights remain in the
  page cache, so switching/detaching involves no weight I/O).
- **REPL**: `/lora PATH` appends to the stack, bare `/lora` detaches all,
  `/lora-scale F` changes the most recently attached adapter's strength, `/lora-drop PATH` removes
  one; `/show` lists the stack.

### Interactive session (REPL)

The model loads once; enter prompts line by line and keep generating:

```sh
qwen repl -d $MODELS/Qwen-Image-2.1-turbo.models
```

```
qwen 0.1.0 - interactive session (model resident)
  size 1024x1024  steps 8  seed 1793112200  cfg 1.00  neg (none)
qwen> a photorealistic red apple on a wooden table
wrote out-0.png (1024x1024, 8 steps, 21.4s)
qwen> /width 512
qwen> /cfg 4
qwen> /neg blurry, low quality
qwen> a cozy log cabin in a snowy forest
wrote out-1.png (512x512, 8 steps, 22.5s)
qwen> /quit
```

| Command | Description |
|---|---|
| `/width N`, `/height N` | Adjust output size |
| `/steps N` | Denoising steps |
| `/seed N` | Seed (auto-increments by 1 per image) |
| `/cfg F` | Guidance strength (>1 requires `/neg`) |
| `/neg TEXT` | Negative prompt (bare `/neg` clears it) |
| `/image PATH` | Add a reference image (repeatable, ≤8; bare `/image` clears; from then on prompts run the ti2i path and may use `<imageN>`) |
| `/res N` | ti2i output resolution tier (bare `/res` resets to 1024) |
| `/lora PATH` | Append a LoRA adapter to the stack (bare `/lora` detaches all; each change re-fuses in ~3-5 s) |
| `/lora-scale F` | Strength of the most recently attached adapter (re-fuses immediately) |
| `/lora-drop PATH` | Remove a single adapter from the stack |
| `/out PATH` | Output path template (`%llu` is the image index, default `out-%llu.png`) |
| `/show`, `/help`, `/quit` | Show state / help / quit |

Output size in ti2i sessions: by default derived from the first reference image according to the
`/res` tier; explicit `/width` / `/height` override it (must be multiples of 32). CFG
(`/neg` + `/cfg > 1`) also works on the ti2i path.

### JSONL service mode (serve)

```sh
qwen serve -d $MODELS/Qwen-Image-2.1.models
```

A resident JSONL worker: one request per line (JSON object); progress / done / error events go to
stdout, diagnostics to stderr. Any host program (Go / Python / Rust…) can spawn it as a subprocess
and drive the same protocol by reading and writing lines. Key points:

- Model resident (same as REPL), vision tower created lazily; exits on stdin EOF; cancel/idle unload =
  the host kills the process group
- Request fields: `id/prompt/negativePrompt/width/height/steps/cfgScale/seed/
  images[]/output/lora/loraScale` (≤8 reference images, given as local paths;
  steps ∈ [1, 1000], sizes multiple of 32 in [32, 8192])
- Progress: a 0..1 total mapping of `encode → sample (per-step callback) → decode → save`
  plus a 4 s heartbeat; LoRA sync (same path + same scale = no-op, otherwise detach/refuse/re-fuse)
- Malformed input / over-long lines (>64 KB) / out-of-range parameters return
  `{"type":"error","message":…}` events (global errors without an id can be ignored by the host as
  unknown ids); the stream stays line-aligned
- Must be run from the repository directory (`h3_shaders.metal` is loaded relative to cwd)

### Measured performance (M5 Max 128 GB)

| Configuration | Time per image (excluding model load) | Notes |
|---|---|---|
| base, 1024², 40 steps | **~68 s** (flash2 fallback 83.7 s, **1.22×**) | ~1.5 s per step |
| ti2i single image, 1024², 8 steps | **~23 s** (with cond KV cache; full recompute 41.8 s, **1.9×**) | 4 segments |
| ti2i 3 images, 1024², 8 steps | **~41 s** (with cond KV cache; full recompute 94.1 s, **2.3×**) | 8 segments |
| turbo, 512², 8 steps | **~3.1 s** | First run includes ~13 GB of cold weight reads (+5 s) |
| turbo, 256², 8 steps | **~1.4 s** | Peak memory ~33.5 GB |

Measured in the same thermal state (fans at full speed, model pages warm, 2026-10-30); run in pairs
against the `QWEN_FLASH_ATTN=1` fallback path with identical parameters and seed. The M5 Max
thermally throttles after sustained heavy load (t2i 8 steps creeping 13.7→21.9 s), so cross-round
timings must be compared at equal thermal state.

Compared with the official diffusers PyTorch MPS stack (bf16 pipeline + fp32 VAE + MPS F.pad fix +
turbo fixed schedule), same parameters and seed:

| Scenario | Official MPS | This engine | Ratio |
|---|---|---|---|
| t2i 1024² 40 steps | 77.4 s | **68.5 s** | 1.13× |
| t2i 1024² 8 steps | 16.9 s | **15.3 s** | 1.10× |
| ti2i single image 1024² 8 steps | 26.6 s | **22.6 s** | 1.18× |
| turbo 512² 8 steps | 3.6 s | **3.1 s** | 1.16× |

Ahead on all four. ti2i used to lag the official stack by 1.57× because the official edit pipeline
carries a KV cache (`use_kv_cache`) — the same mechanism is now implemented (see optimization log),
and this engine is 1.18× ahead.

Breakdown (1024² single step ~1.5 s): linear layers ~0.8 s (MPSGraph bf16, 51-65 TFLOPS,
fused qkv three-GEMM graph + MLP graph per image, MPSGraph encodes per layer 4→3), attention
~0.15 s (fused SDPA, 47-54 TFLOPS), element-wise / scheduling ~0.5 s; VAE decode of the whole
image is only ~2 s.

Peak memory (phys_footprint): ~17 GB at 1024², ~10 GB at 512²,
~34 GB at 1536 × 2752, ~47 GB at 3264 × 2176.

### Resident memory composition (serve / REPL modes)

`generate` is one-shot and exits when done; the resident composition and footprint behavior of
serve/REPL are as follows (with `QWEN_SERVE_MEM=1`, serve prints one live/footprint line per
request on stderr):

- **Zero-copy weight mapping does not count toward footprint** (see above); what stays resident is
  activations / workspaces / the two by-design structures below
- **ti2i cond KV cache**: per-layer K/V reused across sampling steps for the condition rows — about
  **12 GB** for 4-image 1024² output (36 layers × 2 × joint row count × 4096 × 2 B; each reference
  image is encoded at 1024² area, token count varies with aspect ratio), with a **single-request
  lifetime** (the cached rows encode this request's prompt/reference images/weights and are freed
  when the request ends — even the next request with the identical layout must not reuse them,
  otherwise it would silently inherit the previous request's conditioning)
- **LoRA fused copy**: while an adapter is active, layer weights are fused private copies,
  ~**14 GB** at full bf16 (the base originals are referenced through the mmap shadow, so rebuilds
  do no weight I/O); released on detach. Re-scaling / swapping adapters is an **in-place rebuild**
  (the fused tensors have fixed shapes, written back into existing buffers, zero new allocations) —
  implementing it as "allocate new, then free old" would ratchet the Metal pool up by a whole fused
  set on every change
- **MPSGraph shape caches are fully cleared at request boundaries**: the 6 h3 MPSGraph caches
  (sdpa/gqa/linear/mlp/qkv/conv) key on row count / sequence length and never evict entries —
  a long-lived session gains one entry per new prompt length or output size, measured at
  **~95 MB/entry** and monotonically growing (8 different-length prompts in a row: 16.88 → 17.64 GB).
  The graphs are only acceleration artifacts; dropping one just pays a one-time recompile
  (<0.2 s per request), so serve clears everything at the end of each request and REPL after each
  generation (`h3_gpu_clear_graph_cache`, no inflight commands at request boundaries): shape variety
  is bounded within a single request and footprint no longer accumulates across shapes (same
  sequence measured flat at 16.88 → 16.99 → 16.85 GB; mixed-load peak over 17 requests
  49.2 → 44.9 GB).
- **Footprint is the Metal pool's high-water mark**: freed MTLBuffer pages stay in the process-wide
  allocation pool and are never returned to the system (see below); once the pool has grown to the
  largest size class seen, it plateaus. Mixing requests of different sizes (varying reference-image
  count/ratio) pushes the plateau up; idle unload (host kills the process) or exit returns
  everything. The text tower / DiT / VAE text-row workspaces used to interleave "allocate new, then
  free" enqueues into a single command buffer, which made GPU execution of the same workload ~5×
  slower (MPS intermediate resources retained until commit) — fixed by committing the text tower in
  5 segments
- **Leak diagnosis**: `live_bytes` (h3 tensors currently alive) is the leak discriminator;
  `allocated_bytes` is **cumulative allocation** (monotonically increasing, ~10 GB of throughput per
  512² request — that is normal). On the malloc side, launching serve with `MallocStackLogging=1`
  and running `leaks <pid>` yields allocation stacks for leaked objects directly (this is how the
  ti2i vision main working buffer was located); currently after a 17-request mixed load `leaks`
  reports 0 leaks, fd count steady at 24, thread count steady.

Two key mechanisms:

- **Zero-copy weight mapping (on by default)**: weights are not copied into private buffers; instead
  safetensors files are mmapped and wrapped directly via `bytesNoCopy` — the file pages belong to
  the page cache, **do not count toward phys_footprint**, and can be evicted under memory pressure
  and re-read on demand (moving the 16 GB text tower + 13 GB DiT out of the memory budget).
  `H3_ZERO_COPY_WEIGHTS=0` disables it (falls back to private copies); `=transformer` maps only the DiT.
- **Metal does not return freed MTLBuffer pages to the system** (measured in-process: live_bytes
  drops while phys_footprint stays constant); pool reuse only covers new allocations that fit into
  free blocks. Accordingly: VAE convolutions commit command buffers per conv, the pad buffer is a
  single block reused across the whole chain (~4.9 GB at the largest size), resnet/attention/upsample
  stage boundaries commit immediately; `generate` mode releases the text tower after prompt encoding
  and the DiT after sampling — before these optimizations, peak at large resolutions reached 117 GB.

Optimization log:
- GEMM used to run on a 16×16 scalar tiled kernel (2.6 TFLOPS); restoring the MPSGraph bf16 fast path
  raised it to ~60 TFLOPS (**23×**, all three implementations bit-identical), single step 10.0 → 3.1 s.
- A hand-written scalar flash attention was 3.7× slower (the rect kernel was in fact K/V bandwidth
  saturated, 1.34 TB/s); switching to the MPP `matmul2d` TensorOps flash (64 queries × 64 keys tiles,
  cooperative fp32 accumulator with online softmax) brought the single step 3.1 → ~2.2 s. KVT=128
  exceeds the 32 KB threadgroup memory limit; 64 is the measured optimum.
- DiT image attention moved to the MPSGraph-native fused SDPA op
  (`scaledDotProductAttentionWithQueryTensor:...`, same as the vision tower):
  microbenchmark (`make bench-sdpa`, 1024² production shape, 32 heads × 128) measures
  47-54 TFLOPS vs the hand-written flash2 at 16 TFLOPS — **3.0-3.3× faster** (relL2
  0.0015); full 40-step image 100.3 → 77.8 s (**1.29×**). `QWEN_FLASH_ATTN=1`
  falls back to the flash2 kernel for A/B. **The SDPA graph is compiled at
  `MPSGraphOptimizationLevel0`** (`h3_gpu_sdpa_exec_descriptor`): the default Level1 runs the Neural
  Engine placement pass, which for this kind of single-op graph always fails ANE probing and prints
  "Failed to import MPS module" (ANECompiler `ParseMPSModule`) on stderr before falling back to the
  GPU — Level0 skips the pass, the message disappears, output is bit-identical to Level1, and the
  five-shape microbenchmark stays in band. The ti2i block-causal prefix kernel switched the same way:
  instead of a 539 MB attnMask matrix, it **splits per run** — the visible domain of an image run is
  exactly the rectangle [0, run_end), which goes straight to the fused rect path; the text run keeps
  the causal flash prefix kernel within the segment (~20 rows, negligible). Single image 8 steps
  58.9 → 40.1 s (**1.47×**); same-seed output differs from the prefix kernel by 0.20/255 pixels
  (noise band), and the ti2i fixture thresholds were calibrated to both implementations' intrinsic
  spread and relaxed to 0.06 (fused accumulation-order wobble ~0.001, same order as the engine's
  intrinsic nondeterminism).
- Post-attention re-verification (same-noise injection, step-by-step against the official pipeline):
  1024² single image step0 lat relL2 0.0034 (baseline 0.0034), final 0.0153 (baseline 0.0161),
  pixel difference 0.57/255 — aligned with the official full pipeline, no regression. 3-image
  1024² 8 steps at 95.5 s (flash2 fallback same parameters 160.4 s, **1.68×**).
- Element-wise fusion (2026-10-30): q_pad eliminated (the fused rect path consumes exact row counts;
  only the flash2 fallback keeps the 64-row tile pad), per-layer workspace reuse (six
  [rows, hidden] activation buffers allocated once per forward, reallocated when t2i/ti2i switch row
  counts), q/k/v pre-concatenation (`qw_gpu_qkv_bf16`: a single MPSGraph graph does the 3 matmuls
  sharing one input, weights stay in native [out, in] zero-copy layout, encodes per layer 3→1).
  `QWEN_BLOCK_PROFILE=1` (per-segment forced submit sync, for diagnostics) measured the input segment
  at 17.8→12.7 ms/layer (encode tax -0.16 s/step), full 40-step image 77.8→73.2 s (measured warm-GPU,
  so the gain is a lower bound); the fused output is **bit-identical** to the three-GEMM version.
  Measured gate+LN fusion is only ~0.2 ms/layer — the data does not support it, so it was not done;
  the remaining CPU encode tax (~5 MPSGraph encodes per layer × ~2-3 ms) partially overlaps GPU
  execution, and compressing it further requires whole-step GPU timeline capture — on hold.
- ti2i cond KV cache (2026-10-30): the cond/text rows' inputs are invariant across sampling steps
  (embed fixed, modulation pinned to the t=0 slot, rope fixed), so the first step runs in full and
  writes each layer's K/V into buffers that persist across steps (`dit->cache_k/v`, +4.3 GB single
  image, +8.8 GB for 3 images); subsequent steps run a closed loop over target rows only
  (embed → per-layer LN/fused-qkv/norms/rope → scatter K/V back into the target-row region of the
  full buffers → attention Q=target × K/V=full buffers → out/gate/MLP/gate → final takes only target
  rows). Numerically strictly isomorphic to the full path (the target-segment attention is the same
  rect call); same-seed pixel difference of 0.2-0.8/255 is cross-process MPS nondeterminism noise.
  Single image 8 steps 41.8 → 22.6 s (1.9×), 3 images 94.1 → 41.3 s (**2.3×**), now ahead of the
  official edit pipeline even with its KV cache. `QWEN_NO_STEP_CACHE=1` falls back to the full path;
  `QWEN_FLASH_ATTN=1` / `QWEN_NO_PREFIX` diagnostic modes automatically disable the cache. Layout
  changes (switching images/resolution in REPL) invalidate and rebuild automatically; in REPL, with
  the same layout, even the first step's full pass is reused.
- VAE 3×3 convolutions changed from per-pixel scalar kernels (0.35-0.6 TFLOPS) to a 9-tap
  accumulating GEMM: activations are first copied into a ring zero-padded buffer, each tap becomes a
  dense tensor view expressed purely as pointer offsets, and nine `matmul2d` fp32
  multiply_accumulate passes accumulate (14.9 TFLOPS; accumulation-order differences vs PyTorch are
  in the 1e-5 noise band). VAE decode ~45 s → ~2 s (**~25×**), full 40-step image 147.7 → ~84 s.
  The original scalar kernel remains as the `H3_VAE_SCALAR_CONV=1` fallback.
- The text-tower KV cache was analyzed to save only 0.6% at 1024² (text-row share) and is not
  implemented for now.

## Tests and equivalence verification

```sh
make host-tests       # host-side math/JSON/weight tools
make dit-tests        # DiT: per-segment blocks + full 32-layer forward
make text-tests       # text encoder: bit-exact tokenization + 36-layer hidden states
make vae-tests        # VAE: ~50 segments, bit-exact / noise-band
make pipeline-tests   # scheduler/RoPE/RNG components
```

- Fixtures for DiT / text encoder / VAE require a
  [diffusers](https://github.com/huggingface/diffusers) Python environment and model weights;
  generate them with `make *-fixture` or `tests/gen_*.py` (venv path in the script header comments).
- Fixture sizes: `pipeline_fixture` (84 KB) and `dit_fixture` (2.5 MB) ship with the repo and the
  corresponding tests run out of the box; `text_fixture` (58 MB) is optional; `vae_fixture` (17 GB)
  is not checked in and must be generated locally.
- Deep bf16 paths (text encoder, full DiT forward) carry intrinsic chaotic noise; the acceptance
  criterion is "distance from the reference implementation ≤ √2 × the reference's own
  bf16-vs-fp32 noise band", while component-level tests are bit-exact / 1e-6 level.

## Architecture

```
qwen-metal/
├── h3_gpu.{h,m}          # Metal runtime (kernel dispatch, MPSGraph, caches)
├── h3_shaders.metal      # All Metal kernels (GEMM/attention/RoPE/conv/…)
├── h3_weights.*          # safetensors streaming loader (shard index)
├── qwen_host.*           # Architecture constants and host math
├── qwen_text_encoder.*   # Qwen3-VL text tower (36 layers, mrope, pre-final-norm)
├── qwen_dit.*            # 7B single-stream DiT (32 layers, block-causal attention)
├── qwen_vae.*            # VAE decoder (Wan lineage; the image path is a pure 2D network)
├── qwen_pipeline.*       # Scheduler / RoPE tables / denoise loop / CFG / decode
├── qwen_png.*            # Minimal PNG encoder
├── main.c                # CLI
└── tests/                # Per-segment equivalence tests and fixture generators
```

Precision notes (semantic details aligned with the official diffusers reference):

- The text encoder output is taken **before the final RMSNorm** (matching the hook behavior of
  QwenImage21Pipeline); the system prompt ("Comprehend and analyze the provided prompt.") prefix is dropped
- `inv_freq` is a module buffer and gets quantized by `to(bf16)` — the RoPE table replicates this
- The text tower uses rotate_half half-allocation pairs for RoPE, the DiT interleaves adjacent pairs;
  both kernel families coexist
- RMSNorm / SwiGLU / attention scale all replicate PyTorch CPU bf16's per-rounding-point semantics
- Scheduler: exponential dynamic shifting (`mu = f(seq_len)`) + terminal 0.02 stretch + trailing sigma 0

## Debug environment variables

| Variable | Effect |
|---|---|
| `QWEN_TEXT_DUMP=<dir>` | Dump text-encoder intermediates segment by segment (equivalence debugging) |
| `QWEN_VAE_DUMP=<dir>` | Dump VAE intermediates segment by segment |
| `H3_DISABLE_MPS_LINEAR=1` | Disable the linear MPSGraph fast path (pure kernel path) |
| `H3_NAX=0` | Disable M5 TensorOps |

## Known limitations

- ti2i's CFG keeps one cross-step K/V cache per branch (with cfg > 1, cond and neg each hold one,
  roughly doubling peak within a request; the cache has a single-request lifetime and is freed when
  the request ends)
- Windows / Linux are not supported (deeply Metal-dependent)

## Acknowledgments and license

- The Metal runtime, safetensors loading, and terminal components are derived from
  [h3-metal](https://github.com/antirez/h3-metal) (MIT, Salvatore Sanfilippo),
  with the original `h3_` naming preserved; its
  [linenoise](https://github.com/antirez/linenoise) line-editing library is BSD-style licensed
  (antirez / Pieter Noordhuis, notice preserved in the source file headers)
- Model weights are copyright Qwen and subject to their release license
- This project's code license: see `LICENSE` (MIT)
