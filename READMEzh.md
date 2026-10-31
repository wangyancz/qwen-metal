# qwen-metal

[English](README.md) | 简体中文

原生 **Qwen-Image-2.1** 文生图推理引擎，面向 Apple Silicon（Metal）。

纯 C11 + Objective-C 实现，零 Python 运行时依赖：文本编码器（Qwen3-VL 文本塔）、
7B DiT、VAE 解码器与 FlowMatch 调度器全部运行在自研 Metal 内核与 MPSGraph 之上，
一条命令从提示词到 PNG。

```
qwen generate -d /path/to/Qwen-Image-2.1.models \
    -p "a photorealistic red apple on a rustic wooden table, soft morning light" \
    --width 512 --height 512 --steps 40 --seed 7 --out apple.png
```

## 特性

- **全 Metal 推理**：DiT 与文本编码器 bf16（fp32 累加）、VAE fp32；M5 TensorOps 线性层加速，旧芯片自动回退
- **完整管线**：模板分词 → 文本编码 → 动态 shifting 调度器 → 32 层单流 DiT（块级因果注意力 + causal_condition）→ VAE 解码 → PNG
- **双检查点**：base（40 步）与 turbo（官方 8 步蒸馏 `sample_sigmas`）自动识别
- **分类器自由引导（CFG）**：`--cfg` + `--negative`
- **LoRA 适配器**：单文件 safetensors（PEFT 命名），加载时把
  `scale · B@A` 融合进权重副本；CLI/REPL 可随时挂载、调强度、摘除
- **等价性驱动开发**：每个组件（分词、文本编码、DiT 逐段、VAE 逐段、调度器、RoPE）
  都有对 diffusers/transformers 参考实现的逐位/噪声带对齐测试

## 环境要求

| 项目 | 要求 |
|---|---|
| 芯片 | Apple Silicon（M5 Max 开发验证；TensorOps 路径需 M5，其余芯片自动回退普通内核） |
| 系统 | macOS（需 Metal / MPSGraph 运行时） |
| 工具链 | Xcode 命令行工具（clang + ObjC + Metal 框架）、zlib（系统自带） |
| 统一内存 | **16 GB 起**（1024² 峰值约 17 GB）；官方全尺寸档峰值 ≤ 47 GB，64 GB 充裕 |
| 磁盘 | 模型权重 31 GB（base）/ 30 GB（turbo） |

## 模型准备

从 Qwen-Image-2.1 官方发布渠道下载 diffusers 布局的权重目录
（`model_index.json` + `transformer/` + `text_encoder/` + `vae/` + `scheduler/` + `processor/`），
例如：

```
/path/to/Qwen-Image-2.1.models          # base
/path/to/Qwen-Image-2.1-turbo.models    # turbo（可选）
```

注意：模型权重的使用遵循其官方许可协议。

## 构建

前置要求：安装 Xcode 命令行工具（`xcode-select --install`，提供 clang、
Objective-C 与 Metal 框架）；zlib 与 Accelerate 由 macOS 系统自带。

```sh
cd qwen-metal
make clean && make     # 全量构建，约 2 秒
```

产物：

- `./qwen` — 命令行推理程序（`--info` / `generate`）
- `libqwen.a` — 静态库（含全部引擎与 Metal 运行时，可嵌入其他程序）

全部 make 目标：

| 目标 | 说明 | 需要模型权重 |
|---|---|---|
| `make` / `make all` | 构建 CLI 与静态库 | 否 |
| `make host-tests` | 宿主侧数学 / JSON / 权重工具测试 | 否 |
| `make pipeline-tests` | 调度器 / RoPE / RNG 组件测试 | 否（fixture 已随仓库） |
| `make dit-tests` | DiT 单块逐段 + 32 层全 forward | 是 |
| `make text-tests` | 文本编码器分词 + 36 层 hidden | 是 |
| `make vae-tests` | VAE ~50 个分段比对 | 是 |
| `make lora-tests` | LoRA 融合数值对拍 + 挂载/摘除生命周期 | 是 + LoRA 文件 |
| `make dit-fixture` 等 | 重新生成等价性 fixture | 是 + diffusers 环境 |
| `make clean` | 清理产物 | — |

不依赖模型权重的快速验证：

```sh
make host-tests pipeline-tests
```

## 使用

### 查看检查点信息（不加载权重）

```sh
qwen --info -d /path/to/Qwen-Image-2.1.models
```

### 文生图

```sh
qwen generate -d MODEL_DIR -p "提示词" [选项]
```

| 选项 | 说明 | 默认 |
|---|---|---|
| `-p TEXT` | 提示词（必填） | — |
| `--negative TEXT` | 反向提示词（配合 `--cfg > 1`） | — |
| `--width N` / `--height N` | 输出尺寸（32 的倍数） | 1024 × 1024 |
| `--steps N` | 去噪步数（turbo 自动取 8） | base 40 |
| `--seed N` | 随机种子（同种子可复现） | 0 |
| `--cfg F` | 引导强度（>1 启用 CFG） | 1.0 |
| `--lora PATH` | LoRA 适配器（safetensors，可重复叠加 ≤8，见下节） | — |
| `--lora-scale F` | 第 i 个 `--lora` 的强度（可重复按序配对；缺省 1.0，不钳制） | 1.0 |
| `--out PATH` | 输出 PNG 路径 | out.png |

示例：

```sh
# base，1024x1024
qwen generate -d $MODELS/Qwen-Image-2.1.models \
    -p "一只戴宇航员头盔的柴犬，皮克斯风格" --width 1024 --height 1024 --seed 42

# turbo，8 步快速出图（官方固定 sigmas 轨迹；--steps 在 turbo 下被忽略）
qwen generate -d $MODELS/Qwen-Image-2.1-turbo.models \
    -p "cyberpunk skyline at dusk, neon reflections" --width 512 --height 512

# CFG 引导
qwen generate -d $MODELS/Qwen-Image-2.1.models \
    -p "a cozy log cabin in snowy forest, warm window light" \
    --negative "blurry, low quality, watermark" \
    --cfg 4 --width 768 --height 512 --seed 3

# 多参考图：提示词用 <imageN> 指定第 N 张 --image（N = 出现顺序）
qwen generate -d $MODELS/Qwen-Image-2.1.models \
    -p "replace the apple in <image2> with a green pear, keep <image1> unchanged" \
    --image apple.png --image blue_apple.png --res 512 --steps 8 --seed 7
```

### 分辨率与比例

输出边长必须是 **32 的倍数**（管线 `multiple_of = vae_scale_factor × 2`）。
推荐直接使用官方 aspect-ratio 分桶表：每个比例三档固定像素，成本由
token 数（= 像素 / 256）决定，与长宽组合无关。**极致档即各比例的最大
分辨率**（≈ 7.0-7.1M 像素，27.0-27.7k token）：

| 比例 | 标准 | 高清 | 极致（最大） | 极致像素 |
|---|---|---|---|---|
| 1:1 | 1024 × 1024 | 2048 × 2048 | **2656 × 2656** | 7.05M |
| 3:2 | 1248 × 832 | 2528 × 1696 | **3264 × 2176** | 7.10M |
| 2:3 | 832 × 1248 | 1696 × 2528 | **2176 × 3264** | 7.10M |
| 4:3 | 1248 × 928 | 2400 × 1792 | **3040 × 2272** | 6.91M |
| 3:4 | 928 × 1248 | 1792 × 2400 | **2272 × 3040** | 6.91M |
| 16:9 | 1376 × 768 | 2752 × 1536 | **3520 × 1984** | 6.98M |
| 9:16 | 768 × 1376 | 1536 × 2752 | **1984 × 3520** | 6.98M |

说明：

- 本引擎没有 Python/MPS 参考栈的 conv2d 8M 像素硬限制，表中全部尺寸
  （含 3:2 极致 3264 × 2176）均已实测通过；更大的边长语法上可用但落在
  训练分布外，质量不保证。
- 峰值内存（phys_footprint，权重零拷贝映射后）：1024² 约 17 GB、
  512² 约 10 GB、1536 × 2752 约 34 GB、3264 × 2176 约 47 GB——
  **64 GB 机器官方全尺寸档畅跑**。
- 等比之外的任意 32 倍数组合也可用，但优先选表内桶——这是模型训练时
  见过的比例。

### 参考图编辑（ti2i）

```sh
qwen generate -d MODEL_DIR -p "把苹果的颜色改成绿色" \
    --image /path/to/ref.png [选项]
```

| 选项 | 说明 | 默认 |
|---|---|---|
| `--image PATH` | 参考图 PNG（RGBA，可重复传多张，≤8） | — |
| `--res N` | 参考图编码分辨率面积（32 对齐，比例随参考图） | 1024 |
| `--width N` / `--height N` | 输出尺寸（32 的倍数） | 随参考图编码尺寸 |

说明：

- 提示词描述「要改成什么」（换色/换风格/增删元素），参考图提供构图与内容基底。
- **提示词里可用 `<imageN>` 指定某张参考图**（N = `--image` 的出现顺序，
  1 起）。词面是普通文本（tokenizer 无此特殊 token），模型按训练分布
  将其与前面的第 N 个图组关联——与官方栈同机制同写法。
  例：`把 <image2> 里的苹果换成 <image1> 的颜色`。引用了不存在的编号
  时模型多半忽略（CLI 不校验，官方同款透传）。
- 支持 `--negative` + `--cfg > 1`（classifier-free guidance，与官方
  QwenImage21Pipeline 同语义：neg 分支带同款参考图、独立文本编码与
  K-V 缓存；cfg > 1 未提供 negative 时降回 1 并告警）。
- 输出与参考图编码比例不同时构图自动适配，建议两者比例接近。
- 与官方 QwenImage21Pipeline 同初始噪声对拍：平均像素差 0.42/255
  （1024²）；CFG 开启后 1.27/255（512²，引导差值放大浮点顺序噪声）。

### LoRA 适配器

```sh
qwen generate -d MODEL_DIR -p "Q版 一只柴犬" \
    --lora /path/to/lora.safetensors \
    --lora-scale 1.0 [选项]

# 多适配器叠加（--lora 可重复；--lora-scale 按序配对，缺省 1.0）
qwen generate -d MODEL_DIR -p "Q版 3D 一只柴犬" \
    --lora qingban.safetensors --lora three_d.safetensors \
    --lora-scale 0.7 --lora-scale 0.5 [选项]
```

- **作用面**：PEFT 命名的单文件适配器（`transformer_blocks.N.<module>
  .lora_A/lora_B.default.weight`），每个 block 的 7 个线性层
  （attn.to_q/k/v/out.0 + img_mlp.gate_layer/proj/out）全部融合；
  缺键的部分适配器允许（跳过并提示）。
- **多适配器叠加**：适配器栈上限 8，`W' = bf16RNE(fp32(W) + Σ_a
  scale_a·B_a@A_a)`——任何变更（追加/摘除/改强度）都从 base 一次性
  重融合全部适配器，与顺序无关、无舍入积累；重建一次 ~3-5 s
  （与栈大小无关），摘除全部即还原（输出与从未挂载逐位一致）。
- **实现是权重域融合**（fp32 累加 + round-to-nearest-even 舍入，与
  `tests/test_lora` 的独立手算对拍 ≤2 bf16 ulp）。推理路径零改动，
  CFG 双分支天然共享；对同一适配器重复 apply = 更新其强度。
- **无 `lora_alpha` 语义**（PEFT 缺省 scaling=1）：`--lora-scale` 就是
  全部强度；训练触发词（如「Q版」）是普通词面，直接写进提示词。
- **代价**：挂载期间 DiT 权重从零拷贝映射转为私有副本，footprint
  +14 GB（摘除即还原；base 权重仍在 page cache，切换/摘除无 I/O）。
- **REPL**：`/lora PATH` 追加到栈、裸 `/lora` 全部摘除、
  `/lora-scale F` 改最近挂载的强度、`/lora-drop PATH` 摘单个；
  `/show` 列出整栈。

### 交互式会话（REPL）

模型只加载一次，逐行输入提示词连续出图：

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

| 指令 | 说明 |
|---|---|
| `/width N`、`/height N` | 调整输出尺寸 |
| `/steps N` | 去噪步数 |
| `/seed N` | 种子（每张自动 +1） |
| `/cfg F` | 引导强度（>1 需 `/neg`） |
| `/neg TEXT` | 反向提示词（`/neg` 清空） |
| `/image PATH` | 添加参考图（可重复 ≤8，裸 `/image` 清空；此后 prompt 走 ti2i，可用 `<imageN>` 引用） |
| `/res N` | ti2i 输出分辨率档（裸 `/res` 重置 1024） |
| `/lora PATH` | 追加 LoRA 适配器到栈（裸 `/lora` 全部摘除；每次变更重融合 ~3-5 s） |
| `/lora-scale F` | 最近挂载适配器的强度（即时重融合） |
| `/lora-drop PATH` | 从栈中摘除单个适配器 |
| `/out PATH` | 输出路径模板（`%llu` 为图片序号，默认 `out-%llu.png`） |
| `/show`、`/help`、`/quit` | 查看 / 帮助 / 退出 |

ti2i 会话的输出尺寸：默认按 `/res` 档从第一张参考图推导；`/width`、
`/height` 显式设置后覆盖（须 32 的倍数）。CFG（`/neg` + `/cfg > 1`）在
ti2i 下同样生效。

### JSONL 服务模式（serve）

```sh
qwen serve -d $MODELS/Qwen-Image-2.1.models
```

常驻 JSONL worker：请求一行一个（JSON 对象），进度 / done / error 事件走
stdout、诊断走 stderr，任何宿主程序（Go / Python / Rust…）以子进程方式
spawn 并按行读写即可驱动同一协议。要点：

- 模型常驻（REPL 同款），vision 惰性创建；stdin EOF 退出，取消/空闲卸载 =
  宿主杀进程组
- 请求字段：`id/prompt/negativePrompt/width/height/steps/cfgScale/seed/
  images[]/output/lora/loraScale`（参考图 ≤8 张，图片为本地路径）
- 进度：`encode → sample（每步回调）→ decode → save` 的 0..1 总映射 +
  4 s 心跳；LoRA 同步（同路径同强度 no-op，否则摘挂重融合）
- 必须在仓库目录运行（`h3_shaders.metal` 按 cwd 相对加载）

### 性能实测（M5 Max 128 GB）

| 配置 | 单张耗时（不含模型加载） | 说明 |
|---|---|---|
| base，1024²，40 步 | **~68 s**（flash2 回退 83.7 s，**1.22×**） | 单步 ~1.5 s |
| ti2i 单图，1024²，8 步 | **~23 s**（cond KV cache 后；全量重算 41.8 s，**1.9×**） | 段数 4 |
| ti2i 多图 3 图，1024²，8 步 | **~41 s**（cond KV cache 后；全量重算 94.1 s，**2.3×**） | 段数 8 |
| turbo，512²，8 步 | **~3.1 s** | 首跑含 ~13 GB 权重冷盘读（+5 s） |
| turbo，256²，8 步 | **~1.4 s** | 峰值内存 ~33.5 GB |

满散热（风扇全速）、模型页缓存已热的同态实测（2026-10-30）；同参数
同 seed 与 `QWEN_FLASH_ATTN=1` 回退路径成对对照。连续重负载后 M5 Max
会热节流（t2i 8 步 13.7→21.9 s 递增），跨轮计时须同热态对照。

与官方 diffusers PyTorch MPS 栈（bf16 管线 + VAE fp32 + MPS F.pad 修复 +
turbo 固定调度）同参数同 seed 对比：

| 场景 | 官方 MPS | 本引擎 | 对比 |
|---|---|---|---|
| t2i 1024² 40 步 | 77.4 s | **68.5 s** | 1.13× |
| t2i 1024² 8 步 | 16.9 s | **15.3 s** | 1.10× |
| ti2i 单图 1024² 8 步 | 26.6 s | **22.6 s** | 1.18× |
| turbo 512² 8 步 | 3.6 s | **3.1 s** | 1.16× |

四项全面领先。ti2i 曾落后官方 1.57×，根因是官方编辑管线带 KV cache
（`use_kv_cache`）——已实现同款（见优化记录），现反超 1.18×。

构成（1024² 单步 ~1.5 s）：线性层 ~0.8 s（MPSGraph bf16，51-65 TFLOPS，
qkv 单图三 GEMM + MLP 融合图，每层 MPSGraph encode 4→3 次）、注意力
~0.15 s（fused SDPA，47-54 TFLOPS）、逐元素 / 调度 ~0.5 s；VAE 解码
整图仅 ~2 s。

峰值内存（phys_footprint）：1024² 约 17 GB、512² 约 10 GB、
1536 × 2752 约 34 GB、3264 × 2176 约 47 GB。

### 常驻内存构成（serve / REPL 模式）

`generate` 一次性模式用完即退，serve/REPL 的常驻构成与 footprint 行为
如下（`QWEN_SERVE_MEM=1` 时 serve 每请求在 stderr 打一行
live/footprint 报告）：

- **权重零拷贝映射不计 footprint**（见上），常驻的是激活/工作区/
  以下两类按设计驻留的结构
- **ti2i cond KV cache**：跨采样步复用条件行的每层 K/V——4 图 1024²
  输出约 **12 GB**（36 层 × 2 × joint 行数 × 4096 × 2 B；每张参考图按
  1024² 面积编码，token 数随宽高比变化），**单请求生命周期**（缓存行
  编码的是本次请求的提示词/参考图/权重，请求结束即释放——布局相同
  的下一个请求也不能复用，否则会静默沿用上一个请求的条件）
- **LoRA 融合副本**：adapter 激活期间 layer 权重为融合后的私有拷贝，
  bf16 全量约 **14 GB**（base 原件经 mmap shadow 引用，重建不做权重
  I/O）；摘除即释放。换强度/换适配器为**原位重建**（融合张量形状固定，
  写回现有缓冲，零新分配）——按「新分配→再释放旧块」实现会让 Metal 池
  每次变更棘轮一整个融合集
- **footprint 是 Metal 池高水位**：释放的 MTLBuffer 页留在进程内分配池
  不归还系统（见下），池按「见过的最大尺寸类」增长后平台化；不同尺寸
  的请求混合（参考图数量/比例变化）会推高平台，空闲卸载（宿主杀进程）
  或退出时全部归还。文本塔/DiT/VAE 的文本行工作区曾按「新分配→再释放」
  交错排队进单条命令缓冲，实测同样工作量的 GPU 执行慢 ~5×（MPS 中间
  资源滞留到 commit）——文本塔按 5 组分段提交后恢复

两个关键机制：

- **权重零拷贝映射（默认开启）**：权重不走私有缓冲拷贝，而是 mmap
  safetensors + `bytesNoCopy` 直包——文件页属 page cache，**不计入
  phys_footprint** 且可被 OS 按压力淘汰/按需重读（文本塔 16 GB +
  DiT 13 GB 由此移出内存预算）。`H3_ZERO_COPY_WEIGHTS=0` 可关闭
  （回退私有拷贝路径），`=transformer` 仅映射 DiT。
- **Metal 不把释放的 MTLBuffer 页归还系统**（进程内实测 live_bytes
  下降而 phys_footprint 恒定），池内复用仅限能装进空闲块的新分配。
  据此：VAE 卷积逐 conv 提交命令、pad 缓冲整链单块复用（最大档
  ~4.9 GB）、resnet/attention/上采样块边界即时提交；`generate` 模式
  在 prompt 编码后释放文本塔、采样后释放 DiT——未优化前大分辨率
  峰值曾达 117 GB。

优化记录：
- GEMM 曾走 16×16 标量分块内核（2.6 TFLOPS），恢复 MPSGraph bf16 快路后
  提升至 ~60 TFLOPS（**23 倍**，三种实现输出逐位一致），单步 10.0 → 3.1 s。
- 注意力自研标量 flash 反而慢 3.7 倍（rect 内核实为 K/V 带宽饱和，
  1.34 TB/s）；改用 MPP `matmul2d` TensorOps flash（64 查询 × 64 键 tile、
  coop fp32 累加器在线 softmax）后单步 3.1 → ~2.2 s。KVT=128 会超出
  32 KB threadgroup 内存上限，64 为实测最优点。
- DiT 图像注意力换 MPSGraph 原生 fused SDPA 算子
  （`scaledDotProductAttentionWithQueryTensor:...`，vision 塔同款）：
  微基准（`make bench-sdpa`，1024² 生产形状 32 heads × 128）实测
  47-54 TFLOPS，对自研 flash2 的 16 TFLOPS 快 **3.0-3.3 倍**（relL2
  0.0015）；40 步整图 100.3 → 77.8 s（**1.29 倍**）。`QWEN_FLASH_ATTN=1`
  可回退 flash2 内核做 A/B。ti2i 块因果 prefix 内核同样切换：不做
  539 MB 的 attnMask 矩阵，而是**按 run 拆分**——image run 的可见域
  恰是矩形 [0, run_end)，直接走 fused rect；text run 段内因果保留
  flash prefix 内核（行数 ~20 可忽略）。单图 8 步 58.9 → 40.1 s
  （**1.47×**）；同 seed 与 prefix 内核输出像素差 0.20/255（噪声带），
  ti2i fixture 门限按双实现本征标定放宽到 0.06（fused 累加序摆动
  ~0.001，与引擎固有非确定性同量级）。
- 注意力收尾后复确认（同噪声注入官方管线逐步对拍）：1024² 单图
  step0 lat relL2 0.0034（基线 0.0034）、终态 0.0153（基线 0.0161）、
  像素差 0.57/255——与官方全链路对齐无回归。多图 3 图 1024² 8 步
  95.5 s（flash2 回退同参数 160.4 s，**1.68×**）。
- 逐元素融合（2026-10-30）：q_pad 消除（fused rect 消费精确行数，
  仅 flash2 回退保留 64 行 tile pad）、层工作区复用（六个
  [rows, hidden] 激活缓冲每 forward 分配一次，t2i/ti2i 切行数时
  重分配）、q/k/v 预拼接（`qw_gpu_qkv_bf16`：MPSGraph 单图 3 matmul
  共享输入，权重保持原生 [out, in] 零拷贝不动，每层 encode 3→1）。
  `QWEN_BLOCK_PROFILE=1`（分段强制 submit 同步，诊断用）实测 input 段
  17.8→12.7 ms/层（编码税 -0.16 s/步），40 步整图 77.8→73.2 s（GPU
  热态下测得，收益为下界）；融合输出与三 GEMM 版**逐位一致**。经
  测量 gate+LN 融合仅 ~0.2 ms/层，数据不支持，未做；剩余 CPU 编码
  税（每层 ~5 次 MPSGraph encode × ~2-3 ms）与 GPU 执行部分重叠，
  进一步压缩需整步 GPU timeline 捕获定位，挂起。
- ti2i cond KV cache（2026-10-30）：cond/text 行的输入跨采样步不变
  （embed 固定、modulation 挂 t=0 槽、rope 固定），第一步全量跑并把
  每层 K/V 写入跨步保活的缓冲（`dit->cache_k/v`，单图 +4.3 GB、
  3 图 +8.8 GB），后续步只对 target 行闭环计算（embed → 每层
  LN/fused-qkv/norms/rope → K/V 散射回全缓冲 target 行区 → attention
  Q=target × K/V=全缓冲 → out/gate/MLP/gate → final 只取 target 行）。
  数值与全量路径严格同构（target 段 attention 是同一个 rect 调用），
  同 seed 像素差 0.2-0.8/255 为跨进程 MPS 非确定性噪声带。单图 8 步
  41.8 → 22.6 s（1.9×）、多图 3 图 94.1 → 41.3 s（**2.3×**），
  反超官方带 KV cache 的编辑管线。`QWEN_NO_STEP_CACHE=1` 回退全量
  路径；`QWEN_FLASH_ATTN=1` / `QWEN_NO_PREFIX` 诊断模式自动禁用
  缓存。布局变更（REPL 换图/换分辨率）自动失效重建；REPL 同布局
  连出时连第一步全量都可复用。
- VAE 3×3 卷积从逐像素标量内核（0.35-0.6 TFLOPS）改为 9-tap 累积 GEMM：
  激活先拷入圈零 pad 缓冲，每个 tap 成为纯指针偏移的稠密张量视图，
  九次 `matmul2d` fp32 multiply_accumulate 累加（14.9 TFLOPS，逐位
  累加序与 PyTorch 的差异在 1e-5 噪声带）。VAE 解码 ~45 s → ~2 s
  （**~25 倍**），40 步整图 147.7 → ~84 s。原标量内核保留为
  `H3_VAE_SCALAR_CONV=1` 回退路径。
- KV cache 经分析在 1024² 下仅节省 0.6%（文本行占比），暂不实施。

## 测试与等价性验证

```sh
make host-tests       # 宿主侧数学/JSON/权重工具
make dit-tests        # DiT：单块逐段 + 32 层全 forward
make text-tests       # 文本编码器：分词逐位 + 36 层 hidden
make vae-tests        # VAE：~50 个分段逐位/噪声带
make pipeline-tests   # 调度器/RoPE/RNG 组件
```

- DiT / 文本编码器 / VAE 的 fixture 需要
  [diffusers](https://github.com/huggingface/diffusers) Python 环境与模型权重，
  用 `make *-fixture` 或 `tests/gen_*.py` 生成（见脚本头注释中的 venv 路径）。
- fixture 体积：`pipeline_fixture`（84 KB）与 `dit_fixture`（2.5 MB）随仓库分发，
  对应测试开箱即跑；`text_fixture`（58 MB）可选；`vae_fixture`（17 GB）不入库，
  需自行生成。
- 深层 bf16 路径（文本编码器、DiT 全 forward）存在固有混沌噪声，
  验收标准为「与参考实现的距离 ≤ 参考自身 bf16-vs-fp32 噪声带的 √2 倍」，
  组件级测试则为逐位/1e-6 级。

## 架构

```
qwen-metal/
├── h3_gpu.{h,m}          # Metal 运行时（内核调度、MPSGraph、缓存）
├── h3_shaders.metal      # 全部 Metal 内核（GEMM/注意力/RoPE/卷积/…）
├── h3_weights.*          # safetensors 流式加载（shard 索引）
├── qwen_host.*           # 架构常量与宿主数学
├── qwen_text_encoder.*   # Qwen3-VL 文本塔（36 层，mrope，pre-final-norm）
├── qwen_dit.*            # 7B 单流 DiT（32 层，块级因果注意力）
├── qwen_vae.*            # VAE 解码器（Wan 血统，图像路径 = 纯 2D 网络）
├── qwen_pipeline.*       # 调度器/RoPE 表/去噪循环/CFG/解码
├── qwen_png.*            # 极简 PNG 编码器
├── main.c                # CLI
└── tests/                # 逐段等价性测试与 fixture 生成器
```

精度要点（与官方 diffusers 参考实现对齐的语义细节）：

- 文本编码器输出**取 final RMSNorm 之前**（与 QwenImage21Pipeline 的 hook 行为一致），
  系统 prompt（"Comprehend and analyze the provided prompt."）前缀丢弃
- `inv_freq` 作为模块 buffer 会随 `to(bf16)` 量化——RoPE 表按此复刻
- 文本塔 RoPE 为 rotate_half 半分配对，DiT 为相邻对交错旋转，两套内核并存
- RMSNorm / SwiGLU / attention scale 均复刻 PyTorch CPU bf16 的逐舍入点
- 调度器：指数动态 shifting（`mu = f(seq_len)`）+ terminal 0.02 拉伸 + 末位 sigma 0

## 调试环境变量

| 变量 | 作用 |
|---|---|
| `QWEN_TEXT_DUMP=<dir>` | 逐段 dump 文本编码器中间量（等价性排查） |
| `QWEN_VAE_DUMP=<dir>` | 逐段 dump VAE 中间量 |
| `H3_DISABLE_MPS_LINEAR=1` | 禁用 linear 的 MPSGraph 快路（纯内核路径） |
| `H3_NAX=0` | 禁用 M5 TensorOps |

## 已知限制

- ti2i 的 CFG 双分支各持一套跨步 K/V 缓存（cfg > 1 时 cond/neg 各一份，
  请求内峰值约翻倍；缓存为单请求生命周期，请求结束释放）
- Windows / Linux 不支持（深度依赖 Metal）

## 致谢与许可

- Metal 运行时、safetensors 加载、终端组件衍生自
  [h3-metal](https://github.com/antirez/h3-metal)（MIT，Salvatore Sanfilippo），
  保留原 `h3_` 命名；其中的
  [linenoise](https://github.com/antirez/linenoise) 行编辑库为 BSD 风格许可
  （antirez / Pieter Noordhuis，声明保留于源文件头）
- 模型权重版权归 Qwen 官方所有，遵循其发布许可
- 本项目代码许可证见 `LICENSE`（MIT）
