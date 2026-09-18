# Report: HF pretrained molecular generator

本文记录 `torch-molecule` 接入 Hugging Face 预训练分子生成模型的工作：对外一个类 `HFPretrainedMolecularGenerator`，sklearn 风格 `fit` / `generate`，与 `HFPretrainedMolecularEncoder` 对称。依据仓库内的 `generation_model.md`、`issues_to_be_fixed.md` 以及 `torch_molecule/generator/pretrained/` 的现有代码；不把未实现的接口写成已完成。

---

## 1. Purpose

现有 `HFPretrainedMolecularEncoder` 只用 Hugging Face `AutoModel` 做编码，不做生成。仓库里已有 8 个自研生成器（`LSTMMolecularGenerator`、`MolGPTMolecularGenerator`、`DigressMolecularGenerator`、`GDSSMolecularGenerator`、`GraphDITMolecularGenerator`、`DeFoGMolecularGenerator`、`JTVAEMolecularGenerator`、`GraphGAMolecularGenerator`），它们**没有被替换**。

这次新增的是第三条路径：从 Hugging Face Hub 加载已预训练的生成权重，用**一个**对外类 `HFPretrainedMolecularGenerator` 提供与 LSTM / MolGPT 相同的 sklearn 风格接口：

- `fit()` 无数据：下载并加载 Hub 权重
- `fit(smiles)`：加载预训练权重后再微调
- `generate()`：返回 `List[str]` SMILES

六个 Hub 模型的架构不同（因果 LM、seq2seq SELFIES、Fragment-SELFIES + 官方推理引擎），无法共用同一个 `generate()`。家族分流（`novomolgen` / `gp_molformer` / `molgen` / `molexar` / fallback `causal_lm`）只在 `generator/pretrained/` 内部完成。用户不直接实例化 `families/` 里的类；那些模块是内部 dispatch，不是对外 API。

数据集层（`load_qm9()`、`load_zinc250k()`、`SMILESDataset`、`MolecularInputChecker`）没有改。用户始终传入 SMILES、始终拿到 SMILES。SELFIES / Fragment-SELFIES 转换只发生在 `generator/pretrained/` 内部。

---

## 2. What we added (feature work)

对应 `generation_model.md` 的 Phase 1–5。**Phase 6（文档站点 API 页、README 模型列表、CI 分层）仍未完成**，见第 3 节末尾。

### 2.1 对外 API

```python
from torch_molecule import HFPretrainedMolecularGenerator
```

导出链：`torch_molecule/generator/pretrained/__init__.py` → `torch_molecule/__init__.py`（已加入 `__all__`）。

与 encoder 的对称关系：

| | `HFPretrainedMolecularEncoder` | `HFPretrainedMolecularGenerator` |
|---|---|---|
| `fit()` 无参 | 加载 `AutoModel` | 加载 `AutoModelForCausalLM` / `AutoModelForSeq2SeqLM`，或 Molexar 官方 `MolexarInference` |
| `fit(X)` | 不支持 | 可选微调 |
| 主方法 | `encode()` → embedding | `generate()` → SMILES |

**`fit()` 语义与 LSTM 不同：**

- `LSTMMolecularGenerator.fit(X_train)`：从零初始化网络并在 SMILES 上训练。
- `HFPretrainedMolecularGenerator.fit()`：无 `X` 时只从 Hub 拉预训练权重；`fit(smiles)` 是加载权重后再微调，不是从零训练。

`generate(n_samples=...)` 返回 `List[str]` SMILES。MolGen / Molexar 解码失败的条目会被丢掉，返回长度可以小于 `n_samples`（见问题 2、7）。

条件微调参数 `y` 目前会 warn 后忽略，走无条件语言模型微调（见「仍未做」）。

### 2.2 用户侧始终是 SMILES

| 方向 | 约定 |
|---|---|
| `fit(X)` 输入 | `List[str]` SMILES |
| `generate()` 输出 | `List[str]` SMILES |
| SELFIES / Fragment-SELFIES | 仅内部：`_encode_inputs()` / `_decode_outputs()` |
| 数据集 loader | **未改** |

内部转换（`modeling_pretrained.py`）：

- MolGen（seq2seq）：`smiles_to_selfies` / `selfies_to_smiles`
- Molexar：`smiles_to_fragment_selfies` / `fragment_selfies_to_smiles`
- NovoMolGen / GP-MoLFormer / fallback causal LM：直接用 SMILES（生成后去掉空格）

### 2.3 六个 Hub 模型与内部家族

| 模型 | `repo_id` | 内部 family | 表示 | 加载 / 生成要点 |
|---|---|---|---|---|
| NovoMolGen | `chandar-lab/NovoMolGen_32M_SMILES_BPE` | `novomolgen` | SMILES + BPE | 因果 LM；`revision` 默认 `hf-checkpoint` |
| GP-MoLFormer | `ibm-research/GP-MoLFormer-Uniq` | `gp_molformer` | SMILES | 因果 LM；`scaffold=` 骨架补全；tokenizer 默认 `ibm-research/MoLFormer-XL-both-10pct`；IBM remote code 需要 `transformers<=4.56.2` |
| MolGen-large | `zjunlp/MolGen-large` | `molgen` | SELFIES | seq2seq；默认苯环 prefix |
| MolGen-large-opt | `zjunlp/MolGen-large-opt` | `molgen` | SELFIES | 同上，权重已偏 QED / p-logP |
| Molexar-10M-base | `fairydance/molexar-10m-base` | `molexar` | Fragment-SELFIES | 官方 `MolexarInference` |
| Molexar-10M-omni | `fairydance/molexar-10m-omni` | `molexar` | Fragment-SELFIES | 同引擎；`conditions` / 性质键等 omni kwargs |

未知 `repo_id`：`resolve_family()` 返回 fallback `causal_lm` 并 `warnings.warn`（问题 6 修过后，前缀能匹配的变体如 `chandar-lab/NovoMolGen_157M` **不再**误报）。

`FAMILY_PREFIXES` 允许同一前缀下的变体（例如其它 NovoMolGen 尺寸）映射到对应家族，而不必全部写进 `KNOWN_REPOS`。

### 2.4 分阶段（Phase 1–5 已落地）

**Phase 1 — 骨架 + NovoMolGen（MVP）**

- `registry.py`、`modeling_pretrained.py`、`families/causal_lm.py`
- `__init__.py` 导出
- `tests/generator/hfpretrained.py` smoke（含 `@pytest.mark.integration` 的 NovoMolGen `fit()` + `generate()`）
- optional extra：`[hf-gen]`

**Phase 2 — GP-MoLFormer + SELFIES 工具**

- `utils.py`：`smiles_to_selfies` / `selfies_to_smiles`
- `families/causal_lm.py`：de novo + `scaffold=`
- `compat.py`：`transformers>=4.57` 时对 GP-MoLFormer raise；为 IBM remote code 提供 `transformers.onnx` stub
- extra：`[gp-molformer]`（`transformers>=4.40,<=4.56.2`）

**Phase 3 — MolGen-large / MolGen-large-opt**

- `families/seq2seq.py`
- `generate(prefix_selfies=...)`；未提供时使用默认苯环 SELFIES（见「仍未做」）
- extra：`[molgen]`（`selfies>=2.1.0`，无上界）

**Phase 4 — Molexar**

- `utils.py`：Fragment-SELFIES 转换
- `families/molexar.py`：wrap 官方 `MolexarInference`（de novo、`start_smiles` / fragment 约束、omni `conditions`）
- extra：`[molexar]`

**Phase 5 — 微调与本地存盘**

- `finetune.py`：按家族 dispatch
  - 因果 LM（NovoMolGen / GP-MoLFormer / fallback）：next-token prediction（`finetune_causal_lm`）
  - MolGen seq2seq：denoising（`finetune_seq2seq`，问题 9 之后：mask encoder、labels 保持干净）
  - Molexar：官方 training template 后再走因果 LM loop（`finetune_molexar`）
- `checkpoint.py` + `save_to_local` / `load_from_local`：HF 目录（`save_pretrained`）+ `hf_generator_metadata.json`
- `save_to_hf()` **仍是** `NotImplementedError`
- `load_from_hf()` / `load()` 无本地 path 时等价于 `fit()`

**Phase 6 — 仍开放**（见第 3 节）。

### 2.5 目录布局

```
torch_molecule/generator/pretrained/
├── __init__.py                      # 只导出 HFPretrainedMolecularGenerator
├── modeling_pretrained.py           # 唯一对外类
├── registry.py                      # repo_id → family
├── utils.py                         # SMILES ↔ SELFIES / Fragment-SELFIES
├── finetune.py                      # 家族微调
├── checkpoint.py                    # 本地 metadata
├── compat.py                        # GP-MoLFormer / transformers 兼容
└── families/
    ├── __init__.py
    ├── causal_lm.py                 # NovoMolGen, GP-MoLFormer, fallback
    ├── seq2seq.py                   # MolGen-large, MolGen-large-opt
    └── molexar.py                   # Molexar base / omni
```

### 2.6 Optional extras（`pyproject.toml`）

| extra | 依赖 |
|---|---|
| `[hf-gen]` | `transformers>=4.40`, `accelerate` |
| `[molgen]` | `selfies>=2.1.0`, `transformers>=4.40`, `accelerate` |
| `[gp-molformer]` | `transformers>=4.40,<=4.56.2`, `accelerate` |
| `[molexar]` | `fragment-selfies>=1.0.0`, `transformers>=4.40`, `accelerate`, `loguru`, `molexar @ git+https://github.com/fairydance/Molexar.git` |

`selfies` **没有**在 `pyproject.toml` 里 pin `<3`（老师要求；问题 5 用 README / `install.rst` 说明）。

### 2.7 存盘

- `save_to_local(path)`：`model.save_pretrained` + `tokenizer.save_pretrained` + `hf_generator_metadata.json`
- `load_from_local(path)`：读 metadata，再按家族从该目录加载
- `save_to_hf(...)`：`NotImplementedError`（「HFPretrainedMolecularGenerator does not support saving to Hugging Face.」）

注意：`fit()` **每次**都会 `_load_pretrained()` 从 Hub 再拉一遍。即使刚 `load_from_local`，再调用 `fit()` 仍会覆盖为 Hub 权重。老师要求这一轮先不动（见「仍未做」）。

### 2.8 微调：MolGen 是 denoising，不是 identity copy

问题 9 修完后，仅 `finetune_seq2seq` 改变目标：

- encoder `input_ids`：非 special token 以 `mask_prob=0.15` 换成 `<mask>`
- `labels`：仍是干净序列
- `fit(X)` 签名不变
- **不影响** NovoMolGen、GP-MoLFormer、Molexar、LSTM

### 2.9 用法片段（NovoMolGen）

```python
from torch_molecule import HFPretrainedMolecularGenerator

model = HFPretrainedMolecularGenerator(
    repo_id="chandar-lab/NovoMolGen_32M_SMILES_BPE",
)
model.fit()  # 只加载 Hub 预训练权重，不训练
smiles_list = model.generate(n_samples=10)

# 微调：仍传 SMILES（与 LSTM 不同：这里是在预训练权重上继续训）
model.fit(["CCO", "CC(=O)O", "c1ccccc1"])
```

---

## 3. Issues we found and how we handled them

依据 `issues_to_be_fixed.md`。原则：进模型（训练数据）坏样本默认 **raise**，失败要带 index；出模型（生成）坏样本可以丢掉，但 `n_samples` 是**尝试条数**，返回长度 = 解码成功条数，并留下计数警告。

问题 2（MolGen 用 `""` 撑长度）和问题 7（Molexar 静默变短）**原 bug 不同**；修完后共用同一条返回约定：不要补 `""`，成功几条返回几条。

### 3.1 问题 1 — `fit()` 坏 SMILES：keep raise，不 skip

**原问题：** `fit(X)` 里一条坏样本会让整次微调失败。讨论过是否 skip / 重试。

**决策 / 修法：** **不加重试、不加 skip。** 坏样本就 raise，与 `MolecularInputChecker` 一致。

- RDKit 过不了：`MolecularInputChecker` 给出带 index 的错误（`Invalid SMILES structure at index {idx}: ...`），`_validate_inputs` 聚合成 `ValueError`。`smiles_to_selfies` 里同样对 `MolFromSmiles is None` raise `ValueError(f"Invalid SMILES at index {idx}: ...")`。
- RDKit 过了、`selfies.encoder` 挂：见问题 4，同样 raise，不 skip。

**状态：solved。**

**关键文件：** `torch_molecule/utils/checker.py`、`torch_molecule/generator/pretrained/utils.py`、`modeling_pretrained.py`（`fit` → `_validate_inputs`）、`tests/generator/hfpretrained.py`（`test_smiles_to_selfies_invalid_smiles`）。

### 3.2 问题 2 — MolGen `generate` 用 `""` 占位，`len == n_samples` 看起来成功

**原问题（仅 MolGen 这条路径）：** 解码失败时 `selfies_to_smiles` 写入空串。`len(out) == n_samples` 像成功，空串比短列表更坑。

这与问题 7 **不是同一个原 bug**：这里是用 `""` **把长度撑满**；问题 7 是 Molexar **直接丢掉且不警告**。

**决策 / 修法：** 不要 `""`，也不要重试凑满。成功几条就返回几条（2 条有效 → 返回 2）。`n_samples` = 尝试条数。

例：`generate(n_samples=5)`，其中 2 条解不出：

```text
原来: ["c1ccccc1", "CCO", "", "CC(=O)O", ""]   # len=5，两个假成功
现在: ["c1ccccc1", "CCO", "CC(=O)O"]          # len=3，只留真分子
```

实现上 `_decode_outputs` 对 seq2seq 丢掉无效 SMILES，并 warn `got {k}/{n} valid SMILES`。

**状态：solved。**

**关键文件：** `utils.py`（`selfies_to_smiles`）、`modeling_pretrained.py`（`_decode_outputs`）、`tests/generator/hfpretrained.py`（`test_selfies_to_smiles_drops_invalid_entries`、`test_decode_outputs_molgen_drops_empty_and_warns`）。

### 3.3 问题 3 — `except Exception` 把 `ImportError` 吞成 `""`

**原问题：** `selfies_to_smiles` 和 `fragment_selfies_to_smiles` 都是 `except Exception: append("")`。依赖缺失、键盘中断等也会变成空串，且没有计数、没有日志。

**决策 / 修法：** **不要** `return_stats` 新 API。删掉 `except Exception`。分两种：

- 不该发生的错（`ImportError`、其它意外）→ **raise**，不要吞成 `""`
- 分子解不出来（`DecoderError`、RDKit `mol is None`、空串）→ **不要 raise**（否则问题 2 无法返回成功的 2/3 条）。丢掉这条，**warn**（例如 `dropped {n} invalid SELFIES` / `dropped {n} invalid Fragment-SELFIES`）

**状态：solved。**

**关键文件：** `torch_molecule/generator/pretrained/utils.py`。

### 3.4 问题 4 — RDKit-valid 但 `selfies.encoder` 失败（dummy `*`）

**原问题：** `smiles_to_selfies` 在 canonical 之后直接 `sf.encoder(canonical)`，没有接 `EncoderError`。`fit` 带着库异常全挂，看不出是第几条。dummy atom（`*CCO`、`[*]CCO`、`[*]c1ccccc1`）是典型触发点。

**决策 / 修法：** **不加 skip。** 接住后 **raise** 成带 index 的 `ValueError`：

```python
raise ValueError(
    f"SMILES at index {idx} is RDKit-valid but not SELFIES-encodable: {smiles_string}"
) from exc
```

**状态：solved。**

**关键文件：** `utils.py`（`smiles_to_selfies`）、`tests/generator/hfpretrained.py`（`test_smiles_to_selfies_encoder_error_has_index`）。

### 3.5 问题 5 — `selfies>=2.1` 无上界；老师不要求 pyproject pin `<3`

**原问题：** `pyproject.toml` 是 `selfies>=2.1.0`，没有上界。3.x 字母表若破坏兼容，安装器不会挡。

**决策 / 修法：** **不改 pyproject 的版本上界。** 老师意见：放到 README **Additional Packages**（以及 `docs/source/install.rst`），和其他可选依赖一样写明：

| Model | Required Packages |
|---|---|
| HFPretrainedMolecularEncoder | transformers |
| HFPretrainedMolecularGenerator | transformers |
| HFPretrainedMolecularGenerator (MolGen) | transformers, selfies 2.x (3.x not guaranteed) |
| HFPretrainedMolecularGenerator (GP-MoLFormer) | transformers<=4.56.2 |
| HFPretrainedMolecularGenerator (Molexar) | transformers, fragment-selfies, molexar |

README 另有安装示例：`pip install "selfies>=2.1"`、`pip install torch-molecule[molgen]`、`[gp-molformer]`、`[molexar]`。

`[gp-molformer]` extra 在 pyproject 里 **有** `transformers<=4.56.2`；MolGen 的 `selfies` 仍只有 `>=2.1.0`。运行时 GP-MoLFormer 还会在 `compat.ensure_gp_molformer_transformers_compat()` 对 `>=4.57` raise `ImportError`。

**状态：solved（文档，不是 pyproject 给 selfies 加上界）。**

**关键文件：** `README.md`、`docs/source/install.rst`、`pyproject.toml`、`compat.py`。

### 3.6 问题 6 — 未知 `repo_id` 警告只查精确 `KNOWN_REPOS`

**原问题：** `__init__` 只查 `KNOWN_REPOS` 的完整字符串。`chandar-lab/NovoMolGen_157M` 会被 `FAMILY_PREFIXES` 正确分成 `novomolgen`，却仍警告 “family may not be implemented”。

**决策 / 修法：** 在 `resolve_family()` 之后，**只对 fallback `causal_lm` 警告**。

**状态：solved。**

**关键文件：** `modeling_pretrained.py`（`__init__`）、`registry.py`（`resolve_family`）、`tests/generator/hfpretrained.py`（`test_known_family_prefix_does_not_warn_unknown_repo`、`test_unknown_repo_fallback_warns`）。

### 3.7 问题 7 — Molexar `generate()` 列表变短且无警告

**原问题（Molexar，与问题 2 不同）：** `_decode_outputs` 对 Molexar `if smiles` 过滤，解码失败直接丢掉。调用方要 10 条可能拿到 6 条，**没有警告**。

**决策 / 修法：** 要 10 条、只有 6 条成功，就用这 6 条，**不要补 `""` 凑满**。缺的是警告，例如 `got 6/10 valid SMILES`。

修完后与问题 2 **共用返回约定**（成功几条返回几条 + 计数 warn），但原缺陷分别是「假满长度」vs「静默变短」。

**状态：solved。**

**关键文件：** `modeling_pretrained.py`（`_decode_outputs`，seq2seq 与 molexar 共用 warn）、`utils.py`（`fragment_selfies_to_smiles`）。

### 3.8 问题 8 — 边角化学：confirmed，不为转换层加特殊 case

**原问题：** 担心电荷、立体、dummy `*`、叠氮等边角 SMILES 需要单独一套转换逻辑。

**结论（已扫 12 大类、约 45 条 SMILES，`selfies` 2.1.1）：**

| 结果 | 条数 |
|---|---|
| OK | 40 |
| RDKit 拒 | 1 |
| `EncoderError` | 4（dummy `*`；`[*]CCO` 在 dummy 和 Molexar 挂点里各计一次） |
| decode 挂 | 0 |

会 roundtrip 的不必为转换层加特殊处理，包括：电荷、两性离子、四面体/顺反/联烯、萘/桥环/螺环/大环/三元环、吡啶、Kekulé 苯、三键、过氧、高价 S/P、N-oxide、硝基、自由基、卡宾、显式氢、有机叠氮 `CCN=[N+]=[N-]`、重氮、同位素、`.` 断开、Si/Se/膦/硼酸根、`[Cu+2]` / `[Fe]` / 类格氏。

会失败的走问题 1 / 4（进模型 raise），不是新分支：

| SMILES | 卡在哪 | 行为 |
|---|---|---|
| `C[N-]=[N+]=N` | RDKit 不认（价态） | **raise** 问题 1。有机叠氮请用 `CCN=[N+]=[N-]` |
| `*CCO` | RDKit 能吃，`selfies.encoder` 挂 | **raise** 问题 4 |
| `[*]CCO` | 同上 | **raise** 问题 4 |
| `[*]c1ccccc1` | 同上（Molexar 挂点写法） | **raise** 问题 4 |

只有 **MolGen** 走 SMILES ↔ SELFIES，才会在 `*` 上撞问题 4。NovoMolGen / GP-MoLFormer / LSTM 不走这条转换。

**决策：** 转换层不加边角化学 special case。失败路径已由问题 1、4 覆盖。`issues_to_be_fixed.md` 提到可选把 3 个 `*` 和 1 个坏叠氮加进回归测试；当前 `tests/generator/` **没有**这组 SMILES 作为独立回归用例。

**状态：confirmed；no extra conversion cases。**

**关键文件：** `utils.py`（通用 raise / drop，无化学分类表）。

### 3.9 问题 9 — MolGen 微调曾是 identity reconstruction

**原问题：** MolGen 预训练是 **denoising seq2seq**（损坏的 SELFIES → 还原完整 SELFIES）。实现曾把 `labels = input_ids`，等于 identity copy，和论文目标不一致。

**决策 / 修法：** **只改** `finetune_seq2seq`：

- `corrupt_token_ids(..., mask_prob=0.15)`：非 special token 换成 tokenizer 的 `<mask>`
- `labels` 仍是干净序列
- `fit(X)` 签名不变
- NovoMolGen / GP-MoLFormer / Molexar / LSTM 微调不变

**状态：solved（fixed）。**

**关键文件：** `torch_molecule/generator/pretrained/finetune.py`（`corrupt_token_ids`、`finetune_seq2seq`）、`tests/generator/test_finetune.py`（`test_corrupt_token_ids_masks_non_special_tokens`、`test_finetune_seq2seq_labels_stay_clean`）。

### 3.10 仍未做

`issues_to_be_fixed.md` 写明这一轮不做，以及 `generation_model.md` Phase 6 仍是未勾选：

1. **`fit()` 在 `load_from_local` 之后仍会从 Hub 再加载。** `fit()` 无条件调用 `_load_pretrained()`（无 `local_path` 时用 `self.repo_id`）。老师要求先不动。
2. **MolGen 默认苯环 prefix。** `families/seq2seq.py` 中 `DEFAULT_MOLGEN_PREFIX_SELFIES = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"`；未传 `prefix_selfies` / `scaffold` 时仍用它。这一轮明确不改默认。
3. **Phase 6 文档与 CI**
   - `docs/source/api/generator.rst` **尚未**收录 `HFPretrainedMolecularGenerator`（仍只有 8 个自研生成器）。
   - README「List of Supported Models → Generative Models」仍是 8 个自研模型，没有把 6 个 HF repo 列进去（Additional Packages 表已有 generator extras，那是问题 5，不是 Phase 6 的模型列表）。
   - CI：`pyproject.toml` 已声明 pytest marker `integration`，测试里 Phase 1/2/3/4 的 Hub 下载用例标了 `@pytest.mark.integration`；仓库 `.github/workflows/` 目前只有 `docs.yml`，**没有**「Phase 1 必跑、Phase 3+ 用 optional marker」的测试 CI。
4. **`save_to_hf()` 对本类未实现**（`NotImplementedError`）。不要把它写成可用。
5. **带 `y` 的条件微调未实现。** `fit(X, y)` 若 `y is not None` 会 `UserWarning`（「Conditional fine-tuning with y is not implemented yet」），然后 `y = None`，继续无条件 LM 微调。

---

## 4. Current data flow (SELFIES models)

用户侧不变：`load_zinc250k().data` 仍是 SMILES；`fit` / `generate` 仍收发 SMILES。

```
用户 SMILES
    │
    ▼
HFPretrainedMolecularGenerator.fit / generate
    │
    ├─ MolGen fit
    │     SMILES → _validate_inputs (RDKit)
    │            → smiles_to_selfies          # 坏样本 raise（问题 1 / 4）
    │            → tokenize
    │            → finetune_seq2seq           # denoising：mask encoder，labels 干净
    │
    ├─ MolGen generate
    │     prefix SELFIES（默认苯环，或 prefix_selfies= / scaffold=）
    │            → seq2seq model.generate
    │            → selfies_to_smiles          # 解不出：warn + drop，不补 ""
    │            → 可能再 warn got k/n valid SMILES
    │
    ├─ Molexar
    │     fit:    SMILES → smiles_to_fragment_selfies → 官方 training template → causal LM loop
    │     generate: MolexarInference → fragment_selfies_to_smiles → drop invalids + warn
    │
    └─ Causal LM（NovoMolGen / GP-MoLFormer / fallback）
          SMILES 直接 tokenize / 生成；不走 smiles_to_selfies /
          selfies_to_smiles / smiles_to_fragment_selfies / fragment_selfies_to_smiles
          NovoMolGen: BOS → generate；默认 revision hf-checkpoint
          GP-MoLFormer: de novo 或 scaffold= 前缀
```

---

## 5. Files touched (high level)

**新增（生成器核心）**

- `torch_molecule/generator/pretrained/modeling_pretrained.py`
- `torch_molecule/generator/pretrained/registry.py`
- `torch_molecule/generator/pretrained/utils.py`
- `torch_molecule/generator/pretrained/finetune.py`
- `torch_molecule/generator/pretrained/checkpoint.py`
- `torch_molecule/generator/pretrained/compat.py`
- `torch_molecule/generator/pretrained/__init__.py`
- `torch_molecule/generator/pretrained/families/causal_lm.py`
- `torch_molecule/generator/pretrained/families/seq2seq.py`
- `torch_molecule/generator/pretrained/families/molexar.py`
- `torch_molecule/generator/pretrained/families/__init__.py`

**导出与依赖**

- `torch_molecule/__init__.py`（加入 `HFPretrainedMolecularGenerator`）
- `pyproject.toml`（`[hf-gen]` / `[molgen]` / `[gp-molformer]` / `[molexar]`，以及 pytest `integration` marker）

**测试**

- `tests/generator/hfpretrained.py`
- `tests/generator/test_finetune.py`
- `tests/generator/test_causal_lm.py`
- `tests/generator/test_seq2seq.py`
- `tests/generator/test_molexar.py`

**文档（问题 5；不是 Phase 6 API 页）**

- `README.md` Additional Packages
- `docs/source/install.rst` Additional Packages

**未改（按设计）**

- `torch_molecule/datasets/*`（loader、CSV）
- `torch_molecule/encoder/pretrained/*`
- 8 个自研生成器实现

**Phase 6 仍未改**

- `docs/source/api/generator.rst`
- README 生成模型一览表（仍 8 个自研）

---

## 6. How to try it

```bash
pip install -e ".[hf-gen]"
```

按模型再装 extras：

```bash
pip install -e ".[molgen]"          # MolGen：selfies 2.x；3.x 不保证
pip install -e ".[gp-molformer]"    # pins transformers<=4.56.2
pip install -e ".[molexar]"         # fragment-selfies + molexar
```

NovoMolGen 推理：

```python
from torch_molecule import HFPretrainedMolecularGenerator

model = HFPretrainedMolecularGenerator(
    repo_id="chandar-lab/NovoMolGen_32M_SMILES_BPE",
)
model.fit()
print(model.generate(n_samples=5))
```

需要下载 Hub 权重的测试标了 `@pytest.mark.integration`，例如：

```bash
pytest tests/generator/hfpretrained.py -m "not integration"
pytest tests/generator/test_finetune.py
```
