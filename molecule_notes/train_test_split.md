# Train / Test Split 实现方案

本文档记录 `torch-molecule` 中分子数据划分（train/test split）的设计与实现计划。  
背景：分子数据不能简单使用 sklearn 的 `train_test_split`，需支持 structure-aware 划分。

---

## 1. 设计原则

1. **Split 是数据工具，不是模型工具** — 不要写进 `BaseMolecularPredictor.fit()`。
2. **返回值仍是** `SMILESDataset` — 与现有 `load_qm9()` → `fit()` 流程对接。
3. `method` 支持 `random`、`scaffold`、`butina`、`size`。三路划分见 3.3。
4. **SizeShiftReg 的 coarsening/CMD 不属于 split** — 那是 SSR 模型的训练正则，split 只按原子数切分。
5. **默认** `method="random"` — 保证可复现、与旧脚本一致；文档说明 random 分数往往偏乐观。

---

## 2. 文件结构

```
torch_molecule/datasets/
  constant.py              # SMILESDataset 增加 train_test_split / subsample
  split.py                 # 新增：各划分方法实现
  __init__.py              # 导出 train_test_split, SMILESDataset

tests/datasets/
  test_split.py            # 新增单元测试

README.md                  # 更新：加载 → 划分 → fit 完整示例
```

**不要修改：** `predictor/*/modeling_*.py`、`base/predictor.py`（训练接口已足够）。

**依赖：** RDKit（已在 `pyproject.toml`）只负责 SMILES → 指纹。无需 DeepChem、无需 chemfp。

**增强版 Butina：** 不是 RDKit `ClusterData` 的稠密距离矩阵。生产路径是 **稀疏阈值邻居图 + exclusion sphere**（chemfp / Chalcedon 同一精确算法，内存 $O(n+E)$）。原文、RDKit、chemfp、Chalcedon 的对照表见下方 5.3；更长的文献笔记见 `molecule_notes/butina_optimization.md`。

---

## 3. 用户 API

### 3.1 挂在数据集上（推荐）

```python
from torch_molecule.datasets import load_qm9

data = load_qm9(local_dir="torchmol_data")
# subsample 仅用于本地调试 / CI，不要为了 Butina 而缩小 QM9
# data = data.subsample(n=5000, seed=0)

train, val = data.train_test_split(
    test_size=0.2,
    method="scaffold",   # "random" | "scaffold" | "butina" | "size"
    seed=42,
)

train, val = data.train_test_split(
    test_size=0.2,
    method="scaffold",   # "random" | "scaffold" | "butina" | "size"
    seed=42,
)

predictor.fit(train.data, train.target, val.data, val.target)
```

### 3.2 函数式

```python
from torch_molecule.datasets import train_test_split

train, test = train_test_split(data, test_size=0.2, method="random", seed=42)
```

### 3.3 三路划分（对标 MoleculeNet 80/10/10）

```python
train, val, test = data.train_val_test_split(
    train_size=0.8, val_size=0.1, test_size=0.1,
    method="scaffold",
    seed=42,
)
```

### 3.4 可选参数


| 参数                  | 适用 method                               | 说明                                                          |
| ------------------- | --------------------------------------- | ----------------------------------------------------------- |
| `test_size`         | 全部                                      | 测试集比例，默认 0.2                                                |
| `seed`              | random；scaffold/butina 的 `_group_split` | 随机种子，默认 42。Butina **聚类本身**用 index 平局，不用 seed                |
| `similarity_cutoff` | butina                                  | Tanimoto **相似度**阈值，默认 0.65（chemfp 语义；不是 DeepChem 距离 cutoff） |
| `use_csk`           | scaffold                                | 是否用 cyclic skeleton（全碳），默认 False                            |
| `direction`         | size                                    | `small_to_large`（默认）等                                       |


---

## 4. `split.py` 内部结构

```python
def train_test_split(dataset, test_size=0.2, method="random", seed=42, **kwargs):
    smiles = dataset.data
    y = dataset.target
    n = len(smiles)

    if method == "random":
        idx_train, idx_test = _random_split(n, test_size, seed)
    elif method == "scaffold":
        groups = _scaffold_groups(smiles, use_csk=kwargs.get("use_csk", False))
        idx_train, idx_test = _group_split(groups, test_size, seed=seed)
    elif method == "butina":
        groups = _butina_groups_or_oom(
            smiles,
            similarity_cutoff=kwargs.get("similarity_cutoff", 0.65),
        )
        idx_train, idx_test = _group_split(groups, test_size, seed=seed)
    elif method == "size":
        idx_train, idx_test = _size_split(smiles, test_size, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown split method: {method}")

    return _subset(dataset, idx_train), _subset(dataset, idx_test)
```

核心：**按样本切**（random, size） vs **按组切**（scaffold, butina）→ 共用 `_group_split`。

---

## 5. 各方法实现要点

### 5.1 Random split

- 与 sklearn 等价：`np.random.RandomState(seed).permutation(n)`。
- 固定 `seed` 保证可复现。
- 支持 `target is None`（如 ZINC）。
- **测什么泛化：** 同分布插值；分数往往偏高，仅作对照。

**参考：** MoleculeNet (Wu et al., 2018) — baseline split。

---

### 5.2 Scaffold split

**步骤：**

1. SMILES → RDKit Mol（无效则 `ValueError`，与 `MolecularInputChecker` 一致）。
2. `MurckoScaffold.GetScaffoldForMol(mol)` → scaffold Mol。
3. `MolToSmiles(scaffold)` 作为 group id；无环小分子用 `"_acyclic_"` 或规范 SMILES。
4. `dict[scaffold_smiles] -> list[index]`。
5. **组级别**分配到 train/test（同一 scaffold 不能跨集合）。

**组分配策略（对齐 DeepChem / scikit-fingerprints）：**

- 按组大小降序排列 scaffold 组。
- 贪心：将组填入 train 或 test，使 test 比例接近 `test_size`（或最小组优先进 test，测稀有骨架）。

**测试断言：**

- train 与 test 的 scaffold 集合 **交集为空**。
- 每个 index 恰好出现一次。
- 实际比例可能偏离 `test_size`（group split 正常现象，需在文档说明）。

**可选：** `use_csk=False`（默认，保留原子类型）；`True` 时用 `MakeScaffoldGeneric`（更粗）。

**测什么泛化：** 未见过的 Bemis–Murcko 骨架。

**参考：**

- Bemis & Murcko (1996) — 骨架定义。
- MoleculeNet (2018) — ML 标准协议。

**局限：** 差一个原子可能换 scaffold，但分子仍很相似（Walters 2024）；RDKit 实现与原文细节略有差异。

---

### 5.3 Butina split（增强 / 稀疏精确版）

代码：`torch_molecule/datasets/split.py` 中 `_butina_clusters` / `_butina_neighbor_lists` / `_butina_exclusion_spheres`。  
**精确 Taylor–Butina（Butina 1999）**，不是 BitBIRCH，也不是「先 subsample 再聚类」。

#### 为什么叫增强

朴素实现（DeepChem / Datamol / 直接 `Butina.ClusterData`）要先填满 condensed 距离：

$$
\text{dists 长度} = n(n-1)/2,\quad \text{内存 } O(n^2)
$$

QM9（$n \approx 133885$）这条路会 OOM（他人报告 15 万分子涨到几十～上百 GB）。  
**精确 Butina 其实只需要布尔邻居** $N(i,j)=\mathbf{1}[T(i,j)\ge t]$。增强版只存 $T \ge t$ 的边，再按原文做度数排序 + exclusion sphere。簇与小集上的 RDKit `ClusterData` **membership 一致**（`tests/datasets/test_split.py`）。


|        | 朴素 `ClusterData`                   | 本库增强版                                                    |
| ------ | ---------------------------------- | -------------------------------------------------------- |
| 算法     | 原文 Butina                          | **同一套** exclusion sphere                                 |
| 指纹     | 调用方 / DeepChem 1024 bit            | Morgan r=2，**2048** bit                                  |
| 阈值     | DeepChem 默认 **距离 0.6**（$T\ge 0.4$） | `**similarity_cutoff=0.65` 是相似度**（不是 Chalcedon 的距离 0.65） |
| 存储     | 全部 pair 距离                         | 稀疏邻接表 $O(n+E)$                                           |
| 搜索     | Python 填矩阵                         | RDKit `BulkTanimotoSimilarity`（C++/POPCNT）+ numpy 筛边     |
| 平局     | `(degree, index)` 降序（高 index 优先）   | **对齐 RDKit**（不是 chemfp 默认 `randomize`）                   |
| 度数     | 含自身（对角距离 0）                        | 含自身                                                      |
| QM9 全集 | OOM                                | 可跑完（Colab 约十几～几十分钟，不是 2 小时）                              |


chemfp 的 `threshold_tanimoto_search_symmetric`、Chalcedon 的分块两遍，是同一精确算法的另两种工程布局。本库 **不 `import chemfp`**；OOM 时明确报错，**禁止 subsample 来「修好」split**。预设簇文件按教授要求放 **Hugging Face**，不进 git。

#### 步骤（与代码一致）

1. `GetMorganGenerator(radius=2, fpSize=2048)`。
2. 对 $i=1\ldots n-1$：`BulkTanimotoSimilarity(fps[i], fps[:i])`，只把 $T \ge t$ 的 $j$ 写入双方邻居表（另加自身，以对齐 RDKit 零自距）。
3. 按 `(邻居数, index)` **降序**（`list.sort(reverse=True)`）。
4. Exclusion sphere：未标记且度数 $>1$ 的点当中心，收走未标记邻居；剩余单点各自成簇。
5. 每个 cluster → `_group_split`（与 scaffold 同一套 DeepChem 式贪心）。

Tanimoto：

$$
T(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{c}{a+b-c},\quad d=1-T
$$

$0.65$ 是 Walters / 常见实践默认，**不是**唯一验证过的最优阈值。对齐 DeepChem 应设 `similarity_cutoff=0.4`。Chalcedon 博客的 `cutoff=0.65` 是**距离**（$\Rightarrow T\ge 0.35$），不要和本 API 混用。

#### QM9 默认 $t=0.65$ 实测（全集）

- 133885 分子 → **100653** 簇；单点簇约 **77.6%**；簇大小 min/median/mean/max = 1 / 1 / 1.33 / **23**。
- 最大簇成员对中心 $T$：min=0.65，mean≈0.71，max=1.0（exclusion sphere 成立）。
- 阈值偏严：多数分子没有 $T\ge 0.65$ 的邻居，group split 会接近「按分子切」，但多成员簇仍整组进同一边。

#### 不要做

- 全距离 / `ClusterData` 当生产路径（仅测试、$n\le 2000$）。
- LSH、先 subsample 再叫 `method="butina"`（教授：不要为 split 缩小 QM9）。
- BitBIRCH（另一种算法）。
- chemfp false-singleton 后处理（原文没有）。
- 把预设 `.json.gz` 提交进 `torch_molecule/datasets/data/`。

**测什么泛化：** 指纹空间上不相似的分子（通常比 scaffold 更严；在 QM9+$0.65$ 下因大量 singleton 会略接近 random）。

**参考：** Butina, JCICS 1999, doi:10.1021/ci9803381；chemfp `butina`；Chalcedon (Rowan, 2026)；DeepChem `ButinaSplitter`（只对标整簇分配）；Walters 2024；`molecule_notes/butina_optimization.md`。

---

### 5.4 Size / atom-count split

**只做划分，不包含 SizeShiftReg 的训练正则。**

```python
n_atoms = [mol.GetNumHeavyAtoms() for mol in mols]
order = np.argsort(n_atoms)
n_test = int(round(n * test_size))
idx_test = order[-n_test:]   # 最大分子进 test
idx_train = order[:-n_test]
```

- 按 **重原子数**，不是分子量（DeepChem `MolecularWeightSplitter` 不同）。
- 默认 `direction="small_to_large"`：小 train、大 test（对齐 SizeShiftReg 评估思路）。
- 可选 `mode="sizeshiftreg"`：50% 最小 train / 10% 最大 test（论文协议）。

**测什么泛化：** 尺度外推（小分子 → 大分子），与骨架无关。

**参考：** Buffelli et al., SizeShiftReg, NeurIPS 2022。

**不要：** 在 `split.py` 中实现 graph coarsening 或 CMD loss（属于 `SSRMolecularPredictor`）。

---

## 6. `SMILESDataset` 扩展

当前定义（`torch_molecule/datasets/constant.py`）：

```python
@dataclass
class SMILESDataset:
    data: List[str]
    target: np.ndarray | None
```

建议新增方法（逻辑委托 `split.py`）：

```python
def subsample(self, n: int, seed: int = 0) -> "SMILESDataset":
    """随机抽取 n 条。仅调试 / CI，不要为了 Butina 缩小基准集。"""

def train_test_split(self, test_size=0.2, method="random", seed=42, **kwargs):
    """返回 (train_dataset, test_dataset)。"""

def train_val_test_split(self, train_size=0.8, val_size=0.1, test_size=0.1, ...):
    """三路划分。"""
```

`subsample`：`RandomState(seed).choice(n, size=min(n, n_sub), replace=False)`，保持 `data`/`target` 对齐。

---

## 7. 与现有训练流程对接

```python
from torch_molecule.datasets import load_qm9
from torch_molecule import GREAMolecularPredictor

data = load_qm9(local_dir="torchmol_data")
train, val = data.train_test_split(test_size=0.2, method="scaffold", seed=42)

predictor = GREAMolecularPredictor(num_task=1, task_type="regression")
predictor.fit(train.data, train.target, val.data, val.target)
predictions = predictor.predict(val.data)
```

**不要在** `fit()` **内加** `split=` **参数** — 用户需在同一划分上比较 GREA vs GNN。

---

## 8. 测试计划（`tests/datasets/test_split.py`）


| 测试项              | 断言                                                                |
| ---------------- | ----------------------------------------------------------------- |
| random 可复现       | 同一 seed → 相同索引                                                    |
| random 比例        | test 约等于 `test_size`                                              |
| scaffold 无泄漏     | train/test scaffold 集合交集为空                                        |
| scaffold 覆盖      | 所有 index 仅用一次                                                     |
| 无环分子             | 不崩溃                                                               |
| 无效 SMILES        | `ValueError`                                                      |
| `target is None` | 可划分                                                               |
| 多任务 `y`          | `y.shape[1]` 保持                                                   |
| subsample        | 长度与 seed 可复现                                                      |
| butina           | 同 cluster 不跨集合；小 $n$ 簇与 RDKit `ClusterData` 一致；OOM 文案禁止 subsample |
| size             | test 平均原子数 > train                                                |


可选慢测：`load_qm9` + subsample(1000) + scaffold，CI 可 skip。

---

## 9. 文档说明（每种 method 的 docstring）


| method     | 测什么         | 注意                                               |
| ---------- | ----------- | ------------------------------------------------ |
| `random`   | 同分布插值       | 分数往往偏高，作 baseline                                |
| `scaffold` | 未见 scaffold | MoleculeNet 推荐用于 HIV/BACE/BBBP                   |
| `butina`   | 结构不相似       | 增强版：稀疏精确 Butina；默认 $T\ge 0.65$；不是 ClusterData 矩阵 |
| `size`     | 小→大原子数      | 与 scaffold 正交；SSR 正则另见模型                         |


Group split 时实际比例可能偏离 `test_size` — 文档中明确说明。

---

## 10. 明确不做的事


| 不做                                | 原因                        |
| --------------------------------- | ------------------------- |
| 在 `fit()` 内自动 split               | 无法固定划分比较模型                |
| split 内实现 CMD / coarsening        | 属于 SSR 训练，非划分             |
| 默认 `method=scaffold`              | 破坏旧脚本可复现                  |
| 用 `ClusterData` / 全距离矩阵跑全集 Butina | 精确算法不需要矩阵；QM9 会 OOM       |
| 为了 Butina 把 QM9 subsample         | 教授：QM9 不算大库；缩小数据改变的是划分本身  |
| 把 QM9 簇 `.json.gz` 放进 git 包内      | 预设放 Hugging Face          |
| 把 BitBIRCH / LSH 命名为 `butina`     | 不是原文算法                    |
| 引入 DeepChem 依赖                    | RDKit 指纹 + 自写 chemfp 收球即可 |
| 混淆 stratified（QM7 排序切）与 random    | 若做 stratified，单独 `method` |


---

## 11. 参考文献与资源


| 主题                  | 文献 / 链接                                                                                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scaffold 定义         | Bemis & Murcko, J. Med. Chem. 1996, doi:10.1021/jm9602928                                                                                                              |
| MoleculeNet 协议      | Wu et al., Chem. Sci. 2018, doi:10.1039/C7SC02664A                                                                                                                     |
| Butina 聚类           | Butina, J. Chem. Inf. Comput. Sci. 1999, doi:10.1021/ci9803381                                                                                                         |
| chemfp 稀疏 Butina    | [https://chemfp.com/docs/chemfp_butina_command.html](https://chemfp.com/docs/chemfp_butina_command.html)                                                               |
| 实现对照（Chalcedon 等）   | 本仓库 `molecule_notes/butina_optimization.md`                                                                                                                            |
| Split 实践对比          | Walters, Practical Cheminformatics, 2024                                                                                                                               |
| Size 泛化             | Buffelli et al., SizeShiftReg, NeurIPS 2022, arXiv:2206.07096                                                                                                          |
| DeepChem splitters  | [https://deepchem.readthedocs.io/en/stable/api_reference/splitters.html](https://deepchem.readthedocs.io/en/stable/api_reference/splitters.html)                       |
| scikit-fingerprints | [https://scikit-fingerprints.readthedocs.io/stable/examples/06_dataset_splits.html](https://scikit-fingerprints.readthedocs.io/stable/examples/06_dataset_splits.html) |


---

