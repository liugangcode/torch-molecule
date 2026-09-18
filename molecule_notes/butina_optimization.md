# Butina 聚类 / Split：原文 vs 现有实现与优化

本文记录 **Taylor–Butina** 在分子 ML 划分中的算法原意，以及 2024–2026 年各实现如何加速。  
核心结论：**精确 Butina 不需要全距离矩阵**；当前工业/新开源优化都走「阈值邻居 + exclusion sphere」。DeepChem/RDKit `ClusterData` 是正确但未针对大库优化的参考实现。

与本仓库关系：`method="butina"` 尚未实现。若做全集（如 QM9 ~13 万）且要求 **accuracy**，应对齐 Chalcedon / chemfp 的稀疏路径，而不是 DeepChem 的稠密 `dists` 列表。

---

## 1. 原文算法（必须对齐的「准确」定义）

**文献：** Darko Butina, *Unsupervised Data Base Clustering Based on Daylight’s Fingerprint and Tanimoto Similarity*, J. Chem. Inf. Comput. Sci. **1999**, 39, 747–750.  
https://doi.org/10.1021/ci9803381

**输入：** 分子指纹、Tanimoto 阈值 $t$（例如 0.65 表示「够像才算邻居」）。

**Tanimoto（Jaccard）：**

$$
T(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{c}{a+b-c}
$$

范围 $[0,1]$。距离常写 $d=1-T$。

**三步：**

1. 生成指纹（原文 Daylight；现代几乎一律 Morgan/ECFP）。
2. 对每个分子数邻居：$T \ge t$ 的个数；按邻居数 **降序** 排序（潜在簇中心）。
3. **Exclusion sphere：** 取下一个未标记分子当中心，所有 $T \ge t$ 且未标记的邻居进该簇并标记；已标记者不再当中心、不进别簇。

**精确 Butina 真正需要的信息只有布尔邻居关系：**

$$
N(i,j)=\mathbf{1}[T(i,j)\ge t]
$$

不需要任何 $T < t$ 的数值。因此：

| 做法 | 是否精确 Butina |
|------|-----------------|
| 全距离矩阵 + `Butina.ClusterData` | 是（小 $n$） |
| 阈值邻接表 + **同一套** 排序和收球 | **同样是**（可上全集） |
| LSH / 漏邻居 / 先抽子集再指派 | 否 |
| BitBIRCH 等别的聚类 | 否（即使质量「差不多」） |

---

## 2. 对比总表

| | 原文 1999 | RDKit `ClusterData` | DeepChem / Datamol | Chalcedon (Rowan, 2026) | chemfp `butina` | FPSim2 (ChEMBL) | BitBIRCH |
|--|-----------|---------------------|--------------------|-------------------------|-----------------|-----------------|----------|
| **是不是 Butina** | 定义 | 是 | 是 | 是（对照过 RDKit 簇） | 是 | 否（只做精确相似度搜索） | **否** |
| **指纹** | Daylight | 调用方提供 | Morgan r=2, **1024** bit | Morgan r=2, **2048** bit | 调用方 / fps | RDKit 指纹库 | 二进制指纹 |
| **阈值语义** | 相似度 $T\ge t$ | **距离** `distThresh` | `cutoff` 传给 ClusterData = **距离**（默认 0.6） | `cutoff` = **距离**（博客默认 0.65 ⇒ $T\ge 0.35$） | `--threshold` = **相似度** | `threshold` = **相似度** | 另一套层次阈值 |
| **存什么** | 概念上的邻居 | 全部 condensed 距离 $O(n^2)$ | 全部 `dists` 列表 | 分块；峰值 $O(n)$ + 一块 workspace | **稀疏** NxN（只存 $T\ge t$） | 稀疏 CSR | 树 / iSIM 统计 |
| **加速** | 无 | Bulk 仍填满矩阵 | `BulkTanimotoSimilarity` | NumPy/BLAS、float32、上三角分块、两遍扫描 | POPCNT、稀疏矩阵可存 npz | POPCNT + Swamidass bound、多核/GPU | $O(n)$ 近似层次 |
| **大库** | 当时为替代难调的 Jarvis–Patrick | $n\sim 10^5$ **OOM** | 文档写明 $O(n^2)$，中小集 | **10 万 ≈ 21s / 2.4GB** | 工业级库检索 + 聚类 | 适合建邻居图 | 百万级，但 **不是同簇** |
| **可复现平局** | 按邻居数排序 | 实现相关 | 未强调 | 确定性收球 | 默认 `randomize`；应用 `first`/`last` | N/A | N/A |
| **适合本库** | 语义标准 | 小 $n$ **金标准测试** | API 对标对象，不要抄矩阵 | **全集精确路径的最佳开源参照** | 算法参照；依赖/许可需评估 | 可选搜索后端 | 不要叫 `method="butina"` |

---

## 3. 原文 vs 各实现（分节）

### 3.1 原文 Butina (1999)

**问题：** Jarvis–Patrick 要调两个参数，簇要么极大且杂、要么碎；大库要手工调。

**方法：** 单阈值、簇中心与每个成员都满足 $T\ge t$、exclusion sphere。

**和 split 的关系：** 原文不做 train/test。后来 DeepChem 把 **每个簇当 group**，整组进 train 或 test（与 scaffold 相同贪心）。

**局限：** 阈值无唯一正确答案；簇大小极不均匀；指纹种类会改变结果。

---

### 3.2 RDKit `rdkit.ML.Cluster.Butina`

**做法：**

```text
dists = condensed 下三角 (1 - Tanimoto)   # 必须全部 pair
clusters = Butina.ClusterData(dists, n, distThresh, isDistData=True)
# 每个簇第一个元素是 centroid
```

**优化：** 几乎没有。调用方可用 `DataStructs.BulkTanimotoSimilarity` 加快 **填矩阵**，矩阵本身仍是 $O(n^2)$ 内存。

**实测（他人报告）：**

- Macs in Chemistry：15 万分子聚类时内存涨到 **80–267 GB**，进程被杀。
- Chalcedon 基准：$n=10^5$ **RDKit OOM**；$n=5\times 10^4$ 约 173s / 110GB RSS。

**本库用法：** $n \le 2000$ 的单元测试金标准，**禁止** 对 QM9 全量调用。

---

### 3.3 DeepChem `ButinaSplitter` / Datamol

**源码要点**（`deepchem/splits/splitters.py`）：

- Morgan radius 2，**1024** bits（不是 2048）。
- `BulkTanimotoSimilarity(fps[i], fps[:i])`，`dists.extend(1-x)`。
- `Butina.ClusterData(..., cutoff, isDistData=True)`。
- 簇按大小降序，再按 scaffold 同一套 cutoff 贪心填 train/val/test。
- 文档：**$O(n^2)$**，主要为得到 novel chemotypes；默认 cutoff **0.6（距离）**。

Datamol 教程同一模式：`BulkTanimotoSimilarity(..., returnDistance=True)` + `ClusterData`。

**和原文差别：**

- 指纹：Morgan ≠ Daylight。
- 阈值：DeepChem 0.6 **距离** ≈ $T \ge 0.4$，比「相似度 0.65」松得多。
- 无稀疏化。

**本库：** 可对标「split 时整簇分配」；不要对标其矩阵实现。默认阈值不要盲目抄 0.6 距离，除非文档写死语义。

---

### 3.4 Chalcedon（Rowan, 2026）— 当前最贴近「精确 + 全集」的开源包

**链接：**

- 博客：https://www.rowansci.com/blog/chalcedon （Eli Mann, 2026-05-26）
- 代码：https://github.com/rowansci/chalcedon
- PyPI：`chalcedon`

**动机：** 按 Walters 建议做 Butina split，但 GEOM ~30 万样本时现有开源实现要 **数 TB 内存**。

**仍是精确 Butina：** 分块实现 vs RDKit，在 10k–100k、cutoff=0.65 上他们要求 **簇相同**（bitwise 不完全相同，见浮点）。

**算法改写（内存线性的关键）：**

标准形式要持久化 $n\times n$ 距离。Chalcedon 分成三阶段：

1. **分块**算 pair 相似度，**只持久化每个分子的邻居个数**，丢掉具体相似度。
2. 按邻居数降序排序。
3. 再按排序走，对 **仍未分配** 的集合 **分块重算** 该中心的相似度行，收走未分配邻居。

峰值 ≈ 一块 batch workspace + $O(n)$ 计数。

**其它工程优化：**

- 全程 float32（sgemm 约 2× 于 dgemm）；非二进制描述子他们建议 float64。
- Cutoff 比较改写成 $|A\cap B| \ge (1-\mathrm{cutoff})\cdot |A\cup B|$，少中间数组、略减 ULP。
- 预计算每行 $\|A\|^2$（对 0/1 向量即 popcount）。
- 只走上三角；对角块递归切成矩形 GEMM（类似 SciPy ssyrk 思路）。
- 预分配 buffer，避免反复 malloc。

**Split 部分：** 簇出来后用 Graham **LPT（最长处理时间）** 贪心：每次把最大簇分给「离目标比例最远」的那一份（train/val/test）。与 DeepChem「先填满 train cutoff」不完全相同，但同属整簇分配。

**基准（GEOM 子集，cutoff=0.65，Ryzen 9 7950X3D，128GB RAM）：**

Wall time (s)：

| n | Chalcedon chunked f32 | Chalcedon full matrix f32 | RDKit |
|---|----------------------|---------------------------|--------|
| 1,000 | 0.040 | 0.012 | 0.073 |
| 10,000 | 0.357 | 0.281 | 5.16 |
| 50,000 | 5.75 | 5.74 | 172.8 |
| 100,000 | **21.14** | 22.49 | **OOM** |

Peak RSS (GB)：

| n | Chalcedon chunked f32 | Chalcedon full matrix f32 | RDKit |
|---|----------------------|---------------------------|--------|
| 10,000 | 0.33 | 0.35 | 4.5 |
| 50,000 | 1.23 | 3.16 | 110.4 |
| 100,000 | **2.35** | 11.17 | OOM |

小 $n$ 全矩阵更快；大 $n$ 分块内存近线性，全矩阵仍 $O(n^2)$。

**准确性注意（对「i need accuracy」很重要）：**  
他们在 cutoff sweep 上发现 float32 重排比较 vs RDKit **不是逐 bit 相同**，部分 cutoff 簇数差 $\le 2.5\%$。原因是浮点，不是故意漏边。

要对齐 RDKit / 原文布尔 $T\ge t$：应用 **整数 popcount** 算 $c,a,b$ 再比较，不要用 float32 GEMM 当金标准。

**API 示例：**

```python
splits = chalcedon.butina_split(
    smiles,
    fractions={"train": 0.8, "val": 0.1, "test": 0.1},
    cutoff=0.65,          # 距离 cutoff
    dtype="float32",
)
```

---

### 3.5 chemfp（工业精确稀疏 Butina）

**文档：** https://chemfp.com/docs/chemfp_butina_command.html

**做法：**

1. `threshold_tanimoto_search_symmetric` 生成 **稀疏** 相似度矩阵（只含 $T \ge t_{\mathrm{NxN}}$）。
2. 可存 `.npz`；用更高的 `--butina-threshold` 调参时 **不必重算 Tanimoto**。
3. 按行邻居数排序 + exclusion sphere。
4. `--tiebreaker randomize|first|last`（默认可随机，**可复现必须 first/last + seed**）。
5. 可选：false singleton 贴最近中心（**原文没有**，属于后处理）。

**优化本质：** 与「邻接表精确 Butina」相同，搜索层用 POPCNT 等。  
免费/商业版性能差一截；加依赖前要看许可。

---

### 3.6 FPSim2（ChEMBL）— 精确搜索后端，不是聚类

**文档：** https://chembl.github.io/FPSim2/

- CPU POPCNT；高阈值（$\ge 0.7$）更合适。
- **Swamidass & Baldi 2007** bound（doi: [10.1021/ci600358f](https://doi.org/10.1021/ci600358f)）：  
  $T \le \min(a,b)/\max(a,b)$。query 亮位数 $A$、库分子 $B$ 必须落在 $[tA,\ A/t]$，否则 **不可能** 是邻居 → **剪枝不漏真阳性**。
- 多核、可选 GPU；`symmetric_distance_matrix(threshold=...)` → SciPy CSR。

可放在 Butina 流水线的「建邻居图」一步；exclusion 仍要自己写（或交给 chemfp/Chalcedon）。

---

### 3.7 BitBIRCH — 不要当成优化版 Butina

Miranda-Quintana 等, *Efficient clustering of large molecular libraries*, bioRxiv 2024.  
https://www.biorxiv.org/content/10.1101/2024.08.10.607459v1

- 相对 RDKit Taylor–Butina：45 万分子，4TB 仍不够跑矩阵版；BitBIRCH 约 2 分钟。
- 150 万分子号称 >1000×。
- 质量指标在部分阈值区间与 Butina「无显著差」或更好。
- Chalcedon 对比：**同名义阈值下簇数完全不是一回事**（$n=10^5$：Chalcedon 9590 vs BitBIRCH-Lean 51232）。

这是 **另一种算法**（BIRCH + iSIM）。更快，但不能命名为 `method="butina"`。

---

## 4. 两种「精确」建邻居方式（和 Chalcedon 的关系）

```text
A. 全距离矩阵 + ClusterData
   小 n、易与 RDKit/DeepChem 逐簇对齐
   QM9 全量：内存不可行

B. 阈值邻接 / 分块邻居 + 同一套排序收球
   边集完整 ⇒ 与 A 数学等价
   Chalcedon：分块 + 两遍（先计数再收球），线性内存
   chemfp/FPSim2：稀疏阈值搜索（POPCNT ± bound）
```

Chalcedon 的巧妙处：第一遍 **甚至可以不存邻接表**，只存度数；第二遍对未分配子集重算。  
代价是部分 pair 会算两次；换来内存 $O(n)$ 而不是 $O(n\bar{k})$ 邻接表。  
chemfp 选择 **存稀疏边**，调阈值更快。都精确，工程权衡不同。

---

## 5. 对 torch-molecule 的建议

1. **小 $n$（测试）：** RDKit `BulkTanimotoSimilarity` + `ClusterData`，作为簇相等的金标准。  
2. **大 $n$ / 用户可能丢全集：** Chalcedon 式分块 **或** 稀疏邻接表 + 自写 exclusion；**禁止** 分配 $n(n-1)/2$ 距离数组。  
3. **要比准 RDKit：** 整数 popcount Tanimoto，不要默认 float32 GEMM。  
4. **阈值 API：** 对外只用 `similarity_cutoff`（如 0.65），内部再转距离；文档写清 DeepChem 0.6 是距离。  
5. **指纹默认：** Morgan r=2, 2048 bit（Chalcedon / Walters）；若要对齐 DeepChem 再提供 1024。  
6. **平局：** 稳定 `(-度数, index)`，相当于 chemfp `first`，不要默认 randomize。  
7. **不要** 在 `method="butina"` 里偷偷 subsample 或换 BitBIRCH。  
8. 依赖：能零依赖实现稀疏/分块最好；Chalcedon MIT 且专做 split，可作实现参照，不必强绑。

---

## 6. 参考文献与链接

| 主题 | 来源 |
|------|------|
| 原文 | Butina, JCICS 1999, doi:10.1021/ci9803381 |
| 精确剪枝 bound | Swamidass & Baldi, JCIM 2007, doi:10.1021/ci600358f |
| DeepChem splitter | https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py |
| Chalcedon | https://www.rowansci.com/blog/chalcedon ；https://github.com/rowansci/chalcedon |
| chemfp Butina | https://chemfp.com/docs/chemfp_butina_command.html |
| FPSim2 | https://chembl.github.io/FPSim2/ |
| BitBIRCH | https://www.biorxiv.org/content/10.1101/2024.08.10.607459v1 |
| 大库聚类实践 | https://macinchem.org/2023/03/05/options-for-clustering-large-datasets-of-molecules/ |
| Walters split 评论 | http://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html |

---

*记录日期：对照 Chalcedon 2026-05 博客、chemfp 4.x 文档、DeepChem 源码与 FPSim2/BitBIRCH 文献。待 `method="butina"` 实现时以第 5 节为准。*
