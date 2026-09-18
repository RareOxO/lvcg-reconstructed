# 主路线 C — 观测坐标系处理（Tier 1 报告）

**日期**：2026-09-18 | **协议**：Tier 1，冻结主干，PTB-XL Super-class，3 seed（42/43/44）
**运行**：4 个变体 × 3 seed = 12 次
**结论**：**支持（A/B/C 三条路线中唯一一条同时满足性能与 signature 证据的）——
内在坐标系下的表示相对 V0 有 +0.36 pp 且旋转扫描完全平坦，绝对朝向保留后再 +0.03**

---

## 1. 方法：与 V5/V6 的本质区别

方案 C1 明确要求"新方法不能只是自由 PoseNet 或 A₀R 的重复实现"。V5 学一个自由位姿并把
输入转正，V6 转动导联矩阵——两者都把绝对朝向**整体丢弃**，各自换来约 +0.5 pp 且不带来
不变性。路线 C 不做这件事：

**坐标系由记录本身算出，不是学出来的**：

```
R_frame = [x, y, z]     x = 该记录幅度加权的主心向量轴
                        z = 它扫出的环路法向量（向量面积）
                        y = z × x（右手系，正交化）
```

**表示按变换行为拆成两半**：

| 部分 | 内容 | 全局旋转 G 下的行为 |
|---|---|---|
| `invariant` | `V_int = R_frameᵀ V`，轨迹在自己坐标系里的幅度与单位方向 | **完全不变**（因 `R_frame → G R_frame`） |
| `equivariant` | `q_frame`、其转角与三个轴 | **可预测变化**：`q_frame → q_G ⊗ q_frame` |

**绝对朝向被保留而非删除**（方案的硬性要求），只是被路由进独立的嵌入，
分类器可按需使用，旋转扫描也能精确显示它影响的是哪一部分。

## 2. 变换定律在真实数据上的验证

脚本每次运行都在测试集上重新验证两条性质，而不是写在文档里当断言：

| 性质 | 实测误差 |
|---|---:|
| 等变：`R_frame(G·V) = G·R_frame(V)` | **4.0e-6** |
| 不变：`R_frameᵀ V` 在旋转前后一致 | **3.0e-6** |

单元测试另外覆盖：坐标系正交且 det = 1；`q_frame` 与 `R_frame` 互相一致；
退化环路（直线，法向量为零）走确定性 fallback 且输出有限可复现；
只用 `invariant` 的模型在 90° 旋转下端到端 logits 变化 < 1e-4。

## 3. 结果

### 3.1 精度与旋转扫描（test macro AUROC ×100）

| 变体 | 42 | 43 | 44 | 均值 | vs V0 | 30° | 90° | 150° | 参数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0 | 87.35 | 87.30 | 87.39 | **87.35** | — | 0.00 | 0.00 | 0.00 | 189,069 |
| equivariant | 87.32 | 87.34 | 87.29 | **87.32** | −0.03 | −0.16 | −0.62 | **−0.75** | 13,381 |
| **invariant** | 87.71 | 87.63 | 87.78 | **87.71** | **+0.36** | **0.00** | **0.00** | **0.00** | 178,893 |
| **invariant + equivariant** | 87.82 | 87.67 | 87.79 | **87.76** | **+0.41** | −0.12 | −0.38 | −0.57 | 189,069 |

（扫描列为相对该变体自身 0° 的变化。）

### 3.2 Per-label（test AUROC，seed 42）

| 变体 | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|
| v0 | 0.9143 | 0.8654 | 0.9022 | 0.8794 | 0.8059 |
| equivariant | 0.9142 | 0.8657 | 0.9011 | 0.8781 | **0.8068** |
| invariant | 0.9185 | 0.8757 | 0.9041 | 0.8848 | 0.8026 |
| invariant + equivariant | **0.9195** | **0.8786** | **0.9039** | **0.8871** | 0.8017 |

增益集中在 **MI（+1.3 pp）与 CD（+0.8 pp）**，NORM 与 STTC 小幅上升，
**HYP 一致小幅下降**（−0.3 至 −0.4 pp）。HYP 是电压判据，而 `invariant` 分支保留了
幅度通道却把方向归一到自身坐标系——方向信息对 HYP 本就不关键，这个小幅下降与之自洽。

## 4. 分析

### 4.1 不变性是构造出来的，不是训练出来的

`invariant` 的扫描在全部 7 个角度上都是 **0.00**，包括 150°。这不是"鲁棒性好"，
而是**数学上精确不变**：旋转同时作用于轨迹与坐标系，二者抵消。与 V5 的对照很清楚——
V5 用增强训练出的鲁棒性在 90° 仍掉 2.6 至 5.1 pp，而这里是 0。

**而且不以 clean loss 为代价**：方案 C1 规定"任何 robustness improvement 若以明显
clean loss 为代价，不算完整支持"。`invariant` 的 clean AUROC 比 V0 **高** 0.36 pp。
这一条是本路线成立的关键。

### 4.2 绝对朝向：可以保留，但本身不提供增量

- `equivariant` 单独：**−0.03 pp**，即坐标系在世界中的朝向几乎没有诊断价值。
- 加进 `invariant`：87.71 → 87.76（**+0.05**），代价是引入 −0.57 的旋转敏感性。

所以方案的核心问题有了明确答案：**朝向可以被保留而不损害性能，但它不提供额外信息**。
若部署环境存在朝向变化，应当只用 `invariant`；若不存在，两者皆可。

这与 V5/V6 的结论互相印证：V6 的全局几何校正只转 1.1°、增益 −0.02，
说明 PTB-XL 的记录本身不存在系统性朝向偏差可利用——那么朝向信息稀少也就不意外。

### 4.3 与 A、B 的横向比较

| 路线 | 最佳分支 | vs 各自 V0 | signature 证据 |
|---|---|---:|---|
| A | 多尺度有序合成 | +0.75（但 local 就有 +0.70） | **不成立**（扰动过强，无法区分） |
| B | QRS–T 轴关系 | +0.11 | 可解码性 0.64 → 0.97，但部分是构造的 |
| **C** | **内在坐标系（invariant）** | **+0.36** | **成立**：精确不变，误差 3e-6 |

A 的合成相对 local 只有 +0.05，B 的诊断增量只有 +0.11，**只有 C 同时给出了
性能增量与一条干净、可证明的 signature 证据**。

## 5. Limitation（必须与结果一同阅读）

### 5.1 V0 的旋转扫描为 0 是实现的必然，不能解读为"V0 也不变"

本脚本的扰动作用在心拍轨迹上，而 V0 分支只读**缓存的 beat token**，token 是在扰动之前
由冻结编码器算好的，因此扰动到不了 V0 的通路——它的扫描必然是平的。

**真实情况相反**：V5 在扰动时重算 token，那里冻结的 LVCG 在 90° 旋转下掉 20.4 pp、
150° 退化到接近随机。所以**本表中 "V0 扫描平坦" 是伪影**，不能用来主张
"C 比 V0 更鲁棒"。C 的不变性本身（3.2 节、4.1 节）不受此影响，因为它由数学保证，
与对照无关。

要把"C 比 V0 更鲁棒"做实，需要在扰动评估时重算 token（V5 的 `--recompute` 路径已有
现成实现），约 5 分钟一次。**本报告不作此主张。**

### 5.2 其他限制

- 单一数据集（PTB-XL Super-class），未做 low-label 与外部验证（方案 Level 2/4）。
- 人工全局旋转是 **frame mechanism test**，方案第 9 节明确要求不得等同于真实
  acquisition robustness，本报告遵守此界定。
- 坐标系由主轴与环路法向量定义，对环路近乎平面/直线的记录依赖 fallback；
  实际触发比例未统计。

## 6. C1 判定与下一步

**C1：支持。** 同时满足方案的两项要求——clean AUROC 不降反升（+0.36），
且存在该路线独有的 signature 证据（精确的 frame 不变性，实测误差 3e-6）。

按方案 C2，下一步本应进入 **cross-domain / cross-acquisition 真实泛化**。但方案同时
规定：必须先审计外部数据集的 12 导联可用性、采样率、标签映射、许可与 patient split，
**若本地没有合适外部数据或标签映射无法可靠确定，应输出 "External Dataset Requirement"
清单并暂停该分支，不得自行下载未知许可数据或猜测标签映射**。

**当前状态**：本地只有 PTB-XL（已用）与一个未解压的 CSN/Chapman 压缩包；
ICBEB 未下载；标签映射未定义。因此 C2 应暂停，等待外部数据集确认。

**在此之前可以做、且不依赖外部数据的**（按优先级）：

1. **补 V0 的重算-token 旋转对照**（5 分钟）：把 5.1 节的 limitation 消掉，
   使"C 比 V0 更鲁棒"成为可主张的结论。
2. **low-label 验证**（方案 Level 2，约 25 分钟）：1% / 10% 下比较 v0 与 invariant，
   判断 +0.36 是 representation quality 还是 full-data 单点增益。
3. **辅助路线 D（sparse-lead）**：方案第 14 节把 D 定位为"A/B/C 任一正向结果之后的
   capability test"。**C 已给出正向结果，因此 D 现在可以启动**，而且它与 C 的主张
   天然契合——导联越少，三维方向越不确定，内在坐标系的价值应当越明显。

## 7. 复现与产物

```bash
cd /home/featurize/work/Quaternion/lvcg-reconstructed
python -m pytest tests/test_route_c_frame.py -q
for s in 42 43 44; do
  python scripts/train_frame.py --config configs/eval/frame_c.yaml --variant v0 --seed $s
  for p in invariant equivariant "invariant,equivariant"; do
    python scripts/train_frame.py --config configs/eval/frame_c.yaml --parts $p --seed $s
  done
done
```

**产物**：`probing/results/frame_c.csv`（12 次运行，每行含两条变换定律的实测误差与
7 个角度的扫描）、`frame_c_curve_*.json`、`reports/C_logs.txt`。

**实现**：`lvcg/quaternion/frame.py`（`intrinsic_frame`、`to_frame`、`FrameSplitEncoder`、
`FrameProbe`）、`lvcg/quaternion/utils.py` 新增 `rotation_matrix_to_quaternion`
（Shepperd 方法，四分支择优，半圈旋转不产生 NaN，往返误差 2.5e-12）、
`scripts/train_frame.py`、`configs/eval/frame_c.yaml`、`tests/test_route_c_frame.py`
（23 个测试）。
