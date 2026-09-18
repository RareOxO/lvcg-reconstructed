# 主路线 B — QRS–T 三维旋转关系（Tier 1 报告）

**日期**：2026-09-18 | **协议**：Tier 1，冻结主干，PTB-XL Super-class，3 seed（42/43/44）
**运行**：4 个变体 × 3 seed = 12 次
**结论**：**部分支持（方案 B3 的第二档）—— 几何目标的可解码性大幅增强（R² 0.64 → 0.97），
但诊断增量很小（+0.13 pp）；按方案应转 low-label / transfer 验证，而非放弃**

---

## 1. B1 前置条件：分界情况（必须先声明）

**本仓库没有任何 QRS onset/offset 或 T 波分界**，分段只提供 R 峰。因此本路线使用
**R 峰相对的心拍分数窗口**，这是方案 B1 允许的 fallback，且被方案明确限定为
**Tier 1 exploratory**，不是生理精确分段。两条具体后果：

- 心拍 patch 从**它自己的 R 峰**开始，所以"QRS 窗口"覆盖 R 到 J 点；**R 之前的上升支
  属于上一个 patch，未被包含**。
- 窗口是 R-R 间期的分数（QRS 0–12%，T 15–55%），随心率伸缩，但不反映真实 QT 动态。

**任何依赖真实 onset/offset 的结论都不能由本报告支持。** 若要正式化，需要引入可靠的
分界器（工作区内 `q_wyt/qdg/delineate.py` 是候选，但它工作在原始 100 Hz VCG 上，
移植到重采样 patch 需要改写）。

## 2. 方法

每个心拍分别取 QRS 段与 T 段的紧凑 3D 几何表示，再用四元数描述两者的关系，
复杂度逐级递增（方案 B2 要求"从简单整体 relation 到 trajectory-to-trajectory"）：

| 层级 | 内容 | 可训练参数 |
|---|---|---:|
| `axis` | QRS 轴 → T 轴的四元数与转角（**即空间 QRS–T 角**）+ 两个轴 | 13,189 |
| `plane` | 再加两个环路法向量之间的旋转 | 13,893 |
| `trajectory` | 再加两段重采样后逐步的对应旋转 | 189,841 |

- **轴**：该窗口内幅度加权的主方向（临床 VCG 的 QRS 轴 / T 轴），符号由该段平均向量确定。
- **法向量**：`Σ P_t × P_{t+1}`，即环路的向量面积；直线环路为零。
- 结构表示按研究计划锁定为**所有有效 beat token 的 mean pooling**。

## 3. 结果

### 3.1 诊断精度（test macro AUROC ×100）

| 变体 | seed 42 | 43 | 44 | 均值 | vs V0 |
|---|---:|---:|---:|---:|---:|
| v0 | 87.36 | 87.31 | 87.37 | **87.35** | — |
| **axis** | 87.46 | 87.42 | 87.50 | **87.46** | **+0.11** |
| plane | 87.42 | 87.42 | 87.42 | **87.42** | +0.07 |
| trajectory | 87.12 | 87.19 | 87.05 | **87.12** | **−0.23** |

### 3.2 机制目标：空间 QRS–T 角的线性可解码性（test R²）

测试集上该角度的分布：**均值 90.6°，标准差 24.5°**。

| 变体 | 从 `e_base` 解码 | 加上分支后 | 增量 |
|---|---:|---:|---:|
| v0 | 0.641 | 0.641 | +0.000 |
| **axis** | 0.641 | **0.970** | **+0.329** |
| plane | 0.641 | 0.963 | +0.322 |
| trajectory | 0.641 | 0.877 | +0.236 |

三个 seed 的 `axis` 结果为 0.969 / 0.969 / 0.970，完全一致。

### 3.3 Per-label（test AUROC，seed 42；方案要求全部报告，不得只挑最优类别）

| 变体 | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|
| v0 | 0.9147 | 0.8652 | 0.9024 | **0.8793** | **0.8064** |
| axis | 0.9161 | 0.8655 | 0.9065 | 0.8771 | 0.8065 |
| plane | 0.9163 | 0.8632 | **0.9090** | 0.8762 | 0.8046 |
| trajectory | 0.9162 | 0.8580 | 0.9072 | 0.8774 | 0.8044 |

变化幅度全部在 ±0.5 pp 以内。**STTC 是唯一一致上升的类别**（+0.4 至 +0.7 pp），
**CD 与 HYP 一致小幅下降**。STTC（ST-T 改变）依赖复极化形态与其空间朝向，
方向上与"编码 QRS–T 关系"自洽；但幅度太小，不足以作为机制主张，只能记录。

## 4. 分析

### 4.1 可解码性的增益有多少是"构造出来的"

+0.33 的 R² 增益必须谨慎解读：**分支的输入里直接含有 QRS 轴、T 轴及其四元数关系，
而解码目标正是这两个轴的夹角**。因此 R² 0.97 证明的是"分支确实把这个几何量编码进了
最终嵌入并且线性可读"，**不能证明"模型发现了原版没有的新几何信息"**。

真正有信息量的是另一个数字：**`e_base` 单独就能解出 R² 0.641**。也就是说，
**冻结的 LVCG 嵌入已经隐含编码了约六成的 QRS–T 角信息**——原版对这个几何量并非无知，
这解释了为什么把它显式化之后诊断增量如此之小（+0.11 pp）。

### 4.2 层级越复杂越差

`axis`（13k 参数）+0.11 > `plane`（14k）+0.07 > `trajectory`（190k）−0.23。
与 V2（token 融合）、V4（相位分辨）、路线 A（多尺度合成）的规律一致：
**在这个冻结主干上，增加几何描述的复杂度不会带来收益，反而因参数增多更快过拟合。**
`trajectory` 的可解码性也低于 `axis`（0.877 vs 0.970），说明逐步关系反而稀释了
那个简单的轴间角度。

### 4.3 与路线 A 的对照

| 路线 | 分支 | vs V0 |
|---|---|---:|
| A | 单步四元数（local） | +0.70 |
| A | 多尺度有序合成（loop） | +0.75 |
| **B** | **QRS–T 轴关系（axis）** | **+0.11** |

路线 A 的分支读取整条轨迹的逐步旋转，B 的分支只读两个段的整体朝向关系——
**信息量差一个数量级，增量也差一个数量级**。这与"有用的是重新读取轨迹本身，
而不是某个特定的几何构造"这一跨阶段结论吻合。

## 5. B3 自动决策

方案 B3 的三档：

- **支持**：V0 上有稳定增量 **且** inter-loop geometric target 可解码性增强 → 本次不满足
  （增量仅 +0.11 pp，小于 seed 波动 0.06–0.09 的两倍但也谈不上"稳定增量"）。
- **部分支持**：几何 target 明显增强但 diagnosis 增量弱 → **本次命中**。
  方案规定："转向 low-label / transfer，判断它是否是 representation quality 而非
  full-data ceiling 的优势。"
- **不支持**：可靠 delineation 下 inter-loop relation 对 V0 没有补充 → 不适用，
  因为本次并非"可靠 delineation"。

**判定：部分支持。** 下一步按方案执行 low-label 验证，而不是放弃路线，也不是
增加更多四元数参数（`trajectory` 已证明这条路是负收益）。

## 6. 下一步（方案规定的，按优先级）

1. **low-label 验证**（约 25 分钟）：1% 与 10% 标签下比较 v0 与 axis，各 3 seed。
   若 axis 的优势在低标签下放大，说明它改善的是 representation quality 而非
   full-data ceiling；若同样微弱，路线 B 应结束。
   ```bash
   for r in 0.01 0.1; do for s in 42 43 44; do
     python scripts/train_qrst.py --config configs/eval/qrst_b.yaml --variant v0 --ratio $r --seed $s
     python scripts/train_qrst.py --config configs/eval/qrst_b.yaml --level axis --ratio $r --seed $s
   done; done
   ```
2. **若 low-label 也微弱**：本路线的正式结论应当是
   "冻结的 LVCG 已编码约 60% 的 QRS–T 角信息，显式化它可以把可解码性提到 97%，
   但这在 PTB-XL Super-class 上不转化为诊断能力"——这是一个**阴性但可解释、
   有机制证据**的结论，值得写入。
3. **若要把结论正式化**，必须先引入可靠分界器，否则第 1 节的 caveat 会限制所有表述。

## 7. 复现与产物

```bash
cd /home/featurize/work/Quaternion/lvcg-reconstructed
python -m pytest tests/test_route_b_interloop.py -q
for s in 42 43 44; do
  python scripts/train_qrst.py --config configs/eval/qrst_b.yaml --variant v0 --seed $s
  for l in axis plane trajectory; do
    python scripts/train_qrst.py --config configs/eval/qrst_b.yaml --level $l --seed $s
  done
done
```

**产物**：`probing/results/qrst_b.csv`（12 次运行，每行含机制解码 R² 与窗口设置）、
`qrst_b_curve_*.json`、`reports/B_logs.txt`。

**实现**：`lvcg/quaternion/interloop.py`（`segment_axis`、`segment_normal`、
`relation_quaternion`、`spatial_qrst_angle`、`InterLoopEncoder`、`QRSTProbe`）、
`scripts/train_qrst.py`、`configs/eval/qrst_b.yaml`、`tests/test_route_b_interloop.py`
（26 个测试，含植入 0°/30°/90°/150° 时 QRS–T 角的精确恢复、整体旋转不变性、
以及脚本训练循环的端到端验证）。
