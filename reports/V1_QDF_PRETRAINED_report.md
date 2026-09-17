# V1 — QDF-LVCG，Tier 1 报告（3 seed + magnitude 消融）

**阶段**：Stage 1 | **协议**：Tier 1，冻结主干 | **日期**：2026-09-18
**结论**：**NOT SUPPORTED**（3 seed 确认，Level 4 完成）

---

## 1. Executive Summary

四元数动态分支相对锁定的 V0 有 **+2.72 pp** 增益（3 seed 均值），但**参数匹配的实值对照
有 +4.05 pp**，比四元数分支高 **1.33 pp**。seed 噪声仅 0.04–0.10 pp，逐 seed 配对差值
（−1.31 / −1.31 / −1.37）高度一致，因此这一差距不是噪声。增益来自"在潜在 VCG 上增加
一条可训练的时序分支"，而非四元数表示本身。按方案 6.1，这正是该对照存在的目的。

| 模型 | 可训练参数 | **test macro AUROC**（3 seed） | vs V0 |
|---|---:|---:|---:|
| V0（scale = 0） | 180,693 | **85.28 ±0.04** | — |
| V1 QDF（四元数） | 180,693 | **88.00 ±0.07** | **+2.72** |
| 实值对照（参数匹配） | 181,593 | **89.33 ±0.10** | **+4.05** |
| QDF 去掉 magnitude（seed 42） | 179,565 | 87.49 | +2.24 |

## 2. Research Question / 预注册假设

**问题**（方案 2 表 V1 行）：显式 rotation dynamics 是否为冻结的 LVCG 表示提供增量诊断信息？
**假设**：若心向量的旋转动力学携带 `e_base` 未捕获的信息，则 ΔQ > 0，**且** 四元数分支
应优于仅使用实值特征的参数匹配对照。
**判据**：ΔQ = AUROC(pretrained + Q branch) − AUROC(pretrained)，以及 QDF vs control 的差值。

## 3. Pretrained Initialization

| 项 | 值 |
|---|---|
| checkpoint | `checkpoints/m5fasts1k1/final.pt`（500,000 步 MIMIC SSL） |
| 加载 | 8,285,060 参数，`missing 0 / unexpected 0` |
| frozen | 全部主干参数 `requires_grad=False` |
| trainable | 仅 `DynamicEncoder` + 线性头 |

主干只前向一次，`e_base` [N, 640] 与潜在 VCG [N, 3, 1000] 被缓存复用，三个模型读同一份缓存，
因此三者的输入逐位相同。

## 4. V0-equivalence Check（方案 7.1 Level 0）

`logits = Linear([e_base ; scale · e_Q])`，`scale = 0` 时四元数分支到不了 logits。
单元测试验证：输出与"仅 e_base 的线性探针"**逐位相等**（`torch.equal`），且反向后
分支权重梯度恒为 0。

本脚本的 V0（0.8525）与此前 `run_probing.py` 的 V0（0.8539，3 seed 均值 0.8527）一致，
说明两条独立实现的探针给出同一参考值。

## 5. Mathematical Design

相邻心向量 P_t, P_{t+1}（相隔一个采样点，dt = 1/100 s）：

| 通道 | 定义 | 数 |
|---|---|---:|
| q_t | shortest-arc rotation u_t → u_{t+1}，`[1+dot, cross]` 归一化 | 4 |
| theta_t | `2·atan2(‖q_xyz‖, |q_w|)` | 1 |
| omega_t | theta_t / dt，rad/s | 1 |
| magnitude | ‖P_t‖ | 1 |
| mask | 该 transition 方向是否可靠 | 1 |

数值稳定：反平行使用确定性 orthogonal-axis fallback；模长低于该记录 99 分位的 2% 时
mask 掉并填入 identity 四元数（先于符号连续化，避免人为跳变）；沿时间强制
`dot(q_t, q_{t-1}) ≥ 0`。池化只在 mask 为真的步上进行。

**实值对照**（方案 6.1）：同一编码器、同一 dropout、同一池化、同一头，仅输入通道换成
`position`、`next_position`、`delta`（各 3 通道）+ mask，共 10 通道 vs 四元数的 8 通道，
参数量相差 0.5%。

## 6. Code Changes

| 文件 | 内容 |
|---|---|
| `lvcg/quaternion/utils.py` | 四元数工具（从 q_wyt 原样移植） |
| `lvcg/quaternion/features.py` | `QuaternionDynamicFeatures`、`transition_features`、特征集校验 |
| `lvcg/quaternion/qdf.py` | `DynamicEncoder`、`QDFProbe`（含 `scale` 开关） |
| `scripts/train_qdf.py` | 冻结前向 + 特征缓存 + 训练 + 评估 |
| `configs/eval/qdf_v1.yaml` | V1 配置 |
| `tests/test_quaternion_v1.py` | 17 个测试（附录 B 全部适用项 + V0 等价性） |

## 7. Tensor Shapes 与 Parameter Count

```
ECG [B,12,5000] --(frozen, 500→100 Hz)--> [B,12,1000]
  ├─ ext_ecg_emb            -> e_base [B,640]      frozen
  └─ vcg_inverse            -> VCG    [B,3,1000]   frozen
        └─ features         -> [B,8,999] + mask [B,999]
             └─ DynamicEncoder(3×stride2) -> e_Q [B,128]
logits = Linear([e_base ; scale·e_Q]) -> [B,5]
```

| 组成 | 参数 |
|---|---:|
| 冻结主干 | 8,285,060 |
| 四元数分支（含头中对应 e_Q 的部分） | 177,488 |
| V0 头（640→5） | 3,205 |
| 可训练合计 | 180,693（对照 181,593，+0.5%） |

## 8. Training Configuration

数据 PTB-XL Super-class，官方 fold 1–8 / 9 / 10 = 17,084 / 2,146 / 2,158，100% 标签，
5 个独立 logits，`BCEWithLogitsLoss`。AdamW，lr 1e-3，weight decay 1e-4，batch 256，
最多 50 epoch，patience 5，seeds 42 / 43 / 44。**checkpoint 仅由 validation macro AUROC
选择**，test 只在最后读取一次。三个模型读同一份冻结特征缓存。

同一配置重跑的差异约 0.001（GPU 卷积的非确定性），小于所有被讨论的差值。

## 9. Numerical Sanity Checks

17 个测试全部通过：identity 不改变向量；旋转保范数；`RᵀR = I`、`det R = 1`（1e-9）；
q 与 −q 给出同一 rotation matrix、同一 geodesic 距离与角度；`vectors_to_quaternion(v1,v2)`
把 v1 精确旋到 v2（1e-10）；平行 → identity；反平行 → 确定性 fallback，角度 π，轴垂直于原向量；
近零向量被 mask 且全程 finite；四元数乘法与矩阵复合一致；forward 输出 finite 且为 [B,5]；
scale=0 时逐位等于 V0；scale=1 时分支收到梯度。

## 10. Overall Results

### 10.1 三个 seed 的 test macro AUROC（×100）

| 模型 | seed 42 | seed 43 | seed 44 | 均值 ±σ |
|---|---:|---:|---:|---:|
| V0 | 85.25 | 85.25 | 85.33 | **85.28 ±0.04** |
| V1 QDF | 87.99 | 87.91 | 88.09 | **88.00 ±0.07** |
| 实值对照 | 89.30 | 89.22 | 89.46 | **89.33 ±0.10** |

### 10.2 逐 seed 配对差值（pp）

| seed | ΔQ = QDF − V0 | 对照 − V0 | QDF − 对照 |
|---|---:|---:|---:|
| 42 | +2.74 | +4.05 | −1.31 |
| 43 | +2.66 | +3.97 | −1.31 |
| 44 | +2.76 | +4.13 | −1.37 |
| **均值** | **+2.72** | **+4.05** | **−1.33** |

三个 seed 的配对差值相差不超过 0.10 pp，而 seed 内部方差为 0.04–0.10 pp，
因此 QDF 低于对照 1.33 pp 是稳定结论，不是 seed 噪声。

### 10.3 其余指标（seed 42）

| 指标 | V0 | V1 QDF | 实值对照 |
|---|---:|---:|---:|
| test micro AUROC | 0.8779 | 0.9025 | **0.9131** |
| test macro F1 | 0.5673 | 0.6598 | **0.6800** |
| test micro F1 | 0.6528 | 0.7220 | **0.7381** |
| val macro AUROC | 0.8630 | 0.8835 | **0.8985** |
| best epoch | 19 | 8 | 6 |

**ΔQ（Tier-1 主指标）= +2.72 pp**；**QDF − control = −1.33 pp**。

## 11. Per-label Results（test AUROC）

seed 42：

| label | V0 | V1 QDF | QDF − V0 | 对照 | 对照 − QDF |
|---|---:|---:|---:|---:|---:|
| NORM | 0.8982 | 0.9212 | +0.0230 | 0.9296 | +0.0084 |
| MI | 0.8397 | 0.8875 | +0.0478 | 0.9094 | +0.0219 |
| STTC | 0.8869 | 0.9042 | +0.0173 | 0.9243 | +0.0201 |
| CD | 0.8556 | 0.8832 | +0.0276 | 0.9036 | +0.0204 |
| HYP | 0.7819 | 0.8033 | +0.0214 | 0.7984 | **−0.0049** |

四元数分支在所有 label 上都优于 V0，MI 提升最大（+4.8 pp）。对照在除 HYP 外的
所有 label 上又优于四元数分支；HYP 是唯一四元数略胜的 label，但差距仅 0.5 pp，
小于合理的 seed 噪声。

## 11.1 Magnitude 消融（方案 6.2）

| 特征集 | seed 42 test macro AUROC | vs 完整 QDF | vs V0 |
|---|---:|---:|---:|
| q + theta + omega + magnitude | 87.99 | — | +2.74 |
| q + theta + omega（无 magnitude） | 87.49 | −0.50 | **+2.24** |

去掉幅度通道只损失 0.50 pp，纯旋转特征相对 V0 仍有 +2.24 pp。**因此"增益主要来自
magnitude 通道"这一假设不成立**：旋转动力学本身确实携带 `e_base` 未捕获的诊断信息。
这同时说明四元数落后于实值对照的原因不是简单的"少了幅度信息"，需要其他解释。

## 12. Real-valued Control（方案 6.1）

对照使用完全相同的编码器、池化、dropout、头与超参，仅输入特征不同，参数多 0.5%。
它的 val 与 test AUROC 全程高于四元数分支，且第 0 个 epoch 的训练 loss 就更低
（0.369 vs 0.409），说明原始向量对与差分对该任务更易优化。

**解释**：四元数特征把方向变化与幅度解耦，只保留 `magnitude` 一个幅度通道；
而 `position/next_position/delta` 同时携带**绝对空间位置**、形态与幅度。
消融（11.1 节）显示补回单一幅度通道只值 0.50 pp，所以差距的主因**不是幅度缺失，
而是绝对朝向信息的丢失**：q_t 只描述相邻两点之间的相对转动，心向量在胸腔坐标系中
指向哪里（电轴方向）被归一化掉了，而电轴偏移正是 CD、HYP 等诊断的直接依据。
对照的 `position` 通道保留了这一信息。

这提示了一个可检验的改进方向：在四元数分支中补入绝对方向（例如 u_t 本身或
相对固定参考向量的四元数），而不是只补幅度。

## 13. Training Curve / 失败与替代解释

- 三个模型都在 6–20 epoch 内触发早停，无发散。
- V1 与对照在 epoch 8 之后 val AUROC 开始下滑（训练 loss 继续下降），是轻度过拟合，
  已由 val-based selection 处理。
- **替代解释 1（最重要）**：增益可能只来自额外的 18 万可训练参数与一条直接看到原始
  VCG 的路径，与"旋转几何"无关。对照结果支持这一解释。
- **替代解释 2（已排除）**：seed 噪声。3 个 seed 的标准差为 0.04（V0）、0.07（QDF）、
  0.10（对照）pp，而 QDF 与对照的配对差值稳定在 −1.31 至 −1.37 pp，远超噪声。
- **替代解释 3**：编码器结构对四元数特征未必最优（BatchNorm + 3 层 stride-2 卷积是为
  实值信号设计的），但方案要求两者结构一致，这是公平性与最优性之间的必然取舍。

## 14. Stage Conclusion

**NOT SUPPORTED（3 seed 确认）**：四元数动态分支确实为冻结的 LVCG 表示带来了稳定的
增量信息（+2.72 ±0.07 pp，去掉幅度通道后仍有 +2.24 pp），但**未能超过参数匹配的
实值对照（−1.33 pp，三个 seed 一致）**，因此本实验不支持"显式四元数 rotation 表示
优于实值动态特征"这一假设。

需要注意区分两个结论：**旋转动力学有信息**（成立），**四元数是表达它的更好方式**（不成立）。

注意：这是工程研究状态，不是统计显著性声明。

## 15. Next Experiment Recommendation

1. **多 seed 确认已完成**（Level 4），无需重复。
2. **不建议进入 Tier 2**：Tier 1 未显示四元数优于实值对照，放开 fine-tuning 只会让
   归因更困难。
3. **可选的补充消融**（方案 6.2 尚未跑的部分）：`q only` / `theta only` / `omega only`，
   用于分辨增益来自完整旋转还是仅来自转角大小。成本约 3 分钟。
4. **对 V3（MRQ）的启示**：本次结果提示差距来自**绝对朝向**而非幅度。V3 显式分解
   magnitude 与 rotation 两支，其消融矩阵（Magnitude only / Rotation only /
   VCG+Rotation / …）正好能检验这一解释，因此 **V3 仍值得按计划进行**，
   并建议在 V3 中额外加入一路"绝对方向 u_t"作为诊断性对照。

## 16. Exact Reproduction Command 与产物

```bash
cd /home/featurize/work/Quaternion/lvcg-reconstructed
python -m pytest tests/test_quaternion_v1.py -q
for s in 42 43 44; do for m in v0 qdf control; do
  python scripts/train_qdf.py --config configs/eval/qdf_v1.yaml --model $m --seed $s
done; done
python scripts/train_qdf.py --config configs/eval/qdf_v1.yaml --model qdf --seed 42 \
  --features q theta omega --tag _nomag
```

产物：`probing/results/qdf_v1.csv`（每次运行一行，含 per-label AUROC/F1）、
`probing/results/qdf_v1_curve_{model}_s42.json`（逐 epoch 曲线）、
`probing/results/feature_cache/`（冻结特征缓存）。
