# V8 — LQA-LVCG，Tier 1 报告（标签条件注意力）

**阶段**：Stage 9 | **协议**：Tier 1，冻结主干 | **日期**：2026-09-18
**运行**：3 模式 × 3 seed + 1 个 `mq` 对照 = 10 次
**结论**：**NOT SUPPORTED —— 注意力相对均值池化最多 +0.15 pp，标签特异没有任何额外贡献**

---

## 1. Executive Summary

给 5 个 superclass 各配一个可学习 query 去读动态序列，**精度与均值池化没有差别**：

| 模式 | seed 42 | 43 | 44 | 均值 ±σ | vs mean | vs V0 |
|---|---:|---:|---:|---:|---:|---:|
| mean（= V3 的 `mo`） | 89.51 | 89.57 | 89.57 | **89.55 ±0.03** | — | +4.27 |
| shared（1 个共享 query） | 89.61 | 89.72 | 89.87 | **89.73 ±0.11** | +0.18 | +4.45 |
| label（5 个 query） | 89.52 | 89.58 | 90.00 | **89.70 ±0.21** | +0.15 | +4.42 |
| label + `mq` 特征（seed 42） | 88.05 | — | — | 88.05 | −1.50 | +2.77 |

**`label` 与 `shared` 打平（−0.03）**，两者相对 `mean` 的 +0.15/+0.18 又落在 seed 波动
（σ 最大 0.21）之内。因此方案 V8 的核心问题"5 个 superclass 是否关注不同的 rotation
dynamics"——**在精度上答案是否定的**。

注意力分布本身给出了一条可能有意义的信号：**STTC 在两个 seed 上都是最偏向 T 段的标签**
（49.3% / 51.5%），与 ST-T 改变的临床定义一致；但其余标签之间的差异在 seed 间不稳定。

## 2. Research Question 与判定

| 假设 | 判定 | 依据 |
|---|---|---|
| 注意力优于均值池化 | **不支持** | +0.15 pp，小于 seed 波动（σ 0.21） |
| **标签特异**的注意力优于共享注意力 | **不支持** | label 89.70 vs shared 89.73，差 −0.03 |
| 不同 superclass 关注心动周期的不同片段 | **弱证据** | 仅 STTC 偏向 T 在 seed 间稳定；其余不稳定 |
| V3 的表示结论在注意力下仍成立 | **支持** | `mq` 比 `mo` 低 1.50 pp，与 V3 的 1.55 一致 |

## 3. 方法

```
h_t     = DynamicEncoder(features(VCG))            编码器特征图，未池化
a_{k,t} = softmax_t( q_k · LN(h_t) / √E )          每标签一个 query
e_Q^k   = Σ_t a_{k,t} h_t
logit_k = w_k · [ e_base ; scale · e_Q^k ] + b_k   每标签一个小头
```

三种模式让主张可被证伪：`label`（5 个 query）、`shared`（1 个 query，保留注意力但去掉
标签条件）、`mean`（均匀掩码平均，即 V1/V3 的做法）。**`mean` 与 V3 同变体一致到 1.2e-7**
（同样的算术，求和顺序不同），这是本阶段的等价性检查。

**一处必要的实现修正**：编码器特征图的标准差只有 0.035，直接点积会让 softmax 在任何
query 下都接近均匀、梯度也弱。因此分数对 **LayerNorm 后的 h** 计算，而池化仍用原始 h
（这样 `mean` 对 V3 的等价性不受影响）。初始化时权重最大/最小之比 1.33（起点接近均值
池化，是期望的性质），query 学开后可达 17 倍。

参数代价极小：`label` 比 `mean` 只多 **0.36%**（5×128 的 query + 每标签一个小头）。

## 4. 注意力落在哪里

窗口沿用 V4：QRS = R−40ms..R+60ms（占 12.9% 的采样点），T = R 后 R-R 的 0.15–0.55
（占 37.8%）。

**`label` 模式，注意力质量占比：**

| 标签 | seed 42 QRS / T | seed 44 QRS / T |
|---|---|---|
| HYP | 25.1% / 45.0% | 24.9% / 44.1% |
| NORM | 21.9% / 47.9% | 19.6% / 49.9% |
| MI | 24.6% / 44.2% | **29.8%** / 43.4% |
| CD | **30.3%** / 40.5% | 23.5% / 46.0% |
| STTC | 19.5% / **49.3%** | 20.1% / **51.5%** |

`shared` 模式五行完全相同（26.9% / 49.1%），符合预期——共用一个 query，也验证了报告逻辑。

**三点解读**：

1. **按时间占比看偏向 T，按密度看偏向 QRS**。T 拿到约 45% 的注意力但占 37.8% 的时间
   （密度 1.19），QRS 拿到约 25% 却只占 12.9%（密度 **1.94**）。**模型实际上超额关注
   QRS 段**，只是 QRS 太短。这与 V4 的发现自洽：QRS 用 12.9% 的采样点达到整拍 99.1% 的效果。
2. **只有 STTC 的偏向在 seed 间稳定**：两个 seed 都是 T 占比最高、QRS 占比最低。
   ST-T 改变按定义就在复极化段，这是唯一一条与临床先验吻合且可复现的信号。
3. **其余标签的排序在 seed 间翻转**（CD 与 MI 的 QRS 占比互换）。因此
   "不同疾病关注不同片段"在本实验中**没有稳定证据**，与 V4 的结论一致。

## 5. Per-label 精度（seed 42）

| 模式 | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|
| mean | 0.9304 | 0.9041 | 0.9261 | 0.9028 | 0.8119 |
| shared | 0.9310 | 0.9052 | 0.9268 | 0.9040 | 0.8134 |
| label | 0.9299 | 0.9049 | 0.9253 | 0.9037 | 0.8121 |

即使在注意力分布差异最大的 STTC 上，三种模式的 AUROC 也只差 0.15 pp。
**注意力改变了模型"看哪里"，却几乎没有改变它"看得多准"。**

## 6. Numerical Sanity Checks

15 个测试通过：三种模式的权重都是非负且按时间求和为 1；被可靠性掩码排除的时间步
**注意力恒为 0**；完全无可靠步的记录仍输出有限值（不产生 NaN）；`label` 有 5 个 query、
`shared` 有 1 个、`mean` 没有；两个相反的 query 给出明显不同的分布；`mean` 与 V3 一致
到 1e-6；`scale=0` 时与 V0 等价；**改动某个标签的 query 只影响该标签的 logit**
（其余变化 < 1e-6）；每标签头的形状是 5×768 而非 5 个 MLP；注意力相对 `mean` 的参数
开销 < 1%；梯度到达 query 与编码器；未知模式/特征集被拒绝。

一个副产物性质：分数对 query 的**常数平移不变**（因为 `LN(h)` 沿特征维均值为零），
测试里已据此调整扰动方式。

## 7. Stage Conclusion

**NOT SUPPORTED**。标签条件注意力既没有超过共享注意力，也没有明显超过均值池化。
V8 的价值落在可解释性而非精度上，而可解释性结果也只支持一条：**STTC 稳定地偏向 T 段**。

**证据强度**：3 seed，σ ≤ 0.21，差值 ≤ 0.18 pp——差值小于噪声，因此是"无差别"而非
"小幅有效"。

## 8. Next Experiment Recommendation

1. **不建议继续投入 V8**：三种模式在精度上无法区分。
2. **若要追这条线**，唯一值得做的是把 STTC 的注意力信号做实：用更细的相位窗口
   （P / QRS / ST / T）和更小感受野的编码器（V4 报告指出当前感受野约 43 个采样点，
   相位之间本就泄漏），看 STTC 的偏向是否仍然稳定并能转化为精度。
3. **整个 Quaternion track 的建议见 `QUATERNION_TRACK_SUMMARY.md`。**

## 9. Exact Reproduction Command 与产物

```bash
cd /home/featurize/work/Quaternion/lvcg-reconstructed
python -m pytest tests/test_quaternion_v8.py -q
for s in 42 43 44; do for m in mean shared label; do
  python scripts/train_lqa.py --config configs/eval/lqa_v8.yaml --mode $m --seed $s
done; done
python scripts/train_lqa.py --config configs/eval/lqa_v8.yaml --mode label --variant mq --tag _mq --seed 42
python scripts/summarize_quaternion.py --results probing/results/lqa_v8.csv
```

**产物**：`probing/results/lqa_v8.csv`（每行含每个标签在 QRS / T 内的注意力占比）、
`lqa_v8_curve_*.json`、`reports/v8_logs.txt`。

**实现**：`lvcg/quaternion/attention.py`（`LabelAttentionProbe`）、`scripts/train_lqa.py`、
`configs/eval/lqa_v8.yaml`、`tests/test_quaternion_v8.py`（15 个测试）。
