# V2 — QDT-LVCG，Tier 1 报告（3 seed + 融合方式消融）

**阶段**：Stage 3（方案顺序）| **协议**：Tier 1，冻结主干 | **日期**：2026-09-18
**结论**：**NOT SUPPORTED**（3 seed 确认；且受原版架构的结构性上限限制）

---

## 1. Executive Summary

把四元数轨迹编码成 beat token 并以"保留预训练"的方式融合进冻结的时序路径，相对本阶段
V0 只有 **+0.43 pp**，远小于 V1 的 +2.72 pp；参数匹配的实值对照有 **+1.61 pp**，再次
高于四元数分支（高 1.18 pp）。三种融合写法（gated / concat / 近零初始化）结果几乎完全
相同（86.09 / 86.10 / 86.09），说明瓶颈不在融合参数化，而在**原版架构只让 beat token 1
进入输出**这一结构性上限。

| 变体 | seeds | 可训练参数 | **test macro AUROC** | vs V0（配对） | best epoch |
|---|---:|---:|---:|---:|---:|
| V0（融合旁路） | 3 | 311,637 | **85.67 ±0.03** | — | 17.7 |
| V2 QDT（gated） | 3 | 311,637 | **86.10 ±0.04** | **+0.43** | 3.0 |
| V2 QDT（concat） | 1 | 278,613 | 86.09 | +0.45 | 4.0 |
| V2 QDT（gated, init 0.01） | 1 | 311,637 | 86.09 | +0.44 | 5.0 |
| 实值对照（参数匹配） | 3 | 312,537 | **87.28 ±0.20** | **+1.61** | 5.3 |

## 2. Research Question / 预注册假设

**问题**（方案 2 表 V2 行）：把 quaternion trajectory 送进 temporal token 是否比 V1 的
并行分支更有效？
**假设**：若心拍级旋转动力学与预训练 token 互补，则 ΔQ > 0，且理想情况下应优于
V1 的并行方案，并优于参数匹配的实值对照。
**判据**：ΔQ = AUROC(fusion) − AUROC(V0，同一脚本)，以及 QDT vs control、QDT vs V1。

## 3. Pretrained Initialization

| 项 | 值 |
|---|---|
| checkpoint | `checkpoints/m5fasts1k1/final.pt`（500,000 步 MIMIC SSL） |
| frozen | 主干全部参数 `requires_grad=False`；BeatEncoder、rhythm 嵌入在缓存阶段前向 |
| 训练时重放（冻结） | StateGRU 展开、`norm_struct`、`norm_dynamic` |
| trainable | 心拍四元数编码器 + TokenFusion + 线性头 |

缓存的是**融合点之前**的张量：心拍 patch [N, 20, 3, 128]、RR、beat mask、BeatEncoder
token [N, 20, 256]、rhythm 嵌入 [N, 128]、整条记录 VCG 幅度的 99 分位、心拍数。
实测每条记录平均 12.7 个心拍（最少 1，最多 20，按 20 补齐）。

## 4. V0-equivalence Check（方案 7.1 Level 0）

`scale = 0` 时**完全旁路融合**（而不只是让它初始化为恒等），因此融合内部即使训练出
非零偏置也无法影响预训练 token。单元测试验证 logits 与直接走预训练路径**逐位相等**，
且四元数分支收不到梯度。

另外两个等价性测试：
- `rollout_hidden` 与原版 `StateGRU` 在相同步数下输出一致（1e-6）；
- 冻结 GRU 在 train / eval 两种模式下输出一致（dropout 已置 0）。

## 5. Mathematical Design

心拍 patch 内的相邻心向量（patch 长 P = 128，由 R-R 间期重采样而来）：

```
z'_n = z_n^VCG + sigmoid(W_g [z_n^VCG ; z_n^Q]) · W_q z_n^Q      (gated, W_q = 0 初始化)
z'_n = W [z_n^VCG ; z_n^Q] + b                                   (concat, W = [I, 0] 初始化)
```

与 V1 的两处差别：
- **dt 按心拍归一**：`dt_n = (rr_n − 1) / ((P − 1)·fs)`，因此 ω 是跨心拍可比的物理角速度
  （rad/s）；每步转角 θ 则不是，两者都提供给编码器。
- **可靠性阈值用整条记录的 99 分位**，而不是 patch 自身的——否则一个安静的边界心拍会被
  判为"方向可靠"。补齐的心拍全程 mask。

## 6. Code Changes

| 文件 | 内容 |
|---|---|
| `lvcg/quaternion/features.py` | 新增 `BeatQuaternionFeatures`（心拍内四元数轨迹） |
| `lvcg/quaternion/qdt.py` | `TokenFusion`、`rollout_hidden`、`QDTProbe` |
| `scripts/train_qdt.py` | 缓存融合点之前的张量 + 训练 + 评估 |
| `configs/eval/qdt_v2.yaml` | V2 配置（含 `fusion`、`fusion_init_std`） |
| `tests/test_quaternion_v2.py` | 11 个测试（含结构上限与 V0 等价性） |

## 7. Tensor Shapes 与 Parameter Count

```
beat patches [B,20,3,128] --BeatQuaternionFeatures--> [B,20,8,127] + mask
   └─ DynamicEncoder(每心拍)                        -> z^Q  [B,20,128]   trainable
z^VCG [B,20,256] (frozen BeatEncoder)
   └─ TokenFusion(z^VCG, z^Q)                       -> z'   [B,20,256]   trainable
z'[:,1] ──┬─ norm_struct (frozen)                   -> emb_struct  [B,256]
          └─ StateGRU 展开 (frozen) -> norm_dynamic -> emb_dynamic [B,256]
emb_rhythm [B,128] (frozen, cached)
logits = Linear([emb_struct ; emb_dynamic ; emb_rhythm]) -> [B,5]
```

| 组成 | gated | concat |
|---|---:|---:|
| 四元数分支 | 176,848 | 176,848 |
| 融合 | 131,584 | 98,560 |
| 头（640→5） | 3,205 | 3,205 |
| 可训练合计 | 311,637 | 278,613 |

## 8. Training Configuration

数据与协议同 V1：PTB-XL Super-class，官方 fold 1–8 / 9 / 10，100% 标签，
5 个独立 logits，BCEWithLogitsLoss，AdamW（lr 1e-3，wd 1e-4），batch 256，
最多 50 epoch，patience 5，seeds 42 / 43 / 44，checkpoint 仅由 validation macro AUROC 选择。

**与 V1 的两处实现差异（必须注意）**：
1. **GRU 展开用每条记录自己的心拍数**，原版用 batch 内最大值（会让嵌入依赖 batch 组成）。
2. 每条记录补齐到 20 个心拍（模型自身上限）。

因此 V2 的 V0 必须在本脚本内重算，不能沿用 V1 的 85.27。

## 9. Numerical Sanity Checks

11 个测试全部通过：心拍特征形状与补齐心拍全 mask；两种融合初始为恒等；`rollout_hidden`
与原版 StateGRU 一致、且支持逐记录步数；`scale=0` 旁路融合并逐位等于 V0；
**只有 beat token 1 能影响输出**（改动其余 token 后 logits 逐位不变，改动 token 1 则改变）；
冻结模块不在可训练集合内且不收梯度；冻结 GRU 在两种模式下数值一致；反向可穿过冻结 GRU。

**一个被测试固化的性质**：零初始化融合在第 0 步**不给四元数分支任何梯度**
（∂z'/∂z_Q = 0），拿到梯度的是融合本身；融合权重离开零点后分支才开始学习。
`fusion_init_std` 提供近零初始化以消除这个冷启动。

## 10. Overall Results

### 10.1 三个 seed 的 test macro AUROC（×100）

| 变体 | seeds | 均值 ±σ | 配对 vs V0 |
|---|---:|---:|---:|
| V0 | 3 | **85.67 ±0.03** | — |
| QDT gated | 3 | **86.10 ±0.04** | **+0.43** |
| 实值对照 | 3 | **87.28 ±0.20** | **+1.61** |

### 10.2 融合方式消融（seed 42）

| 融合 | init std | test macro AUROC | 配对 vs V0 | best epoch |
|---|---:|---:|---:|---:|
| gated | 0 | 86.10 | +0.43 | 3 |
| concat | 0 | 86.09 | +0.45 | 4 |
| gated | 0.01 | 86.09 | +0.44 | 5 |

三种写法的差异仅 0.01 pp，小于 seed 噪声（0.04）。**融合的参数化方式与冷启动都不是瓶颈。**

### 10.3 与 V1 的对照（各自阶段的 V0 为基准）

| 阶段 | V0 | 四元数 | ΔQ | 实值对照 | 对照 − 四元数 |
|---|---:|---:|---:|---:|---:|
| V1 QDF（并行分支） | 85.27 | 88.00 | **+2.72** | 89.32 | −1.32 |
| V2 QDT（token 融合） | 85.67 | 86.10 | **+0.43** | 87.28 | −1.18 |

**V2 的 V0 比 V1 的 V0 高 0.40 pp**，这正是"按每条记录自己的心拍数展开 GRU"相对
"按 batch 最大值展开"的收益——与四元数无关，是实现选择带来的。

## 11. Per-label Results（test AUROC，seed 42）

| label | V0 | QDT gated | QDT − V0 | 对照 | 对照 − QDT |
|---|---:|---:|---:|---:|---:|
| NORM | 0.9008 | 0.9063 | +0.0055 | 0.9163 | +0.0100 |
| MI | 0.8474 | 0.8485 | +0.0011 | 0.8821 | +0.0336 |
| STTC | 0.8915 | 0.8971 | +0.0056 | 0.9101 | +0.0130 |
| CD | 0.8570 | 0.8660 | +0.0090 | 0.8720 | +0.0060 |
| HYP | 0.7854 | 0.7896 | +0.0042 | 0.7976 | +0.0080 |

QDT 在五个 label 上的提升都在 0.1–0.9 pp 之间，其中 **MI 几乎没有提升（+0.11 pp）**，
而 V1 在 MI 上提升了 4.8 pp。MI 依赖波形形态的细节，经由单个 256 维 token 再穿过冻结的
GRU 之后，这类信息显然被压掉了。concat 变体的 HYP（0.7818）甚至略低于 V0（0.7854），
处于噪声量级。

## 12. Real-valued Control（方案 6.1）

对照沿用同一编码器、同一融合、同一头，仅输入通道换成 `position / next_position / delta`，
参数多 0.3%。它比四元数分支高 1.18 pp，**与 V1 的 1.32 pp 方向一致、幅度相近**。
这一复现性本身是有价值的证据：两种完全不同的接入位置（并行分支 / token 融合）下，
实值特征都稳定地优于四元数特征，说明这不是某一种接法的偶然结果。

结合 V1 的 magnitude 消融（去掉幅度通道只损失 0.50 pp），最合理的解释仍是：
**四元数把心向量的绝对朝向归一化掉了**，而电轴方向对 PTB-XL 的诊断有直接价值。

## 13. Training Curve / 失败与替代解释

- V0 的最佳 epoch 平均 17.7，而 QDT 只有 3.0、对照 5.3：**加了分支之后模型很快过拟合**，
  早停在前几个 epoch 就触发。这与"可训练参数从 3,205（V1 的头）涨到 31 万"一致。
- **结构性上限（最重要）**：原版 `forward_inference` 中 `emb_struct` 就是 beat token 1，
  GRU 也只从它展开，**其余心拍的 token 永远不进入嵌入**。V2 融合了全部 20 个 token，
  但只有 token 1 能起作用；测试直接验证了这一点。因此 V2 实际测试的是
  "改写单个锚点 token"，而不是"把整条心拍序列的旋转动力学送进时序模块"。
  要真正做到后者必须改动预训练的时序路径，而 Tier 1 不允许。
- **信息瓶颈**：V1 把 128 维 e_Q 直接拼到 logits 前，V2 则要把信息挤进一个 256 维的
  token，再穿过冻结的 GRU 与 LayerNorm。ΔQ 从 +2.72 掉到 +0.43 与此一致。
- **已排除**：融合参数化（gated / concat 差 0.01 pp）、冷启动（近零初始化差 0.01 pp）、
  seed 噪声（σ ≤ 0.20 pp）。

## 14. Stage Conclusion

**NOT SUPPORTED**：token 级融合确实带来了可复现的小幅增益（+0.43 ±0.04 pp），但
（a）远小于 V1 的并行分支（+2.72 pp），（b）仍然输给参数匹配的实值对照（−1.18 pp）。
方案 V2 行的问题"quaternion trajectory 进入 temporal token 是否更有效"，答案是**否**。

需要区分两层结论：**token 融合这一接入方式本身效率更低**（受原版架构只读 token 1 的
限制），**以及四元数表示不优于实值特征**（V1、V2 两次独立验证）。

## 15. Next Experiment Recommendation

1. **不建议进入 Tier 2**，理由同 V1：Tier 1 未显示优势，放开 fine-tuning 只会混淆归因。
2. **不建议在 V2 上继续投入**：结构性上限决定了它的天花板，除非允许修改预训练时序路径
   （那已不属于 V2 的定义）。
3. **V3（MRQ）仍按计划进行**，并建议：
   - 采用 V1 式的并行分支接入（已证明比 token 融合有效得多）；
   - 在消融矩阵中加入"绝对方向 u_t"一路，直接检验第 12 节的解释。
4. **一个独立于四元数的发现值得记录**：按每条记录自己的心拍数展开 GRU，比原版按 batch
   最大值展开高 0.40 pp，且使嵌入不再依赖 batch 组成。这对任何使用该 checkpoint 的
   下游评估都适用。

## 16. Exact Reproduction Command 与产物

```bash
cd /home/featurize/work/Quaternion/lvcg-reconstructed
python -m pytest tests/test_quaternion_v2.py -q
for s in 42 43 44; do for m in v0 qdt control; do
  python scripts/train_qdt.py --config configs/eval/qdt_v2.yaml --model $m --seed $s
done; done
python scripts/train_qdt.py --config configs/eval/qdt_v2.yaml --model qdt --seed 42 --fusion concat --tag _concat
python scripts/train_qdt.py --config configs/eval/qdt_v2.yaml --model qdt --seed 42 --fusion-init-std 0.01 --tag _warm
python scripts/summarize_quaternion.py
```

产物：`probing/results/qdt_v2.csv`、`qdt_v2_curve_{model}{tag}_s{seed}.json`、
`probing/results/feature_cache/`（V2 的缓存与 V1 不同，训练集约 1.5 GB）。
