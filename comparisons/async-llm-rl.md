# 异步 LLM RL 横向比较

本页按“问题—信号—机制”比较异步 LLM 强化学习方法，而不是仅按论文时间排列。

## 总览

| 方法 | 主要问题 | 使用信号 | 核心机制 |
|---|---|---|---|
| [DPPO](https://arxiv.org/abs/2602.04879) | ratio 不是合适的分布距离 | Binary/Top-K TV 或 KL | divergence mask |
| [A-3PO](https://arxiv.org/abs/2512.06547) | proximal 重算昂贵且 behavior anchor 陈旧 | policy version lag | log-prob 插值 anchor |
| [SAT](https://arxiv.org/abs/2607.18722) | batch 内 staleness 长尾 | detached sampled log-ratio | 动态收紧 PPO 单侧边界 |
| [ESTR](https://arxiv.org/abs/2607.22186) | 固定阈值混淆低熵噪声与高熵探索 | token entropy | entropy-scaled keep region |
| [VCPO](https://arxiv.org/abs/2602.17616) | stale ratio 导致高方差 | effective sample size | 动态学习率、最小方差 baseline |
| [MinPRO](https://arxiv.org/abs/2601.22718) | token ratio 忽略 prefix state shift | prefix 内最小 ratio | minimum-prefix surrogate |
| [PNPO](https://arxiv.org/abs/2608.01418) | prefix ratio 乘积范围过大 | causal-prefix ratios | prefix 几何平均 |
| [CPPO](https://arxiv.org/abs/2606.10968) | 固定 token 阈值忽略位置与累积偏移 | position、prefix divergence | 位置权重、prefix budget |
| [DRPO](https://arxiv.org/abs/2606.09821) | DPPO mask 无恢复梯度 | Binary-TV shift | 平滑二次正则 |
| [BRRL/BPO](https://arxiv.org/abs/2604.18578) | PPO 越界后停止修正 | target ratio | 双侧 target regression |
| GAC | stale-aligned gradients 导致 overshoot | 连续梯度 cosine | gradient projection |
| R3 | MoE rollout/training routing 不一致 | rollout router decision | router replay |

## Trust-region 几何

- PPO、SAT、A-3PO 主要在 sampled-token ratio 空间工作。
- DPPO、CPPO、DRPO 转向概率质量或分布 divergence。
- BRRL/BPO 认为 pointwise ratio box 有意义，但需要主动回归目标边界。

## Trust-region 锚点

- DPPO：实际 rollout behavior policy。
- Decoupled PPO：近期重算的 proximal policy。
- A-3PO：behavior 与当前 actor 之间的近似 proximal anchor。
- 核心矛盾：behavior 对 IS 是唯一正确锚点，但未必是最相关的优化中心。

## 长序列与 prefix

- MinPRO 用 prefix 内最小 token ratio 近似前缀风险。
- PNPO 使用 causal prefix ratio 的几何平均。
- CPPO 不修改 IS ratio，而是将 divergence budget 按 token 位置和累计 prefix drift 分配。

## 稳定性与系统

- VCPO 控制估计器方差，而非直接改变 trust-region 几何。
- SAT 对 batch 内异常 staleness 尾部更保守。
- ESTR 用局部 entropy 区分噪声与探索。
- GAC 约束跨更新的梯度动力学。
- R3 从系统源头减少 MoE routing mismatch。

## 当前研究问题

### 动态 DPPO 阈值

$$
M_t=
\mathbf 1[
\hat A_t(r_t-1)\le0
\;\lor\;
D_t\le\delta_t].
$$

\(\delta_t\) 可由 policy age、token entropy、position、prefix divergence、batch ESS、engine/router mismatch 和 Top-K coverage 调整。

关键区分：

- rollout 越陈旧，作为优化中心的相关性下降，倾向于放宽 \(\delta_t\)；
- stale gradient 的统计可靠性也下降，应降低样本权重或学习率。

一个标量阈值不应同时承担这两个相反角色。

### 双锚点／预算化 anchor

使用 rollout behavior policy 做 IS，使用 A-3PO proximal policy 做局部 trust region，并保留相对 rollout policy 的全局 TV budget。

### 平滑恢复

用 DRPO 或 BPO 风格的连续恢复梯度替代硬 mask，但保持 DPPO 的绝对概率变化几何。

## 最小实验矩阵

- Lag：0、1、2、4、8、16。
- 模型：dense 与 MoE。
- 设置：同构 engine、训练—推理 mismatch、router mismatch。
- 指标：reward/step、reward/wall-clock、ESS、gradient norm、entropy、global/local TV、mask rate、collapse rate。
- 消融：固定阈值、age-only、entropy-only、prefix-only、联合动态阈值。
- 至少 3 个随机种子，同时报告最终 checkpoint 与最佳 checkpoint。
