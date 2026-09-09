# DPPO：Rethinking the Trust Region in LLM Reinforcement Learning

- 论文：https://arxiv.org/abs/2602.04879
- 分类：LLM RL / Trust Region
- 状态：已精读

## 一句话结论

DPPO 将 PPO 对 sampled-token 相对概率比的裁剪，替换为对策略概率质量变化的 TV/KL 约束，并强调 trust region 必须锚定实际生成数据的 rollout policy。

## 1. 问题

PPO 使用

$$
r_t=\frac{\pi_\theta(a_t\mid s_t)}{\mu(a_t\mid s_t)}
$$

判断更新是否越界。在长尾词表中，低概率 token 很小的绝对变化也可能产生巨大 ratio；高概率 token 的显著概率质量移动则可能只有温和 ratio。

## 2. 目标函数

$$
L^{\mathrm{DPPO}}_\mu(\pi)
=
\mathbb E_{\tau\sim\mu}
\left[
\sum_t M_t r_t\hat A_t
\right].
$$

当更新继续远离 rollout policy 且 divergence 超过阈值时：

$$
M_t=0
\iff
\hat A_t(r_t-1)>0
\ \land\
D_t>\delta.
$$

ratio 负责 importance sampling 和方向判断；divergence 决定 trust-region gate。

## 3. Binary 与 Top-K

Binary-TV：

$$
D_t^{\mathrm{Bin-TV}}
=
|\pi_\theta(a_t\mid s_t)-\mu(a_t\mid s_t)|.
$$

它只需要 rollout sampled-token log-prob 和当前 actor log-prob，不要求保存全词表 logits。

Top-K 将 rollout Top-K token、采样 token 和剩余概率 “other” 构造成低维分布。两种近似都是完整 divergence 的下界，可能漏掉 other 内部的概率重分配。

## 4. 为什么使用 rollout anchor

1. 轨迹来自实际 rollout policy，IS ratio 的分母必须是实际行为概率。
2. 有限时域 performance bound 围绕行为策略展开。
3. 训练端重算的旧策略可能因精度、kernel 或 MoE routing 与 rollout engine 不一致。
4. 使用重算策略会隐藏真实的 training–inference mismatch。

## 5. 理论—实践差距

- 理论约束完整策略 divergence，Binary/Top-K 只是可计算下界。
- mask 只能阻止继续向外更新，不能主动拉回越界 token。
- token-level IS 未解决完整 prefix distribution mismatch。

## 6. 与 A-3PO 的关系

DPPO 坚持以实际 rollout policy 为 trust anchor；A-3PO 认为高度陈旧的 behavior policy 不适合作为当前优化中心，因此构造靠近当前 actor 的插值 anchor。

自然方向是保留 rollout policy 做 IS 校正，使用局部 proximal anchor，并设置相对 rollout policy 的全局 TV 预算。
