# A-3PO：Staleness-aware Proximal Policy Approximation

- 论文：https://arxiv.org/abs/2512.06547
- 分类：异步 LLM RL / Proximal Anchor
- 状态：已精读

## 一句话结论

A-3PO 在 sampled-token log-probability 空间插值 behavior policy 与当前 actor，近似 decoupled PPO 所需的 proximal policy，从而免去额外模型 forward。

## 1. 解耦两个角色

在异步 RL 中，behavior policy 可能明显落后于 learner。Decoupled PPO 将两个角色分开：

- behavior policy：正确的 off-policy importance correction；
- proximal policy：当前更新的 trust-region anchor。

目标为：

$$
L=
\mathbb E
\left[
\frac{\pi_{\rm prox}}{\pi_{\rm behav}}
\min\left(
\frac{\pi_\theta}{\pi_{\rm prox}}\hat A,
\operatorname{clip}\left(
\frac{\pi_\theta}{\pi_{\rm prox}},1-\epsilon,1+\epsilon
\right)\hat A
\right)
\right].
$$

## 2. A-3PO 插值

$$
\log\pi_{\rm prox}
=
\alpha\log\pi_{\rm behav}
+
(1-\alpha)\log\pi_\theta,
$$

$$
\alpha=
\begin{cases}
0,&d=0,\\
1/d,&d\ge1.
\end{cases}
$$

这里使用的是采样 token 的 log-prob，而不是必须保存完整词表 logits。高 staleness 下，proximal anchor 更靠近当前 actor。

由此：

$$
\frac{\pi_\theta}{\pi_{\rm prox}}
=
\left(
\frac{\pi_\theta}{\pi_{\rm behav}}
\right)^\alpha.
$$

## 3. 优势

- 不需要 proximal model 的额外 forward；
- 保留 decoupled objective 的形式；
- 收缩用于 clipping 的 ratio；
- 对大模型异步训练有直接的系统收益。

## 4. 关键局限

1. \(\alpha\) 只依赖 version lag，没有观察实际概率质量偏移。
2. 当 \(d\to\infty\) 时，\(\pi_{\rm prox}\to\pi_\theta\)，局部 trust region 可能退化。
3. sampled-token 插值不等同于构造归一化的完整词表 policy。
4. 高 staleness 同时意味着 behavior anchor 更过时和 IS 方差更大；A-3PO 主要处理前者。

## 5. 与 DPPO 的组合

保持：

$$
\frac{\pi_\theta}{\pi_{\rm behav}}
=
\frac{\pi_{\rm prox}}{\pi_{\rm behav}}
\frac{\pi_\theta}{\pi_{\rm prox}}.
$$

使用前一项做行为分布校正，对后一项使用 Binary-TV trust region。同时保留相对 behavior policy 的全局 divergence budget，避免高 staleness 时约束消失。
