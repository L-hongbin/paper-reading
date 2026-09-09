# Paper Reading Lab

面向大模型强化学习与机器学习论文的精读、归类、横向比较和研究想法沉淀。

## 阅读流程

1. 将论文加入分类索引。
2. 使用 [精读模板](templates/paper-note.md)记录问题、方法、理论、实验与局限。
3. 将同一研究问题下的方法放入 `comparisons/` 横向比较。
4. 明确区分论文结论、实验证据、个人推断和开放问题。
5. 研究想法同时记录最小实验和关键消融。

## 分类导航

| 方向 | 核心问题 | 入口 |
|---|---|---|
| 异步与 off-policy LLM RL | policy lag、旧轨迹复用、训练—推理偏差 | [异步 LLM RL 横向比较](comparisons/async-llm-rl.md) |
| Trust-region 几何 | ratio、TV/KL、概率质量约束 | [DPPO 精读](notes/dppo.md) |
| Proximal anchor | behavior policy 与当前 actor 之间如何选择锚点 | [A-3PO 精读](notes/a-3po.md) |
| 长序列校正 | prefix ratio、位置效应、累积 divergence | [长序列与 prefix](comparisons/async-llm-rl.md#长序列与-prefix) |
| 优化稳定性 | ESS、梯度方差、恢复梯度、MoE 路由偏差 | [稳定性与系统](comparisons/async-llm-rl.md#稳定性与系统) |

## 论文索引

| 论文 | 分类 | 核心标签 | 状态 |
|---|---|---|---|
| [DPPO](https://arxiv.org/abs/2602.04879) | Trust region | Binary/Top-K TV、rollout anchor | [已精读](notes/dppo.md) |
| [A-3PO](https://arxiv.org/abs/2512.06547) | 异步 RL | proximal interpolation、decoupled loss | [已精读](notes/a-3po.md) |
| [MinPRO](https://arxiv.org/abs/2601.22718) | 长序列校正 | minimum prefix ratio | 已比较 |
| [VCPO](https://arxiv.org/abs/2602.17616) | 异步稳定性 | ESS、方差控制 | 已比较 |
| [SAT](https://arxiv.org/abs/2607.18722) | 异步 trust region | staleness tail、动态裁剪 | 已比较 |
| [ESTR](https://arxiv.org/abs/2607.22186) | 异步 trust region | entropy-scaled threshold | 已比较 |
| [PNPO](https://arxiv.org/abs/2608.01418) | 长序列校正 | prefix-normalized ratio | 已比较 |
| [CPPO](https://arxiv.org/abs/2606.10968) | 长序列 trust region | position weight、prefix budget | 已比较 |
| [DRPO](https://arxiv.org/abs/2606.09821) | Trust region | smooth Binary-TV regularizer | 已比较 |
| [BRRL/BPO](https://arxiv.org/abs/2604.18578) | 优化目标 | bounded ratio、恢复梯度 | 已比较 |

## 目录

```text
paper-reading/
├── README.md
├── notes/
│   ├── dppo.md
│   └── a-3po.md
├── comparisons/
│   └── async-llm-rl.md
└── templates/
    └── paper-note.md
```

## 状态标记

- **待读**：已收录，尚未系统阅读。
- **阅读中**：正在核对方法、实验和相关工作。
- **已精读**：已有完整独立笔记。
- **已比较**：已进入专题横向比较，独立笔记待补。
