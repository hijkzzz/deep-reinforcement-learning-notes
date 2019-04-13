# Retrace\(λ\)

## 介绍

> [Safe and Efficient Off-Policy Reinforcement Learning](https://arxiv.org/abs/1606.02647)

在这项工作中，我们重新审视了一些新的和新的算法，用于off-policy，基于回报的强化学习。 以一种常见的形式表达这些，我们揭示了一种新颖的算法Retrace$$(\lambda)$$ 。具有三个所需的属性：（1）它具有低的方差; （2）保证从任何行为策略中收集的样本，无论“off-policy”程度如何; （3）它可以充分利用从近on-policy策略行为中收集的样本。分析了相关算子在off-policy评估和控制设置下的收敛性，推导了基于在线样本的算法。我们认为这是第一个基于回报的off-policy控制算法收敛到 $$Q^{*}$$ ，而无需GLIE假设（\(Greedy in the Limit with Infinite Exploration）。作为推论，我们证明了Watkins’ $$Q(\lambda)$$ 的收敛性。我们举例说明了 Retrace$$(\lambda)$$在标准的Atari 2600游戏上的优势。

## 方法



