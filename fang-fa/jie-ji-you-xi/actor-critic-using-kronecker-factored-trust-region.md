# Actor Critic using Kronecker-Factored Trust Region

## 介绍

> [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/pdf/1708.05144.pdf)

在这项工作中，我们建议用最近提出的Kronecker-factored approximation curvature将信赖域优化应用于深度强化学习。我们扩展了自然策略梯度的框架，提出了利用带信任区域的Kronecker-factored approximation curvature\(K-FAC\)来优化actor和critic；因此，我们此算法其称为Actor Critic using Kronecker-Factored Trust Region \(ACKTR\)，据我们所知，这是Actor-Critic方法中第一个可扩展的信任区域自然梯度方法。它也是一种直接从原始像素输入中学习连续控制中的非平凡任务以及离散控制策略的方法。我们在Atari游戏中的离散域以及MuJoCo环境中的连续域中测试了我们的方法。使用所提出的方法，我们能够获得更高的回报，并比之前的最佳的on-policy的actor-critic样本效率平均提高2 - 3倍。

## 算法

### Natural gradient using Kronecker-factored approximation

### Natural gradient in actor-critic

### Step-size Selection and trust-region optimization

