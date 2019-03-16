# Rainbow

## 介绍

> [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)

深度强化学习社区已经对DQN算法进行了几项独立的改进。 但是，目前还不清楚哪些扩展是互补的，可以有效地结合起来。 本文考察了DQN算法的扩展和经验研究的组合。 我们的实验表明，在数据效率和最终性能方面，这种组合在Atari 2600基准测试中提供了最先进的性能。 我们还提供了详细消融研究的结果，该研究显示了每个组成部分对整体表现的贡献。

## 算法

Rainbow即集成各种DQN算法的变体

### The Integrated Agent

首先，我们用multi-step分布DQN替换1-step分布DQN

然后融入Double Q-Learning的思想，选择动作和评估用不同的网络

然后用KL散度作为优先采样的权重

接下来用Dueling作为网络架构

最后加入噪声就得到了整合的代理

## 实验效果

![](../../.gitbook/assets/image%20%2826%29.png)

![](../../.gitbook/assets/image%20%2831%29.png)

可以看出priority和multi-step的影响最大







