# DDPG + Mixing policy targets

## 介绍

> [On-policy vs. off-policy updates for deep reinforcement learning](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/DeepRL16-hausknecht.pdf)

基于时间差异的深度加强学习方法通​​常由off-policy Q-Learning引导更新。在本文中，我们将研究使用on-policy，Monte Carlo更新的效果。 我们的实证结果表明，对于连续作用空间中的DDPG算法，与仅使用一个或另一个目标相比，混合策略上和非策略更新目标表现出优越的性能和稳定性。在离散动作空间中应用于DQN的相同技术大大减慢了学习。 我们的发现提出了关于on-policy和off-policy和蒙特卡罗更新的性质及其与深度强化学习方法的关系的问题。

## 方法

### 时间差分和蒙特卡洛的关系

![](../../.gitbook/assets/image%20%2834%29.png)

![](../../.gitbook/assets/image%20%2898%29.png)

![](../../.gitbook/assets/image%20%2880%29.png)

### Computing On-Policy MC Targets

![](../../.gitbook/assets/image%20%2865%29.png)

### Mixing Update Targets

![](../../.gitbook/assets/image%20%2833%29.png)

## 实验

### Results in discrete action space

DQN架构\[8\]使用深度神经网络和1步Q-Learning更新来估计每个离散行为的Q值。 使用Arcade学习环境\[2\]，我们评估混合更新对Beam Rider，Breakout，Pong，QBert和Space Invaders的Atari游戏的影响。

![](../../.gitbook/assets/image%20%28125%29.png)

### Results: DDPG

Half Field Offense Domain

回报函数

![](../../.gitbook/assets/image%20%28166%29.png)

![](../../.gitbook/assets/image%20%28134%29.png)



