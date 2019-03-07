# Proximal Policy Optimization

## 介绍

> [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

我们提出了一个用于强化学习的新的策略梯度方法，它通过与环境的相互作用在采样数据之间交替，并使用随机梯度上升来优化“替代”目标函数。 鉴于标准策略梯度方法针对每个数据样本执行一次梯度更新，我们提出了一种新的目标函数，其允许多个时段的小批量更新。 我们称之为邻域策略优化（PPO）的新方法具有信任区域策略优化（TRPO）的一些优点，但它们实现起来更简单，更通用，并且具有更好的样本复杂性（根据经验）。 我们的实验测试PPO的一系列基准任务，包括模拟机器人运动和Atari游戏，我们表明PPO优于其他在线政策梯度方法，总体上在样本复杂性，简单性和wall-time之间取得了有利的平衡。

## 算法

### TRPO

在TRPO中，目标函数\(“代理”目标\)的最大化，受策略更新步长的限制

$$
\underset{\theta}{\operatorname{maximize}} \quad \hat{\mathbb{E}}_{l}\left[\frac{\pi_{0}\left(a_{l} | s_{l}\right)}{\pi_{\theta_{\text { old }}}\left(a_{l} | s_{l}\right)} \hat{A}_{l}\right]
\\
\hat{\mathbb{E}}_{l}\left[\mathrm{KL}\left[\pi_{\theta_\mathrm{old}}(\cdot | s_{l}), \pi_{\theta}(\cdot | s_{t})\right]\right] \leq \delta
$$

在对目标进行线性逼近和约束的二次近似之后，使用共轭梯度算法可以有效地近似解决该问题。

证明TRPO的理论实际上建议使用惩罚而不是约束，即解决下面的无约束优化问题

![](../../.gitbook/assets/image%20%2847%29.png)

然而，TRPO使用一个约束来代替的原因是： 对于不同的问题最佳的$$β$$是不同的，甚至在一个任务的不同阶段都会变化。 

### Clipped Surrogate Objective

设 $$r_{l}(\theta)=\frac{\pi_{\theta}\left(a_{t} | s_{t}\right)}{\pi_{\theta_{\text { old }}}\left(a_{t} | s_{t}\right)}, \text { so } r\left(\theta_{\text { old }}\right)=1$$ ，上面的约束函数变为：

![](../../.gitbook/assets/image%20%289%29.png)

我们提出的代理损失为，即当概率比的变化会使目标函数提高时，我们忽略它：

![](../../.gitbook/assets/image%20%2830%29.png)

其中 $$\epsilon$$是超参数，如 $$\epsilon=0.2$$ ，下图是一个简单示例图

![](../../.gitbook/assets/image%20%2868%29.png)

### Adaptive KL Penalty Coefficient

![](../../.gitbook/assets/image%20%2841%29.png)

这种方法动态调节 $$β$$ ，但是效果没有CLIP好

