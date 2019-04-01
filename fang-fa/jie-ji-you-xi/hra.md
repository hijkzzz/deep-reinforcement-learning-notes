# HRA

## 介绍

> [Hybrid Reward Architecture for Reinforcement Learning](https://arxiv.org/pdf/1706.04208.pdf)

强化学习（RL）的主要挑战之一是泛化。 非典型深度RL方法这是通过使用深度网络用低维表示近似最优值函数来实现的。 虽然这种方法在许多领域都很有效，但在最佳值函数不能简单地降低到低维表示的领域中，学习可能非常缓慢和不稳定。 本文通过提出一种称为混合奖励架构（HRA）的新方法，有助于解决这些具有挑战性的领域.HRA将分解的奖励函数作为输入，并为每个组件奖励函数学习单独的价值函数。 因为每个组件通常仅依赖于所有特征的子集，所以相应的值函数可以通过低维表示更容易地接近，从而实现更有效的学习。我们将在一个玩具问题和Atari游戏《吃豆人》中演示HRA，在游戏中，HRA实现了高于人类的性能。

## 算法

通常使用具有权重向量θ的函数逼近器来估计Q值函数： $$Q(s,a;θ)$$ 。 DQN使用深度神经网络作为函数逼近器，并通过最小化损失函数的顺序迭代地改进 $$Q^*$$ 。

$$
\begin{aligned} \mathcal{L}_{i}\left(\theta_{i}\right) &=\mathbb{E}_{s, a, r, s^{\prime}}\left[\left(y_{i}^{D Q N}-Q\left(s, a ; \theta_{i}\right)\right)^{2}\right] \\ \text { with } \quad y_{i}^{D Q N} &=r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta_{i-1}\right) \end{aligned}
$$

为了使得DQN更容易学习奖励，我们建议分解奖励函数 $$R_{e n v}$$为 $$n$$ 个函数

$$
R_{e n v}\left(s, a, s^{\prime}\right)=\sum_{k=1}^{n} R_{k}\left(s, a, s^{\prime}\right), \quad \text { for all } s, a, s^{\prime}
$$

并且在每一个奖励函数中上训练单独的强化学习代理。奖励函数可能有无限多种不同的分解，但是为了实现易于学习的价值函数，分解应该使得每个奖励函数主要受少量状态变量的影响。

![](../../.gitbook/assets/image%20%2817%29.png)

代理的集合也可以被视为具有多个头的单个代理，每个头在不同的奖励函数下产生当前状态的动作值。HRA的损失函数如下：

$$
\begin{aligned} \mathcal{L}_{i}\left(\theta_{i}\right) &=\mathbb{E}_{s, a, r, s^{\prime}}\left[\sum_{k=1}^{n}\left(y_{k, i}-Q_{k}\left(s, a ; \theta_{i}\right)\right)^{2}\right] \\ \text { with } & y_{k, i}=R_{k}\left(s, a, s^{\prime}\right)+\gamma \max _{a^{\prime}} Q_{k}\left(s^{\prime}, a^{\prime} ; \theta_{i-1}\right) \end{aligned}
$$

$$
Q_{\mathrm{HRA}}^{*}(s, a) :=\sum_{k=1}^{n} Q_{k}^{*}(s, a) \quad \text { for all } s, a
$$

接下来的问题是如何分解回报，这通常需要相关的领域知识，如下：

![](../../.gitbook/assets/image%20%2857%29.png)

