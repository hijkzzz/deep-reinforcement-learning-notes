# Asynchronous Deep Reinforcement Learning

## 介绍

> [Asynchronous methods for deep reinforcement learning](https://arxiv.org/pdf/1602.01783.pdf)

我们为深度强化学习提出了一个概念上简单的轻量级框架，它使用异步梯度下降来优化深度神经网络控制器。我们提出了标准强化学习算法的异步变体，并表明并行Actor-Critic对训练有着不稳定的影响，允许所有四种方法成功训练神经网络控制器。最佳性能方法是Actor-Critic的同步变体，它超过了Atari领域的最新水平，同时在单个多核CPU而不是GPU上训练了一半时间。此外，我们还表明异步Actor-Critic成功地解决了各种各样的连续电机控制问题，以及使用视觉输入导航随机3D迷宫的新任务。

## 算法

现在我们介绍单步Sarsa、单步Q-learning、n-step Q-learning和advantage actor-critic的多线程异步变体。设计这些方法的目的是寻找能够可靠地训练深度神经网络策略且不需要大量资源的RL算法。虽然底层的RL方法差别很大，actor- critic是一种on-policy搜索方法，Q-learning是一种off-policy value-based 方法，但是我们使用两种主要思想使这四种算法在给定我们的设计目标的情况下都具有实用性。

通过在不同的线程中运行不同的探索策略，与应用在线更新的单个代理相比，并行应用在线更新的多个actor-leaners对参数所做的总体更改在时间上可能不太相关。因此，我们不使用经验回放，而是依靠采用不同探测策略的并行程序来执行DQN训练算法中的经验重放所承担的稳定角色。

### 异步 1-step Q-learning

每个线程与自己的环境副本进行交互，并在每个步骤中计算Q-learning损耗的梯度。在计算Q-learning损耗时，我们采用了DQN训练方法中提出的共享且缓慢变化的目标网络。在应用梯度之前，我们还会在多个时间步长上累积梯度，这减少了多个actor-leaner重写彼此更新的几率。通过几个步骤累积更新还提供了以计算效率换取数据效率的能力。

最后，我们发现给每个线程一个不同的探索策略有助于提高健壮性。以这种方式增加勘探的多样性通常也会通过更好的勘探来提高性能

![](../../.gitbook/assets/image%20%2840%29.png)

### 异步优势 Actor-Critic\(A3C\)

![](../../.gitbook/assets/image%20%2853%29.png)

我们还发现，将将策略 $$π$$ 的熵添加到目标函数中，通过阻止结构收敛到次优的确定性策略改进了探索。包括熵正则化项在内的全目标函数对政策参数的梯度形式为：

$$
\nabla_{\theta^{\prime}} \log \pi\left(a_{t} | s_{t} ; \theta^{\prime}\right)\left(R_{t}-V\left(s_{t} ; \theta_{v}\right)\right)+\beta \nabla_{\theta^{\prime}} H\left(\pi\left(s_{t} ; \theta^{\prime}\right)\right)
$$

其中 $$H$$ 是熵

### 优势 Actor-Critic\(A2C\)

A2C即A3C的同步版本

![](../../.gitbook/assets/image%20%2818%29.png)

## 实验

### 雅达利 2600 游戏

![](../../.gitbook/assets/image%20%2878%29.png)







