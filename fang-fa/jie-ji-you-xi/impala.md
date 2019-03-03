# IMPALA

## 介绍

> [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/pdf/1802.01561.pdf)

在这项工作中，我们的目标是使用单个强化学习代理来解决大量任务。 一个关键的挑战是处理增加的数据量和延长的培训时间。 我们开发了一种新的分布式代理IMPALA（Importance Weighted Actor-Learner Architectures），它不仅可以在单机训练中更有效地使用资源，而且可以扩展到数千台机器而不会牺牲数据效率或资源利用率。 通过将解耦的行为学习与一种称为V-trace的离线校正方法相结合，我们实现了稳定的高吞吐学习。 我们展示了IMPALA在DMLab-30和Atari-57中的多任务强化学习的有效性。 我们的结果表明，IMPALA能够在数据较少的情况下取得比以前的代理更好的性能，并且由于其多任务处理方式，在任务之间表现出了正向的迁移。

## 算法

### IMPALA

![](../../.gitbook/assets/image%20%2823%29.png)

如图1，在每个轨迹的开始，actor将自己的本地策略 $$μ$$ 更新为最新的Leaner的策略 $$π$$ ，并在其环境中运行在n步之l后，actor通过队列将状态、动作和回报 $$\{x_{1}, a_{1}, r_{1}, \dots, x_{n}, a_{n}, r_{n}\}$$ 连同相应的策略分布 $$\mu\left(a_{t} | x_{t}\right)$$ 和初始LSTM状态一起发送给leaner。然后，leaner利用样本轨迹更新策略 $$π$$ 。这种简单的架构使得leaner能够使用GPU加速，并且actor能够轻松地分布在许多机器上。然而，在更新时，leaner的策略 $$π$$ 可能比actor的策略 $$μ$$ 提前更新几个版本，因此actor和leaner之间存在一种policy-lag。V-trace校正了这种lag，从而在保持数据效率的同时实现极高的数据吞吐量。使用actor-learner 架构，提供像分布式A3C这样的容错能力，但是由于actor并且没有发送参数/梯度，因此通信开销较低。

### V-trace

在解耦的分布式actor-critic架构中，off-policy学习很重要，因为actor生成动作与leaner估计梯度之间存在滞后。 为此，我们为学习者引入了一种新颖的off-policy actor-critic算法，称为V-trace。

off-policy RL算法的目标是使用某个策略 $$μ$$ 生成的轨迹，称为行为策略，来学习另一个策略 $$π$$ \(可能不同于 $$μ$$ \)的值函数 $$v_π$$ ，称为目标策略。

#### V-trace target

考虑行为策略 $$μ$$ 生成的轨迹 $$\left(x_{t}, a_{t}, r_{t}\right)_{t=s}^{t=s+n}$$ ，n-steps V-trace的目标函数为

$$v_{s} \stackrel{\mathrm{def}}{=} \quad V\left(x_{s}\right)+\sum_{t=s}^{s+n-1} \gamma^{t-s}\left(\prod_{i=s}^{t-1} c_{i}\right) \delta_{t} V$$ 

其中 $$\delta_{t} V \stackrel{\mathrm{def}}{=} \rho_{t}\left(r_{t}+\gamma V\left(x_{t+1}\right)-V\left(x_{t}\right)\right)$$ 是时间差分。 $$\rho_{t} \stackrel{\mathrm{def}}{=} \min \left(\overline{\rho}, \frac{\pi\left(a_{t} | x_{t}\right)}{\mu\left(a_{t} | x_{t}\right)}\right)$$ 和 $$c_{i} \stackrel{\mathrm{def}}{=} \min \left(\overline{c}, \frac{\pi\left(a_{i} | x_{i}\right)}{\mu\left(a_{i} | x_{i}\right)}\right)$$ 是截断的重要性采样权重，

#### Actor-Critic algorithm



## 实验

![](../../.gitbook/assets/image%20%2811%29.png)

