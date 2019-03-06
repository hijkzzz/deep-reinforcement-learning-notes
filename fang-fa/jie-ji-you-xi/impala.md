# IMPALA

## 介绍

> [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/pdf/1802.01561.pdf)

在这项工作中，我们的目标是使用单个强化学习代理来解决大量任务。 一个关键的挑战是处理增加的数据量和延长的培训时间。 我们开发了一种新的分布式代理IMPALA（Importance Weighted Actor-Learner Architectures），它不仅可以在单机训练中更有效地使用资源，而且可以扩展到数千台机器而不会牺牲数据效率或资源利用率。 通过将解耦的行为学习与一种称为V-trace的校正方法相结合，我们实现了稳定的高吞吐学习。 我们展示了IMPALA在DMLab-30和Atari-57中的多任务强化学习的有效性。 我们的结果表明，IMPALA能够在数据较少的情况下取得比以前的代理更好的性能，并且由于其多任务处理方式，在任务之间表现出了正向的迁移。

## 算法

### IMPALA

![](../../.gitbook/assets/image%20%2831%29.png)

如图1，在每个轨迹的开始，actor将自己的本地策略 $$μ$$ 更新为最新的Leaner的策略 $$π$$ ，并在其环境中运行在n步之l后，actor通过队列将状态、动作和回报 $$\{x_{1}, a_{1}, r_{1}, \dots, x_{n}, a_{n}, r_{n}\}$$ 连同相应的策略分布 $$\mu\left(a_{t} | x_{t}\right)$$ 和初始LSTM状态一起发送给leaner。然后，leaner利用样本轨迹更新策略 $$π$$ 。这种简单的架构使得leaner能够使用GPU加速，并且actor能够轻松地分布在许多机器上。然而，在更新时，leaner的策略 $$π$$ 可能比actor的策略 $$μ$$ 提前更新几个版本，因此actor和leaner之间存在一种policy-lag。V-trace校正了这种lag，从而在保持数据效率的同时实现极高的数据吞吐量。使用actor-learner 架构，提供像分布式A3C这样的容错能力，但是由于actor并且没有发送参数/梯度，因此通信开销较低。

### V-trace

在解耦的分布式actor-critic架构中，off-policy学习很重要，因为actor生成动作与leaner估计梯度之间存在滞后。 为此，我们为学习者引入了一种新颖的off-policy actor-critic算法，称为V-trace。

off-policy RL算法的目标是使用某个策略 $$μ$$ 生成的轨迹，称为行为策略，来学习另一个策略 $$π$$ \(可能不同于 $$μ$$ \)的值函数 $$v_π$$ ，称为目标策略。

#### V-trace target

考虑行为策略 $$μ$$ 生成的轨迹 $$\left(x_{t}, a_{t}, r_{t}\right)_{t=s}^{t=s+n}$$ ，n-steps V-trace的价值函数为

$$v_{s} \stackrel{\mathrm{def}}{=} \quad V\left(x_{s}\right)+\sum_{t=s}^{s+n-1} \gamma^{t-s}\left(\prod_{i=s}^{t-1} c_{i}\right) \delta_{t} V$$ 

上式中 $$\delta_{t} V \stackrel{\mathrm{def}}{=} \rho_{t}\left(r_{t}+\gamma V\left(x_{t+1}\right)-V\left(x_{t}\right)\right)$$ 是时间差分。 $$\rho_{t} \stackrel{\mathrm{def}}{=} \min \left(\overline{\rho}, \frac{\pi\left(a_{t} | x_{t}\right)}{\mu\left(a_{t} | x_{t}\right)}\right)$$ 和 $$c_{i} \stackrel{\mathrm{def}}{=} \min \left(\overline{c}, \frac{\pi\left(a_{i} | x_{i}\right)}{\mu\left(a_{i} | x_{i}\right)}\right)$$ 是截断的重要性采样权重，其中 $$ \prod_{i=s}^{t-1} c_{i}=1 for (t = s)$$ ，且假设 $$\overline{\rho} \geq \overline{c}$$ 。

注意：对于on-policy的情况，我们假定 $$\overline{c} \geq 1$$ ， $$c_{i}=1 \text { and } \rho_{t}=1$$ ，上式可被写为：

$$\begin{aligned} v_{s} &=V\left(x_{s}\right)+\sum_{t=s}^{s+n-1} \gamma^{t-s}\left(r_{t}+\gamma V\left(x_{t+1}\right)-V\left(x_{t}\right)\right) \\ &=\sum_{t=s}^{s+n-1} \gamma^{t-s} r_{t}+\gamma^{n} V\left(x_{s+n}\right) \end{aligned}$$ 

所以在on-policy的时候，V-trace退化为on-policy n-steps Bellman update，这个性质允许V-trace同时用于off-policy和on-policy。

注意：截断的重要性采样权重 $$c_{i} \text { and } \rho_{t}$$ 的作用是不同的， $$\rho_{t}$$ 出现在时间差分 $$\delta_{t} V $$ 中，定义了更新的fixed point。在表格法（强化学习）中，更新的fixed point（即当 $$V\left(x_{s}\right)=v_{s}\ for\ all\ state$$ ），特征为 $$\delta_{t} V$$的期望等于0（在策略 $$μ$$ 下），是一些策略 $$\pi_{\overline{\rho}}$$ （定义如下）的值函数 $$V^{\pi} \overline{\rho}$$ 

$$\pi_{\overline{\rho}}(a | x) \stackrel{\mathrm{def}}{=} \frac{\min (\overline{\rho} \mu(a | x), \pi(a | x))}{\sum_{b \in A} \min (\overline{\rho} \mu(b | x), \pi(b | x))}$$ 

这意味着，当 $$\overline{\rho}<\infty$$ 时，策略 $$\pi_{\overline{\rho}}$$ 介于目标策略 $$\pi$$ 和行为策略$$μ$$之间。而 $$\overline{\rho}=\infty$$ 时候，该策略等于目标策略$$\pi$$。

$$c_{i}$$ 类似于"trace cutting"系数， $$c_{s} \ldots c_{t-1}$$ 的乘积度量着时间差分$$\delta_{t} V $$在时间 $$t$$ 出现的频率，影响着 $$t=s$$ 的值函数更新。目标策略$$\pi$$和行为策略$$\pi$$相差越大，这个乘积的方差越大，这里通过截断来限制方差。

总的来说： $$\overline{\rho}$$ 影响收敛到的价值函数的性质， $$\overline{c}$$ 影响收敛到这个函数的速度。

![](../../.gitbook/assets/image%20%2857%29.png)

#### Actor-Critic algorithm

_策略梯度\(Policy Gradient\)_

在on-policy的情况下，价值函数关于策略 $$μ$$ 的参数的梯度为

$$\nabla V^{\mu}\left(x_{0}\right)=\mathbb{E}_{\mu}\left[\sum_{s \geq 0} \gamma^{s} \nabla \log \mu\left(a_{s} | x_{s}\right) Q^{\mu}\left(x_{s}, a_{s}\right)\right]$$ 

其中： $$Q^{\mu}\left(x_{s}, a_{s}\right) \stackrel{\mathrm{def}}{=} \mathbb{E}_{\mu}\left[\sum_{t \geq s} \gamma^{t-s} r_{t} | x_{s}, a_{s}\right]$$ 

现在考虑off-policy的情况，我们可以重要性权重来更新策略参数：

$$\mathbb{E}_{a_{s} \sim \mu(\cdot | x_{s})}\left[\frac{\pi_{\overline{\rho}}\left(a_{s} | x_{s}\right)}{\mu\left(a_{s} | x_{s}\right)} \nabla \log \pi_{\overline{\rho}}\left(a_{s} | x_{s}\right) q_{s} | x_{s}\right]$$ 

其中： $$q_{s} \stackrel{\mathrm{def}}{=} r_{s}+\gamma v_{s+1}$$ _，_最后为了减少方差，我们减去了一个强化学习中的基数 $$V\left(x_{s}\right)$$ 

_扩展到Actor-Critic_

critic梯度

$$\left(v_{s}-V_{\theta}\left(x_{s}\right)\right) \nabla_{\theta} V_{\theta}\left(x_{s}\right)$$ __

_actor梯度_

\_\_$$\rho_{s} \nabla_{\omega} \log \pi_{\omega}\left(a_{s} | x_{s}\right)\left(r_{s}+\gamma v_{s+1}-V_{\theta}\left(x_{s}\right)\right)$$ __

为了防止过早收敛，我们可能增加_一个_熵

$$-\nabla_{\omega} \sum \pi_{\omega}(a | x_{s}) \log \pi_{\omega}(a | x_{s})$$ 

## 实验

### 训练性能

![](../../.gitbook/assets/image%20%2817%29.png)

### 游戏测试

![](../../.gitbook/assets/image%20%2864%29.png)

