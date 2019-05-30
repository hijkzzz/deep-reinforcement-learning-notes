# Noisy-Net

> [Noisy Networks for Exploration](https://arxiv.org/pdf/1706.10295v2.pdf)

我们介绍了一种深度强化学习代理NoisyNet，该代理的权重中添加了参数噪声，并表明该代理策略的诱导随机性可以用来帮助有效的探索。噪声的参数通过梯度下降以及剩余的网络权重来学习。NoisyNet易于实现，并且几乎不增加计算开销。我们发现，用NoisyNet代替A3C、DQN和Dueling代理的传统探索试探法\(分别用熵奖励和 $$\epsilon$$ -greedy \)，在许多Atari游戏中，会获得更高的分数，在某些情况下，会将代理从低于人类游戏分数提升到超过人类的水平。

## 方法

噪声网络是一种神经网络，其权值和偏差受到噪声参数函数的扰动。这些参数是用梯度下降法调整的。设 $$y=f_{\theta}(x)$$ 是由带噪声的向量 $$\theta$$ 参数化的神经网络，定义为 $$\theta \stackrel{\mathrm{def}}{=} \mu+\Sigma \odot \varepsilon$$ ，其中 $$\zeta \stackrel{\mathrm{def}}{=}(\mu, \Sigma)$$ 是可学习的参数。

设线性函数 $$y=w x+b$$ ，则对应的噪声版本

$$
y \stackrel{\mathrm{def}}{=}\left(\mu^{w}+\sigma^{w} \odot \varepsilon^{w}\right) x+\mu^{b}+\sigma^{b} \odot \varepsilon^{b}
$$

现在我们转向噪声网络中线性层噪声分布的显式实例。我们探讨了两种选择:独立高斯噪声，它使用独立的高斯噪声项每权重，以及因子分解高斯噪声，它使用独立噪声每输出，而另一个独立噪声每输入。在我们的算法中，使用因子分解高斯噪声的主要原因是减少随机数生成的计算时间。

其中因子分解噪声即：

$$
\begin{aligned} \varepsilon_{i, j}^{w} &=f\left(\varepsilon_{i}\right) f\left(\varepsilon_{j}\right) \\ \varepsilon_{j}^{b} &=f\left(\varepsilon_{j}\right) \end{aligned}
$$

然后把带噪声的网络应用于Deep Q-Network或者A3C中的critic即可

![](../../.gitbook/assets/image%20%2872%29.png)

