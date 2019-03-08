# Trust Region Policy Optimization

## 介绍

> [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)

我们描述了一个优化策略的迭代过程，保证了单调的改进。 通过对理论上合理的过程进行几次近似，我们开发了一种称为信任区域策略优化（TRPO）的实用算法。 该算法与自然策略梯度方法类似，对于优化大型非线性策略（如神经网络）是有效的。 我们的实验证明了它在各种任务中的强大性能：学习模拟机器人游泳，跳跃和步行步态; 并使用屏幕图像作为输入玩Atari游戏。 尽管它的近似值偏离了理论，但TRPO倾向于提供单调的改进，通过微调超参数。

## 算法

### 预备

定义期望折扣回报

$$\begin{array}{l}{\eta(\pi)=\mathbb{E}_{s 0, a_{0}, \ldots}\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(s_{t}\right)\right], \text { where }} \\ {s_{0} \sim \rho_{0}\left(s_{0}\right), a_{t} \sim \pi\left(a_{t} | s_{t}\right), s_{t+1} \sim P\left(s_{t+1} | s_{t}, a_{t}\right)}\end{array}$$ 

定义价值函数、动作价值函数、动作优势价值函数

$$
\begin{array}{l}{Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{s_{t+1}, a_{t+1}, \ldots}\left[\sum_{l=0}^{\infty} \gamma^{l} r\left(s_{t+l}\right)\right]} \\ {V_{\pi}\left(s_{t}\right)=\mathbb{E}_{a_{t}, s_{t+1}, \ldots}\left[\sum_{l=0}^{\infty} \gamma^{l} r\left(s_{t+l}\right)\right]} \\ {A_{\pi}(s, a)=Q_{\pi}(s, a)-V_{\pi}(s), \text { where }} \\ {a_{t} \sim \pi\left(a_{t} | s_{t}\right), s_{t+1} \sim P\left(s_{t+1} | s_{t}, a_{t}\right) \text { for } t \geq 0}\end{array}
$$

定义策略 $$\tilde{\pi}$$ 相对于策略 $$\pi$$ 的期望优势

$$\eta(\tilde{\pi})=\eta(\pi)+\mathbb{E}_{s_{0}, a_{0}, \cdots \sim \tilde{\pi}}\left[\sum_{t=0}^{\infty} \gamma^{t} A_{\pi}\left(s_{t}, a_{t}\right)\right]$$ 

即动作根据$$\tilde{\pi}$$采样，策略评估还是使用 $$\pi$$

令 $$\rho_{\pi}(s)=P\left(s_{0}=s\right)+\gamma P\left(s_{1}=s\right)+\gamma^{2} P\left(s_{2}=s\right)+\ldots$$ ，上式可写为

$$
\begin{aligned} \eta(\tilde{\pi}) &=\eta(\pi)+\sum_{t=0} \sum_{s} P\left(s_{t}=s | \tilde{\pi}\right) \sum_{a} \tilde{\pi}(a | s) \gamma^{t} A_{\pi}(s, a) \\ &=\eta(\pi)+\sum_{s} \sum_{t=0}^{\infty} \gamma^{t} P\left(s_{t}=s | \tilde{\pi}\right) \sum_{a} \tilde{\pi}(a | s) A_{\pi}(s, a) \\ &=\eta(\pi)+\sum_{s} \rho_{\tilde{\pi}}(s) \sum_{a} \tilde{\pi}(a | s) A_{\pi}(s, a) \end{aligned}
$$

这个式子表明只要能保证 $$\sum_{a} \tilde{\pi}(a | s) A_{\pi}(s, a) \geq 0$$ ，策略的期望回报就能增大。考虑策略迭代 $$\tilde{\pi}(s)=\arg \max _{a} A_{\pi}(s, a)$$ ，只要一个 $$A_\pi$$ 大于0，策略就会被提升，否则收敛。但是在近似的情况下，由于误差的存在，不可避免的有 $$\sum_{a} \tilde{\pi}(a | s) A_{\pi}(s, a)<0$$ 出现。为了简化问题我们提出了下面的近似公式：

$$
L_{\pi}(\tilde{\pi})=\eta(\pi)+\sum_{s} \rho_{\pi}(s) \sum_{a} \tilde{\pi}(a | s) A_{\pi}(s, a)
$$

注意这里使用的是 $$\rho_{\pi}(s)$$ 而不是 $$\rho_{\tilde{\pi}}(s)$$ ，即使用旧策略采样的状态频率。

因为 $$L_{\pi}$$ 与 $$\eta$$ 一阶匹配，则有梯度

$$
\begin{aligned} L_{\pi_{\theta_{0}}}\left(\pi_{\theta_{0}}\right) &=\eta\left(\pi_{\theta_{0}}\right) \\ \nabla_{\theta} L_{\pi_{\theta_{0}}}\left.\left(\pi_{\theta}\right)\right|_{\theta=\theta_{0}} &=\nabla_{\theta} \eta\left.\left(\pi_{\theta}\right)\right|_{\theta=\theta_{0}} \end{aligned}
$$

上式给出了一个增强策略的梯度方向，然而却没有指出最佳的步长，为了解决这个问题

令 $$\pi^{\prime}=\arg \max _{\pi^{\prime}} L_{\pi_{\mathrm{old}}}\left(\pi^{\prime}\right)$$ ，新策略为 $$\pi_{\text { new }}(a | s)=(1-\alpha) \pi_{\text { old }}(a | s)+\alpha \pi^{\prime}(a | s)$$ 

有下界

$$\eta\left(\pi_{\text { new }}\right) \geq L_{\pi_{\text { old }}}\left(\pi_{\text { new }}\right)-\frac{2 \epsilon \gamma}{(1-\gamma)^{2}} \alpha^{2} \\ where\ \epsilon=\max _{s}\left|\mathbb{E}_{a \sim \pi^{\prime}(a | s)}\left[A_{\pi}(s, a)\right]\right|$$

###  一般随机策略的单调改进保证













