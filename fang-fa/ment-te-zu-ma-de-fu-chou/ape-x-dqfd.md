# Ape-X  DQfD

## 介绍

> [Observe and Look Further: Achieving Consistent Performance on Atari](https://arxiv.org/pdf/1805.11593.pdf)

尽管深度强化学习（RL）领域取得了重大进展，但今天的算法仍然无法在Atari 2600游戏等一系列多项任务中持续学习人类级别的策略。我们确定了任何算法都需要掌握的三个关键挑战，才能在所有游戏中表现出色:处理各种奖励分配、长期推理和高效探索。在这篇文章中，我们提出了一个算法来解决每一个挑战，并且能够学习几乎所有雅达利游戏中的人类层面的策略。一个新的transformed Bellman算子允许我们的算法处理不同密度和尺度的奖励; 辅助时间一致性损失使我们能够使用 $$\gamma$$ = 0.999（而不是 $$\gamma$$ = 0.99）的贴现因子稳定地训练，将有效规划的范围扩大一个数量级;我们通过使用人类演示来引导代理人奖励状态，从而缓解探索问题。

## 方法

### Transformed Bellman Operator

$$
(\mathcal{T} Q)(x, a) :=\mathbb{E}_{x^{\prime} \sim P(\cdot | x, a)}\left[R(x, a)+\gamma \max _{a^{\prime} \in \mathcal{A}} Q\left(x^{\prime}, a^{\prime}\right)\right], \quad \forall(x, a) \in \mathcal{X} \times \mathcal{A}
$$

在深度强化学习中，如果 $$\mathcal{T} f_{\theta^{(k-1)}} )(x, a)$$ 的方差太大，容易使训练不稳定而无法收敛，一种方法是截断回报的分布于区间 $$[-1,1]$$ 。不过，我们建议将重点放在行动价值函数上，而不是减少奖励的幅度，我们使用一个函数: $$R→R$$ 来缩小动作值函数的范围。

$$
\left(\mathcal{T}_{h} Q\right)(x, a) :=\mathbb{E}_{x^{\prime} \sim P(\cdot | x, a)}\left[h\left(R(x, a)+\gamma \max _{a^{\prime} \in \mathcal{A}} h^{-1}\left(Q\left(x^{\prime}, a^{\prime}\right)\right)\right)\right], \quad \forall(x, a) \in \mathcal{X} \times \mathcal{A}
$$

![](../../.gitbook/assets/image%20%2868%29.png)

上述定理说明了新的Q函数的收敛性，我们的算法中使用 $$h : z \mapsto \operatorname{sign}(z)(\sqrt{|z|+1}-1)+\varepsilon z \text { with } \varepsilon=10^{-2}$$ 。

新的损失函数可写为：

![](../../.gitbook/assets/image%20%2873%29.png)

### Temporal consistency \(TC\) loss

虽然变换后的贝尔曼算子提供了目标尺度和方差的缩减，但是当折扣因子γ接近1时，不稳定性仍然会发生。增加折扣系数会减少非奖励状态之间的时间价值差异。特别地，神经网络 $$f_{\theta}$$ 到下一个状态 $$x'$$ 的不希望的泛化（由于时间上相邻的目标值的相似性）可能导致灾难性的TD backup。我们通过添加表单的辅助时间一致性（TC）损失来解决问题：

$$
L_{\mathrm{TC}}\left(\theta ;\left(t_{i}\right)_{i=1}^{N},\left(p_{i}\right)_{i=1}^{N}, \theta^{(k-1)}\right) :=\sum_{i=1}^{N} p_{i} \mathcal{L}\left(f_{\theta}\left(x_{i}^{\prime}, a_{i}^{\prime}\right)-f_{\theta^{(k-1)}}\left(x_{i}^{\prime}, a_{i}^{\prime}\right)\right)
$$

TC损失惩罚改变下一个动作值估计 $$f_{\theta}\left(x^{\prime}, a^{\prime}\right)$$ 的权重更新。。这确保了更新后的估计值符合操作要求，从而随着时间的推移保持一致。

### Ape-X DQfD

在本节中，我们将描述如何将变换后的Bellman算子和TC损失与DQfD算法和分布式优先级经验重放相结合。整体的架构和Ape-X类似，不过引入了额外的损失函数。

![](../../.gitbook/assets/image%20%2812%29.png)

#### Leaner Process

用于模仿演示学习的监督损失为（DQfD）：

$$
L_{\mathrm{IM}}\left(\theta ;\left(t_{i}\right)_{i=1}^{N},\left(p_{i}\right)_{i=1}^{N}, \theta^{(k-1)}\right) :=\sum_{i=1}^{N} p_{i} e_{i}\left(\max _{a \in \mathcal{A}}\left[f_{\theta}\left(x_{i}, a\right)+\lambda \delta_{a \neq a_{i}}\right]-f_{\theta}\left(x_{i}, a_{i}\right)\right)
$$

总的损失函数为：

$$
L\left(\theta ;\left(t_{i}\right)_{i=1}^{N},\left(p_{i}\right)_{i=1}^{N}, \theta^{(k-1)}\right) :=\left(L_{\mathrm{TD}}+L_{\mathrm{TC}}+L_{\mathrm{IM}}\right)\left(\theta ;\left(t_{i}\right)_{i=1}^{N},\left(p_{i}\right)_{i=1}^{N}, \theta^{(k-1)}\right)
$$

![](../../.gitbook/assets/image%20%2843%29.png)

## 测试

![](../../.gitbook/assets/image%20%2813%29.png)





