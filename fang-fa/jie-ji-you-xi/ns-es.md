# NS-ES

## 介绍

> [Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents](https://arxiv.org/abs/1712.06560)

进化策略\(ES\)是一组黑盒优化算法，能够粗略地训练深度神经网络，就像Q-learning和策略梯度方法来挑战深度强化学习\(RL\)问题，但速度要快得多\(比如小时vs天\)，因为它们的并行性更好。然而，许多RL问题需要定向探索，因为它们具有稀疏的或欺骗性的奖励\(即包含局部最优\)，而且还不知道如何用ES来鼓励更多的探索。在这里，我们展示了一些已经被发明出来的算法，这些算法通过探索代理的大量出现来促进在小规模进化的神经网络中的定向探索，特别是新颖搜索\(NS\)和质量多样性\(QD\)算法，可以与ES杂交，以提高其在稀疏或欺骗性的深层RL任务中的性能，同时保持可扩展性。我们的实验证实了新算法——神经网络进化算法和两种量子进化算法，NSR-ES和NSRA-ES，避免了神经网络进化算法遇到的局部最优，从而在雅达利和学习绕过欺骗陷阱的模拟机器人上获得了更高的性能。因此，本文介绍了一系列快速、可扩展的强化学习算法，这些算法能够进行定向探索。它还将这一新的探索算法族添加到了推理工具箱中，并提出了一种有趣的可能性，即具有多条同时探索路径的相似算法也可能与专家系统之外的现有推理算法相结合。

## 算法

### Evolution Strategies\(ES\)

NES将群体表示为由参数 $$\phi : p_{\phi}(\theta)$$ 表征的参数向量 $$θ$$ 的分布，对于适应函数 $$f(\theta)$$ ，NES旨在最大限度地提高群体的平均适应度 $$\mathbb{E}_{\theta \sim p_{\phi}}[f(\theta)]$$ ，通过梯度下降优化$$\phi : p_{\phi}(\theta)$$ 。

最近OpenAI的工作把NES应用于RL（见上一篇ES），设群体分布为 $$\theta_{t}^{i} \sim \mathcal{N}\left(\theta_{t}, \sigma^{2} I\right)$$ ，类似于REINFORCE，梯度为

$$
\nabla_{\phi} \mathbb{E}_{\theta \sim \phi}[f(\theta)] \approx \frac{1}{n} \sum_{i=1}^{n} f\left(\theta_{t}^{i}\right) \nabla_{\phi} \log p_{\phi}\left(\theta_{t}^{i}\right)
$$

设 $$\theta_{t}^{i}=\theta_{t}+\sigma \epsilon_{i} \text { where } \epsilon_{i} \sim \mathcal{N}(0, I)$$ ，通过对样本参数扰动的总和进行加权来估计梯度

$$
\nabla_{\theta_{t}} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\left[f\left(\theta_{t}+\sigma \epsilon\right)\right] \approx \frac{1}{n \sigma} \sum_{i=1}^{n} f\left(\theta_{t}^{i}\right) \epsilon_{i}
$$

### Novelty Search\(NS\)

NS鼓励政策参与明显不同于前面所见的行为。 该算法通过计算当前策略与先前生成的策略的相关性来鼓励不同的行为，然后鼓励人口分布向参数空间区域移动，具有高新颖性。在NS中，策略 $$π$$ 被赋予描述其行为的域相关行为特征 $$b(π)$$ 。例如，在人形运动问题的情况下， $$b(π)$$ 可以简单到包含人形的最终位置 $${x，y }$$ 的二维向量。在整个训练过程中，每个被评估的 $$π_θ$$ 都会以一定的概率将 $$b\left(\pi_{\theta}\right)$$ 加到档案集 $$A$$ 中。然后，通过从一个特定策略中选择 $$b\left(\pi_{\theta}\right)$$ 的k -最近邻居并计算它们之间的平均距离，来计算该策略的新奇值 $$N\left(b\left(\pi_{\theta}\right), A\right)$$ 。

$$
\begin{array}{c}{N(\theta, A)=N\left(b\left(\pi_{\theta}\right), A\right)=\frac{1}{|S|} \sum_{j \in S}\left\|b\left(\pi_{\theta}\right)-b\left(\pi_{j}\right)\right\|_{2}} \\ {S=k N N\left(b\left(\pi_{\theta}\right), A\right)} \\ {=\left\{b\left(\pi_{1}\right), b\left(\pi_{2}\right), \ldots, b\left(\pi_{k}\right)\right\}}\end{array}
$$

### NS-ES

我们使用上面中描述的ES优化框架来计算期望新颖性的梯度

$$
\nabla_{\theta_{t}} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\left[N\left(\theta_{t}+\sigma \epsilon, A\right) | A\right] \approx \frac{1}{n \sigma} \sum_{i=1}^{n} N\left(\theta_{t}^{i}, A\right) \epsilon_{i}
$$

获得的梯度估计告诉我们如何改变当前策略的参数θ以增加我们的参数分布的平均新颖性。

我们初始化 $$M$$ 个随机参数向量作为元群体，并在每次迭代时选择一个进行更新。实验中，我们根据 ****$$\theta^{m}$$ 的新颖性计算概率，从离散概率分布中选择更新的向量。

$$
P\left(\theta^{m}\right)=\frac{N\left(\theta^{m}, A\right)}{\sum_{j=1}^{M} N\left(\theta^{j}, A\right)}
$$

在从元群体中选择一个个体后，我们计算当前参数向量 $$θ^m_t$$ 的期望新颖性梯度，并相应地执行更新步骤：

$$
\theta_{t+1}^{m} \leftarrow \theta_{t}^{m}+\alpha \frac{1}{n \sigma} \sum_{i=1}^{n} N\left(\theta_{t}^{i, m}, A\right) \epsilon_{i}
$$

更新当前参数向量后，将计算 $$b\left(\pi_{\theta_{t+1}^{m}}\right)$$ 并将其添加到共享归档 $$A$$ 中。整个过程重复预定的迭代次数，因为NS没有真正的收敛点。 在训练期间，算法会保留具有最高平均奖励的策略，并在训练完成后返回此策略。

![](../../.gitbook/assets/image%20%2851%29.png)

### QD-ES\(NSR-ES and NSRA-ES\)

单独的NS-ES可以使代理人避免在奖励功能中欺骗性的局部最优。 然而，回报信号仍然提供非常丰富的信息，完全丢弃它们可能会导致性能下降。因此，我们训练NS-ES的变体，我们称之为NSR-ES，它结合了（“适应性”）和针对给定的一组策略参数 $$\theta$$ 计算的新颖性。

$$
\theta_{t+1}^{m} \leftarrow \theta_{t}^{m}+\alpha \frac{1}{n \sigma} \sum_{i=1}^{n} \frac{f\left(\theta_{t}^{i, m}\right)+N\left(\theta_{t}^{i, m}, A\right)}{2} \epsilon_{i}
$$

直观地说，该梯度朝向既表现出新颖行为又获得高回报的策略。然而，通常， $$f\left(\theta_{t}^{i, m}\right)$$ 和 $$N\left(\theta_{t}^{i, m}, A\right)$$ 的尺度不同。 为了有效地组合这两个信号，我们在计算平均值之前独立地对它们进行rank-normalize。

![](../../.gitbook/assets/image%20%2814%29.png)

NSR-ES具有相同的回报和新颖度梯度权重，在训练中是静态的。我们探索NSR-ES的进一步扩展，称为NSRAdapt-ES（NSRA-ES），它通过在训练期间智能地调整加权参数 $$w$$ ，以此动态调节 $$f\left(\theta_{t}^{i, m}\right)$$与$$N\left(\theta_{t}^{i, m}, A\right)$$的优先级。

$$
\theta_{t+1}^{m} \leftarrow \theta_{t}^{m}+\alpha_{\overline{n} \sigma}^{n} \sum_{i=1}^{n} w f\left(\theta_{t}^{i, m}\right) \epsilon_{i}+(1-w) N\left(\theta_{t}^{i, m}, A\right) \epsilon_{i}
$$

我们最初设置 $$w = 1.0$$ ，如果性能在固定数量的代间停滞不前，我们会降低它。我们继续下降，直到性能提高，这时我们增加。

![](../../.gitbook/assets/image%20%2867%29.png)

## 实验

![](../../.gitbook/assets/image%20%2868%29.png)

![](../../.gitbook/assets/image%20%2846%29.png)









