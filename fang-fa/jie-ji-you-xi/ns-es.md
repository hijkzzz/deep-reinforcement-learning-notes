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





