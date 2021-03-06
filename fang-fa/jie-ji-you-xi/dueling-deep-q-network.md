# Dueling DQN

> [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)

近年来，在强化学习中使用深度表征已经取得了许多成功。尽管如此，这些应用程序中的许多使用传统的架构，如卷积网络、LSTM或自动编码器。在这篇论文中，我们提出了一种新的无模型强化学习的神经网络结构。我们的dueling网络代表两个独立的估计:一个用于状态值函数，一个用于状态相关的行动优势函数。这种分解的主要好处是，在不改变基础强化学习算法的情况下，将跨动作的学习通用化。我们的结果表明，在许多相似价值的行为面前，这种结构导致了更好的策略评估。此外，dueling的架构使我们的RL代理能够在Atari域上达到最优的效果。

## 方法

![](../../.gitbook/assets/image-9.png)

正如图2所示，我们的新架构背后的关键观点是，对于许多状态来说，没有必要估算每个动作选择的价值。例如，在耐力赛游戏中，知道是否向左或向右移动只在碰撞明显时才重要。在某些状态，知道采取何种行动至关重要，但在许多其他状态，行动的选择对发生的事情没有影响。然而，对于基于自举的算法，状态值的估计对于每种状态都非常重要。

为了实现这一目标，我们设计了一个单一的Q网络架构，如图1所示，它们更像是dueling网络。 与最初的DQN一样，dueling网络的较低层是卷积。 然而，我们不是使用单个完全连接层序列跟随卷积层，而是使用完全连接层的两个序列（或流）。 构造两个流使得它们具有提供价值和优势函数的单独估计的能力

![](../../.gitbook/assets/image-73.png)

利用下式，我们可以在DQN算法中使用该网络结构

$$Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+A(s, a ; \theta, \alpha)$$

但是这个式子有一个缺点，即无法单独提取出真实的 $$V$$ （价值）和 $$A$$ （动作优势）值，可以考虑一个简单的情况，即$$V$$ 加上一个常量、$$A$$ 减去一个常量，$$Q$$不变。

所以一种代替的公式是

$$\begin{array}{c}{Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+} \\ {\left(A(s, a ; \theta, \alpha)-\max _{a^{\prime} \in|\mathcal{A}|} A\left(s, a^{\prime} ; \theta, \alpha\right)\right)}\end{array}$$

这样设计的原因是

![](../../.gitbook/assets/image-47.png)

