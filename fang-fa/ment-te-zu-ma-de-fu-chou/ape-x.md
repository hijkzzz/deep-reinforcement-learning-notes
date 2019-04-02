# Ape-X

## 介绍

> [DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://arxiv.org/abs/1803.00933)

我们提出了一种用于大规模深度强化学习的分布式架构，使代理能够从数量级上有效地学习比以前更多的数据。 该算法将行为与学习分离：通过根据共享神经网络选择动作，actor与他们自己的环境实例交互，并在共享体验重放记忆中累积所得到的体验; 学习者重放经验样本并更新神经网络。 该体系结构依赖于优先级经验重放，仅针对参与者生成的最重要数据。 我们的架构大大改善了Arcade学习环境的最新技术水平，在壁钟培训时间的一小部分内实现了更好的最终性能。

## 方法

![](../../.gitbook/assets/image%20%2824%29.png)

原则上，actor和learner可以分布在多个worker中。在我们的实验中，数百名参与者在处理器上运行以生成数据，一名学习者在处理器上运行以采样最有用的样本\(图1 \)。actor和learner的伪代码在算法1和2中显示。更新后的网络参数会定期从学习者处传达给参与者。

![](../../.gitbook/assets/image%20%2860%29.png)

与共享梯度相比，共享经验具有一定的优势。 低延迟通信并不像分布式SGD那么重要，因为经验数据比梯度过时更慢，只要学习算法对off-policy数据具有鲁棒性。 在整个系统中，我们通过将所有通信与集中式重放进行批处理来利用这一点，以一定的延迟为代价提高效率和吞吐量。 通过这种方法，参与者和学习者甚至可以在不限制性能的情况下在不同的数据中心运行。

最后，通过学习off-policy，我们可以进一步利用Ape-X整合来自许多分布式参与者的数据的能力，为不同的参与者提供不同的策略，扩大他们共同遇到的体验的多样性。正如我们将在结果中看到的，这足以在困难的勘探问题上取得进展。

### APE-X DQN

$$
l_{t}(\boldsymbol{\theta})=\frac{1}{2}\left(G_{t}-q\left(S_{t}, A_{t}, \boldsymbol{\theta}\right)\right)^{2}
$$

![](../../.gitbook/assets/image%20%2877%29.png)

### APE-X DPG

$$
l_{t}(\psi)=\frac{1}{2}\left(G_{t}-q\left(S_{t}, A_{t}, \psi\right)\right)^{2}
$$

![](../../.gitbook/assets/image%20%2843%29.png)

## 实验

![](../../.gitbook/assets/image%20%2831%29.png)

![](../../.gitbook/assets/image%20%2849%29.png)

![](../../.gitbook/assets/image%20%2836%29.png)

![](../../.gitbook/assets/image%20%284%29.png)







