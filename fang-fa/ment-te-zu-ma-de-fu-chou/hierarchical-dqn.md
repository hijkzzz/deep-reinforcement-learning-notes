# Hierarchical-DQN

## 介绍

> [Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/abs/1604.06057)

稀疏反馈环境下的学习目标导向行为是增强学习算法面临的主要挑战。主要的困难是由于探索不够，导致代理无法学习健壮的值函数。具有内在动机的代理可以为了自身的利益而探索新的行为，而不是直接解决问题。这样的内在行为最终可以帮助代理解决环境设置的任务。提出了一种基于层次结构的dqn \(h-DQN\)框架，该框架集成了不同时间尺度下的层次值函数，具有内在的深层强化学习动机。高层值函数学习策略而不是内在目标，底层函数学习满足给定目标的原子操作。h-DQN允许灵活的目标规范，例如实体和关系上的函数。这为复杂环境中的探索提供了有效的空间。我们证明了我们的方法在两个非常稀疏、延迟反馈的问题上的优势:\(1\)一个复杂的离散随机决策过程，和\(2\)经典的ATARI游戏Montezuma的复仇。

## 算法

![](../../.gitbook/assets/image%20%2863%29.png)

![](../../.gitbook/assets/image%20%28121%29.png)

### Agents

我们使用temporal abstraction of options来定义每个目标的策略。 这些代理同时学习这些option策略，同时学习最佳的目标序列。 为了使每个 $$π_g$$ 学习目标 $$g$$ ，代理人还有一个批评者，根据代理人是否能够实现其目标，提供内在的激励信号。

### Temporal Abstractions

代理人使用由controlle和meta-controller组成的两阶段层次结构。 元控制器接收状态 $$s_{t}$$ 选择一个目标 $$g_{\iota} \in \mathcal{G}$$ ，其中 $$ \mathcal{G}$$ 表示有可能的当前目标的集合。目标将在接下来的几个步骤中保持不变，直到达到目标或达到最终状态。内部批评家负责评估目标是否已经达到，并提供适当的奖励。同样，元控制器的目标是优化累积的外在奖励。

### Deep Reinforcement Learning with Temporal Abstractions

用深度强化学习来学习这些策略

controller

![](../../.gitbook/assets/image%20%2880%29.png)

meta-controller

![](../../.gitbook/assets/image%20%28109%29.png)

然后用类似TD的算法学习

![](../../.gitbook/assets/image%20%2887%29.png)

### 伪代码

![](../../.gitbook/assets/image%20%2897%29.png)

## 实验

![](../../.gitbook/assets/image%20%2857%29.png)

代理需要内在的动机去探索场景中有意义的部分，然后才能了解为自己获取钥匙的好处。受developmental psychology literature和面向对象MDP的启发，我们用场景中的entities 或者 objects来参数化环境中的目标。尽管近年来直接从图像或运动数据中获取目标的研究取得了一定的进展，但在计算机视觉中，对视觉场景中目标的无监督检测一直是一个悬而未决的问题。在这项工作中，我们构建了一个定制的对象检测器，它提供了可信的对象候选者。

内部批评者在 $$\left\langle\text {entity}_{1}, \text { relation, entity }_{2}\right\rangle$$ 的空间中定义，其中relation是实体上的的配置。在我们的实验中，代理可以自由选择任何entity2。 例如，如果代理人到达诸如门之类的其他实体，则认为代理人已经完成了目标（并且接收了回报）。

![](../../.gitbook/assets/image%20%2873%29.png)



