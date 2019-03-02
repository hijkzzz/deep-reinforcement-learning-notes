# Bootstrapped Deep Q-Network

## 介绍

> [Deep Exploration via Bootstrapped DQN](https://arxiv.org/pdf/1602.04621.pdf)

有效的探索仍然是强化学习（RL）的主要挑战。 常见的探索策略，例如 $$\epsilon$$ -greedy，不进行时间延长\(或深度\)的探索; 这可能导致数据需求呈指数级增长。 然而，大多数算法具有高效率的RL算法在复杂的环境中不易计算。 随机值函数提供了一种有前景的有效探索方法，但现有算法与非线性参数化值函数不相容。 作为解决此类问题的第一步，我们开发了bootstrapped DQN。 我们证明了bootstrapped DQN可以将深度探索与深度神经网络相结合，以比任何探索策略更快地学习。 在街机学习环境中，bootstrapped DQN大大提高了大多数游戏的学习速度和累积性能。

## 算法

### Bootstrapped Network

![](../../.gitbook/assets/image%20%2832%29.png)

Bootstrapped DQN用bootstrap修改DQN以近似Q值的分布。 在每个周期开始时，Bootstrapped DQN从其近似后验中采样单个Q值函数。 然后，代理遵循在整个事件期间对该样本最优的策略。这是Thompson sampling启发式算法对RL的自然适应，允许时间上的扩展\(或深度\)探索。

如图1\(a\)所示，我们通过并行构建K个Q值函数的估计有效地实现了该算法。重要的是，这些值函数中的每一个函数头 $$Q_k(s,a;θ)$$ 都有各自的目标网络 $$Q_k(s,a;θ^-)$$ 。这意味着每个 $$Q_1,..,Q_K$$ 通过TD估计提供了值不确定性的扩展\(和一致\)估计。为了跟踪哪些数据属于哪个引导头，我们存储了标志 $$w_1,..,w_K∈(0，1)$$ 表示哪些头与哪些数据相关。我们通过在随机选择 $$k∈(1，..，K)$$ 并且在该周期的持续时间内跟随 $$Q_k$$ 来近似一个bootstrap样本。

## 伪代码

![](../../.gitbook/assets/image%20%2831%29.png)

