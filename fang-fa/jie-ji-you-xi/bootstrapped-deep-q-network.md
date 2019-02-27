# Bootstrapped Deep Q-Network

## 介绍

> [Deep Exploration via Bootstrapped DQN](https://arxiv.org/pdf/1602.04621.pdf)

有效的探索仍然是强化学习（RL）的主要挑战。 常见的探索策略，例如 $$\epsilon$$ -greedy，不进行时间延长\(或深度\)的探索; 这可能导致数据需求呈指数级增长。 然而，大多数算法具有高效率的RL算法在复杂的环境中不易计算。 随机值函数提供了一种有前景的有效探索方法，但现有算法与非线性参数化值函数不相容。 作为解决此类问题的第一步，我们开发了bootstrapped DQN。 我们证明了bootstrapped DQN可以将深度探索与深度神经网络相结合，以比任何探索策略更快地学习。 在街机学习环境中，bootstrapped DQN大大提高了大多数游戏的学习速度和累积性能。

## 算法



