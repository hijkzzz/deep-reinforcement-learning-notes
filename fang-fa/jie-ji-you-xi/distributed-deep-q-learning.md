# Distributed Deep Q-Learning

## 介绍

> [Massively Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1507.04296.pdf)

我们提出了第一个用于深层强化学习的大规模分布式架构。 该体系结构使用了四个主要组件：产生新行为的并行的actor; 通过存储经验训练的并行leaner; 分布式神经网络来表示价值函数或行为策略; 一个分布式的经验池。 使用我们的架构来实施Deep Q-Network算法， 并应用于Arcade Learning Environment的49款Atari2600游戏，使用相同的超参数，在49场比赛中有41场超过了非分布式DQN，并且还减少了在大多数比赛中按照一定数量的顺序实现这些结果所需的时间。

## 算法

![](../../.gitbook/assets/image%20%2877%29.png)

## 伪代码

![](../../.gitbook/assets/image%20%2866%29.png)

