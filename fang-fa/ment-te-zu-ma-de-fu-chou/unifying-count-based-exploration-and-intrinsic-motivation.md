# Unifying Count-Based Exploration and Intrinsic Motivation

## 介绍

> [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/pdf/1606.01868.pdf)

我们考虑一个代理对其环境的不确定性，以及将这种不确定性分散到不同状态的问题。具体来说，我们关注非表格强化学习中的探索问题。从内在动机文献中得到启发，我们使用密度模型来测量不确定性，并提出了一种从任意密度模型中导出伪计数的新算法。这种技术使我们能够将基于计数的勘探算法推广到非表格情况。我们将我们的想法应用于雅达利2600游戏，从原始像素中提供合理的伪计数。我们将这些伪计数转化为探索奖励，并在许多游戏中获得显著改善的探索，包括臭名昭著的困难MONTEZUMA 'SREVENGE。

马尔可夫决策过程\( MDP\)的探索算法通常关注于降低代理对环境回报和转换函数的不确定性。在白板中，这种不确定性可以使用从切尔诺夫边界导出的置信区间来量化，或者从环境参数的后验值来推断。事实上，置信区间和后验收缩都是状态行为访问计数 $$( x,a )$$ 的平方根倒数，这使得这个量成为大多数理论探索结果的基础。

基于计数的探索方法直接使用访问计数来指导代理人的行为以减少不确定性。例如Model-based Interval Estimation with Exploration Bonuses使用如下的增强版Bellman方程：

$$
V(x)=\max _{a \in \mathcal{A}}\left[\hat{R}(x, a)+\gamma \mathbb{E}_{\hat{P}}\left[V\left(x^{\prime}\right)\right]+\beta N(x, a)^{-1 / 2}\right]
$$

该奖励考虑了转移和奖励函数的不确定性，并且能够对代理的次优性进行有限时间限制。

内在动机旨在为探索提供定性指导，这个指导可以概括为“探索让你惊讶的事情”，一种典型的方法基于预测误差或者学习进度。如果 $$e_n(A)$$ 是代理在某一事件A发生时所犯的错误，并且在观察到一个新的信息后，再得到错误 $$e_{n+1}(A)$$ ，则学习进度为：

$$
e_{n}(A)-e_{n+1}(A)
$$

在本文中，我们提供了正式的证据，证明内在动机和基于计数的探索是同一枚硬币的两面。我们的贡献是提出一个新的量化机制，即假计数，它将信息增益作为学习进度与基于计数的探索联系起来。

## 算法

### Notation

密度模型

$$
\rho_{n}(x) :=\rho\left(x ; x_{1 : n}\right)
$$

即 $$X_{n+1}=x \text { given } X_{1} \ldots X_{n}=x_{1 : n}$$ 的概率

经验分布，其中 $$N_{n}(x) :=N\left(x, x_{1 : n}\right)$$ 是经验计数函数

$$
\mu_{n}(x) :=\mu\left(x ; x_{1 : n}\right) :=\frac{N_{n}(x)}{n}
$$

在我们的设置中，密度模型假定状态独立（但不一定相同）分布的任何模型; 因此，密度模型是一种特殊的生成模型。

### From Densities to Counts

设recoding probability

$$
\rho_{n}^{\prime}(x)=\operatorname{Pr}_{\rho}\left(X_{n+2}=x | X_{1} \ldots X_{n}=x_{1 : n}, X_{n+1}=x\right)
$$

我们现在假设两个未知数:伪计数函数 $$\hat{N}_{n}(x)$$ 和伪总计数函数 $$\hat{n}$$ ，并服从以下约束

$$
\rho_{n}(x)=\frac{\hat{N}_{n}(x)}{\hat{n}} \quad \rho_{n}^{\prime}(x)=\frac{\hat{N}_{n}(x)+1}{\hat{n}+1}
$$

注意 $$\hat{N}_{n}(x)=0(\text { with } \hat{n}=\infty) \text { when } \rho_{n}(x)=\rho_{n}^{\prime}(x)=0$$ ，且当 $$\rho_{n}(x)<\rho_{n}^{\prime}(x)=1$$ 时不一致。

换句话说:我们要求，在观察到 $$x$$ 的一个实例后，密度模型对同一 $$x$$ 的预测的增加应对应于伪计数的单位增加。伪计数本身是从求解线性系统中推导出来的:

$$
\hat{N}_{n}(x)=\frac{\rho_{n}(x)\left(1-\rho_{n}^{\prime}(x)\right)}{\rho_{n}^{\prime}(x)-\rho_{n}(x)}=\hat{n} \rho_{n}(x)
$$















