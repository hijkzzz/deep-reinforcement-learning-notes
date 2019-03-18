# Off-Policy Actor-Critic

## 介绍

> [Off-Policy Actor-Critic](https://arxiv.org/pdf/1205.4839.pdf)

本文提出了第一个off-policy强化学习的actor-critic算法。我们的算法是在线的和增量的，并且每个时间步长的复杂度与学习的权重数量成线性比例。前期的actor-critic算法的局限于on-policy环境，没有利用off-policy的时间差分梯度的优势。off-policy技术，如Greedy-GQ，能够在跟踪和从另一个\(行为\)策略获取数据的同时学习目标策略。对于许多问题，无论如何，actor-critic方法更实用，因为它们清晰的代表了策略；因此，该策略可以是是随机的，并且可利用大的动作空间。本文阐述了如何将off-policy学习的通用性和学习潜力与actor-critic方法在行动选择中的灵活性相结合，这种灵活性是由actor-critic的方法决定的。我们导出了一个包含资格迹的线性时间和空间复杂度的算法，证明了在类似于以前的off-policy算法的假设下的收敛性，并且在标准强化学习基准问题上，实验显示了比现有算法更好或的性能。

最着名的off-policy方法是Q-learning。 然而，虽然Q-Learnings保证收敛到最优的非线性（非近似）情况，但在使用线性函数逼近时可能会有所不同（Baird，1995）。最小二乘法，如LSTD和LSPI 可以使用off-policy和线性函数，但是计算成本昂贵；它们的复杂度与特征和权重的数量成比例。 最近，这些问题已经被新的梯度-TD（时间差）方法所解决，例如Greedy-GQ，它们具有线性复杂性和收敛性， 具有函数逼近的off-policy学习下。

所有的动作值方法，包括梯度TD方法，如Greedy-GQ，都有三个重要的局限性。首先，他们的目标政策是确定性的，而许多问题都有随机的最优政策，其次，对于较大的动作空间来说，找到关于动作值函数的贪婪动作变得有问题。最后，动作值函数的小变化会导致策略的大变化，这给收敛性证明和一些实时应用带来困难。避免行动价值方法局限性的标准方法是使用策略梯度算法，例如actor-critic。

## 算法

### Off-PAC

价值函数定义

$$V^{\pi, \gamma}(s)=\mathrm{E}\left[r_{t+1}+\ldots+r_{t+T} | s_{t}=s\right] \forall s \in \mathcal{S} $$ 

动作值函数定义

$$\begin{array}{l}{Q^{\pi, \gamma}(s, a)=} \\ {\sum_{s^{\prime} \in S} P\left(s^{\prime} | s, a\right)\left[\mathcal{R}\left(s, a, s^{\prime}\right)+\gamma\left(s^{\prime}\right) V^{\pi, \gamma}\left(s^{\prime}\right)\right]}\end{array} $$

我们的学习目标是找出一个能最大化下面函数的策略 $$u $$ 

$$J_{\gamma}(\mathbf{u})=\sum_{s \in \mathcal{S}} d^{b}(s) V^{\pi_{\mathbf{u}}, \gamma}(s)$$ 

其中 $$d^{b}(s)=\lim _{t \rightarrow \infty} P\left(s_{t}=s | s_{0}, b\right) $$ ，即在动作 $$b$$ 和初始状态 $$s_0$$ 下的极限概率分布。用 $$d^{b} $$加权的原因是，在off-policy中，数据是从行为策略的分布获得。

#### The Critic: Policy Evaluation

用线性函数近似价值函数： $$\hat{V}(s)=\mathbf{v}^{\mathbf{T}} \mathbf{x}_{s}$$ 

λ-weighted mean-squared projected Bellman error

$$\operatorname{MSPBE}(\mathbf{v})=\left\|\hat{V}-\Pi T_{\pi}^{\lambda, \gamma} \hat{V}\right\|_{D}^{2} $$ 

其中$$\hat{V}=X \mathbf{v}$$ ， $$X$$ 的每一行都是一个样本， $$λ$$ 是资格迹权重参数， $$D$$ 是一个矩阵（对角元素为$$d^{b} $$）， $$Π $$ 是一个投影操作， $$T_{\pi}^{\lambda, \gamma}$$ 是λ-weighted Bellman operator\(对于终止概率为 $$γ$$ 的策略 $$π$$ \)。在线性情况下， $$\Pi=X\left(X^{\top} D X\right)^{-1} X^{\top} D$$ 。

请参考 $$GTD(λ)$$ 算法

#### Off-policy Policy-gradient Theorem

策略梯度为

$$\mathbf{u}_{t+1}-\mathbf{u}_{t} \approx \alpha_{u, t} \nabla_{\mathbf{u}} J_{\gamma}\left(\mathbf{u}_{t}\right)$$ 

展开得到

$$\begin{aligned} \nabla_{\mathbf{u}} J_{\gamma}(\mathbf{u})=& \nabla_{\mathbf{u}}\left[\sum_{s \in \mathcal{S}} d^{b}(s) \sum_{a \in \mathcal{A}} \pi(a | s) Q^{\pi, \gamma}(s, a)\right] \\=& \sum_{s \in \mathcal{S}} d^{b}(s) \sum_{a \in \mathcal{A}}\left[\nabla_{\mathbf{u}} \pi(a | s) Q^{\pi, \gamma}(s, a)\right.\\ &+\pi(a | s) \nabla_{\mathbf{u}} Q^{\pi, \gamma}(s, a) ] \end{aligned}$$ 

$$\nabla_{\mathbf{u}} Q^{\pi, \gamma}(s, a)$$ 很难估计，所以忽略掉，得到近似梯度

$$\nabla_{\mathbf{u}} J_{\gamma}(\mathbf{u}) \approx \mathbf{g}(\mathbf{u})=\sum_{s \in \mathcal{S}} d^{b}(s) \sum_{a \in \mathcal{A}} \nabla_{\mathbf{u}} \pi(a | s) Q^{\pi, \gamma}(s, a)$$ 

下面的定理证明了这个近似（证明见原文附录）

![](../../.gitbook/assets/image%20%286%29.png)

![](../../.gitbook/assets/image%20%2825%29.png)

#### The Actor: Incremental Update Algorithm with Eligibility Traces

我们现在使用行为策略中的观察样本推导出增量更新算法，首先写出近似策略梯度的期望形式

$$
\begin{aligned} \mathbf{g}(\mathbf{u}) &=\mathrm{E}\left[\sum_{a \in \mathcal{A}} \nabla_{\mathbf{u}} \pi(a | s) Q^{\pi, \gamma}(s, a) | s \sim d^{b}\right] \\ &=\mathrm{E}\left[\sum_{a \in \mathcal{A}} b(a | s) \frac{\pi(a | s)}{b(a | s)} \frac{\nabla_{\mathbf{u}} \pi(a | s)}{\pi(a | s)} Q^{\pi, \gamma}(s, a) | s \sim d^{b}\right] \\ &=\mathrm{E}\left[\rho(s, a) \psi(s, a) Q^{\pi, \gamma}(s, a) | s \sim d^{b}, a \sim b(\cdot | s)\right] \\ &=\mathrm{E}_{b}\left[\rho\left(s_{t}, a_{t}\right) \psi\left(s_{t}, a_{t}\right) Q^{\pi, \gamma}\left(s_{t}, a_{t}\right)\right] \end{aligned}
$$

其中 $$\rho(s, a)=\frac{\pi(a | s)}{b(a | s)}, \psi(s, a)=\frac{\nabla_{\mathbf{u}} \pi(a | s)}{\pi(a | s)}$$ ，这里引入了新的符号 $$\mathrm{E}_{b}[\cdot]$$期望 来隐含地表示条件，所有随机变量\(按时间步长索引\)都是从行为策略下它们的极限平稳分布中提取出来的。然后减去一个baseline减小方差：

$$\mathbf{g}(\mathbf{u})=\mathrm{E}_{b}\left[\rho\left(s_{t}, a_{t}\right) \psi\left(s_{t}, a_{t}\right)\left(Q^{\pi, \gamma}\left(s_{t}, a_{t}\right)-\hat{V}\left(s_{t}\right)\right)\right]$$ 

下一步是替换动作值

$$\mathbf{g}(\mathbf{u}) \approx \widehat{\mathbf{g}(\mathbf{u})}=\mathrm{E}_{b}\left[\rho\left(s_{t}, a_{t}\right) \psi\left(s_{t}, a_{t}\right)\left(R_{t}^{\lambda}-\hat{V}\left(s_{t}\right)\right)\right]$$ 

其中off-policy λ-return定义的 $$R_{t}^{\lambda}$$ 为：

 $$\begin{aligned} R_{t}^{\lambda}=& r_{t+1}+(1-\lambda) \gamma\left(s_{t+1}\right) \hat{V}\left(s_{t+1}\right) \\ &+\lambda \gamma\left(s_{t+1}\right) \rho\left(s_{t+1}, a_{t+1}\right) R_{t+1}^{\lambda} \end{aligned}$$

这就是前向视角的Off-PAC

### Convergence Proofs

请参考原文附录

## 伪代码 

为了实现前面描述的算法，这里转换为后向视角，关键的一步是

$$\mathrm{E}_{b}\left[\rho\left(s_{t}, a_{t}\right) \psi\left(s_{t}, a_{t}\right)\left(R_{t}^{\lambda}-\hat{V}\left(s_{t}\right)\right)\right]=\mathrm{E}_{b}\left[\delta_{t} \mathrm{e}_{t}\right]$$ 

其中 $$\delta_{t}=r_{t+1}+\gamma\left(s_{t+1}\right) \hat{V}\left(s_{t+1}\right)-\hat{V}\left(s_{t}\right)$$ 是时间差分误差， $$\mathbf{e}_{t} \in \mathbb{R}^{N_{\mathbf{u}}}$$ 是资格迹，由下式更新

$$\mathbf{e}_{t}=\rho\left(s_{t}, a_{t}\right)\left(\psi\left(s_{t}, a_{t}\right)+\lambda \mathbf{e}_{t-1}\right)$$ 

然后我们可以得到： $$\mathbf{u}_{t+1}-\mathbf{u}_{t}=\alpha_{u, t} \delta_{t} \mathbf{e}_{t}$$ 

证明见原文附录或者参考资格迹的相关教程

![](../../.gitbook/assets/image%20%2857%29.png)



