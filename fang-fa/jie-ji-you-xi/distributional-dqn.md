# Distributional DQN

## 介绍

> [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf)

在本文中，我们论证了价值分布（强化学习代理接收的随机回报的分布）的基本重要性。 这与强化学习的通用方法不同，后者为这种回归或价值的期望建模。尽管已经有一套研究价值分布的成熟的体系，但到目前为止，它一直被用于特定目的，例如实施风险敏感的的行为。 我们从策略评估和控制设置的理论结果开始，揭示了后者的显着分布不稳定性。 然后从分布的角度设计了一种新的算法，将Bellman等式应用于近似分布的学习。我们使用街机学习环境中的游戏集来评估我们的算法。我们获得了最新的结果和轶事证据，证明了价值分布在接近配偶强化学习中的重要性。最后，我们从理论和经验两方面论证了数值分布在近似环境下对学习的影响。

## 算法

贝尔曼等式

$$
\begin{array}{l}{Q^{\pi}(x, a) :=\mathbb{E} Z^{\pi}(x, a)=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(x_{t}, a_{t}\right)\right]} \\ {x_{t} \sim P(\cdot | x_{t-1}, a_{t-1}), a_{t} \sim \pi(\cdot | x_{t}), x_{0}=x, a_{0}=a}\end{array}
$$

其中 $$Z$$ 是一个分布而不是期望值，为了最大化 $$Q$$ 值，通常使用下面的方法

$$
Q^{*}(x, a)=\mathbb{E} R(x, a)+\gamma \mathbb{E}_{P} \max _{a^{\prime} \in \mathcal{A}} Q^{*}\left(x^{\prime}, a^{\prime}\right)
$$

贝尔曼/最优算子定义

![](../../.gitbook/assets/image%20%2824%29.png)

### 分布贝尔曼算子

仅对算法感兴趣的读者可以选择跳过这一节

#### Distributional Equations

$$(\Omega, \mathcal{F}, \mathrm{Pr})$$是概率空间

 $$\|\mathbf{u}\|_{p}$$ 表示 $$\mathbf{u} \in \mathbb{R}^{\mathcal{X}}$$ 的 $$L_{p}$$ 范数

 当$$U : \Omega \rightarrow \mathbb{R}^{\mathcal{X}}\left(\text { or } \mathbb{R}^{\mathcal{X} \times \mathcal{A}}\right)$$ ，有$$\|U\|_{p} :=\left[\mathbb{E}\left[\|U(\omega)\|_{p}^{p}\right]\right]^{1 / p}$$，并且 $$\|U\|_{\infty}=\operatorname{ess} \sup \|U(\omega)\|_{\infty}$$ 

累积分布函数 $$F_{U}(y) :=\operatorname{Pr}\{U \leq y\}$$ ，逆函数 $$F_{U}^{-1}(q) :=\inf \left\{y : F_{U}(y) \geq q\right\}$$ 

$$U \overset{D}{:=}V$$ 表示两个随机变量的分布相同

#### The Wasserstein Metric

定义两个累计分布函数的Wasserstein尺度

$$d_{p}(F, G) :=\inf _{U, V}\|U-V\|_{p}$$ 

其中下界是关于累积分布的所有的随机变量 $$(U, V)$$ ，通过均匀随机变量 $$\mathcal{U}\ in\ [0,1]$$ 的逆累计分布函数获得此下限

$$d_{p}(F, G)=\left\|F^{-1}(\mathcal{U})-G^{-1}(\mathcal{U})\right\|_{p}$$ 

 $$p<\infty$$ 时，可写为

$$d_{p}(F, G)=\left(\int_{0}^{1}\left|F^{-1}(u)-G^{-1}(u)\right|^{p} d u\right)^{1 / p}$$ 

对于两个随机变量定义

$$d_{p}(U, V)=\inf _{U, V}\|U-V\|_{p}$$ 

Wasserstein尺度有以下性质

$$
\begin{aligned} d_{p}(a U, a V) \leq|a| d_{p}(U, V) &(\mathrm{Pl}) \\ d_{p}(A+U, A+V) \leq d_{p}(U, V) &(\mathrm{P} 2) \\ d_{p}(A U, A V) \leq\|A\|_{p} d_{p}(U, V) &(\mathrm{P} 3) \end{aligned}
$$

引理1、2

![Wasserstein&#x5C3A;&#x5EA6;&#x5F15;&#x7406;](../../.gitbook/assets/image%20%2856%29.png)

#### Policy Evaluation

定义转移算子 $$P^{\pi} : \mathcal{Z} \rightarrow \mathcal{Z}$$ 

$$\begin{aligned} P^{\pi} Z(x, a) & :=Z\left(X^{\prime}, A^{\prime}\right) \\ X^{\prime} & \sim P(\cdot | x, a), A^{\prime} \sim \pi(\cdot | X^{\prime}) \end{aligned}$$ 

我们定义了分布Bellman算子 $$\mathcal{T}^{\pi} : \mathcal{Z} \rightarrow \mathcal{Z}$$ 

$$\mathcal{T}^{\pi} Z(x, a) : \stackrel{D}{=} R(x, a)+\gamma P^{\pi} Z(x, a)$$ 

随机性的三个来源定义了复合分布，我们通常假设这三个量是独立的

![](../../.gitbook/assets/image%20%2836%29.png)

引理3

![](../../.gitbook/assets/image%20%2852%29.png)

这说明 $$Z_{k+1} :=T^{\pi} Z_{k}$$ 在尺度 $$\overline{d}_{p}$$ 下收缩，然而可能不适用于其他的尺度。

#### Control

最优价值分布和贪心策略定义

![](../../.gitbook/assets/image%20%2840%29.png)

![](../../.gitbook/assets/image%20%2851%29.png)

最优贝尔曼算子等价于

![](../../.gitbook/assets/image%20%2833%29.png)

引理4

![](../../.gitbook/assets/image%20%2837%29.png)

因此，我们希望 $$Z_k$$ 能够快速收敛到一个fixed point，然而这可能很慢或者不确定。实际上，我们可以希望pointwise convergence，即收敛到 nonstationary optimal value distributions。

![](../../.gitbook/assets/image%20%285%29.png)

将定理1与引理4相比较揭示了分布框架和通常的期望设置之间的显著差异。虽然 $$Z_{k}$$ 的平均值迅速指数收敛到 $$Q^*$$ ，但它的分布不需要表现得那么好！为了强调这种差异，我们提供了一些负面结果，如下：

![](../../.gitbook/assets/image%20%2815%29.png)

![](../../.gitbook/assets/image%20%2811%29.png)

![](../../.gitbook/assets/image%20%2858%29.png)

### 近似分布学习

在本节中，我们提出了一种基于分布Bellman最优性算子的算法。 特别是，这将需要选择近似分布。 虽然已经有人考虑过高斯分布，但据我们所知，我们是第一个使用丰富的参数分布的。

#### Parametric Distribution

我们将使用离散分布来模拟价值分布，离散分布具有高度表达和计算友好的优点：

$$
Z_{\theta}(x, a)=z_{i} \quad \text { w.p. } p_{i}(x, a) :=\frac{e^{\theta_{i}(x, a)}}{\sum_{j} e^{\theta_{j}(x, a)}}
$$

其中 $$\left\{z_{i}=V_{\mathrm{MIN}}+i \triangle z : 0 \leq\right.i<N \},\Delta z :=\frac{V_{\mathrm{mAx}}-V_{\mathrm{MIN}}}{N-1}$$ 

#### Projected Bellman Update

使用离散的分布造成了一个问题：贝尔曼更新 $$\mathcal{T} Z_{\theta}$$ 和我们的 $$Z_{\theta}$$ 几乎总是disjoint support。从前面一节（分布贝尔曼算子）的分析来看，将 $$\mathcal{T} Z_{\theta} \text { and } Z_{\theta}$$的 Wasserstein metric（视为损失）最小化之间似乎很自然。但是因为Wasserstein loss不适合采样的转移（参考原文附录）。

相反，我们将采样的贝尔曼更新$$\mathcal{T} Z_{\theta}$$ 投影到 $$Z_{\theta}$$ support，见

$$
\left(\Phi \hat{\mathcal{T}} Z_{\theta}(x, a)\right)_{i}=\sum_{j=0}^{N-1}\left[1-\frac{\left|\left[\hat{\mathcal{T}} z_{j}\right]_{V_{\mathrm{wuv}}}^{V_{\max }}-z_{i}\right|}{\Delta z}\right]_{0}^{1} p_{j}\left(x^{\prime}, \pi\left(x^{\prime}\right)\right)
$$

![](../../.gitbook/assets/image%20%2849%29.png)

