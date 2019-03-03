# Actor-Critic with Experience Replay

## 介绍

> [SAMPLE EFFICIENT ACTOR-CRITIC WITH EXPERIENCE REPLAY](https://arxiv.org/pdf/1611.01224.pdf)

本文提出了一个Actor-Critic深度强化学习代理，具有经验重放，稳定，样本效率高的特点，并且在具有挑战性的环境中表现非常好，包括57个离散的Atari游戏和几个连续的控制问题。 为此，本文介绍了几种创新，包括具有偏差校正的截断重要性采样，随机推导网络架构和新的信任域策略优化方法。

## 算法

### DISCRETE ACTOR-CRITIC WITH EXPERIENCE REPLAY

经验回放的off-policy学习似乎是提高actor-critic样本效率的一个显然的策略。然而，众所周知，控制off-policy估计的方差和稳定性是非常困难的。重要性采样是最流行的off-policy学习方法之一。假设有用行为策略 $$μ$$ 采集的样本 $$\left\{x_{0}, a_{0}, r_{0}, \mu(\cdot | x_{0}), \cdots, x_{k}, a_{k}, \tau_{k}, \mu(\cdot | x_{k})\right\}$$ ，重要性加权策略梯度由下式给出：

$$
\widehat{g}^{\mathrm{imp}}=\left(\prod_{t=0}^{k} \rho_{t}\right) \sum_{t=0}^{k}\left(\sum_{i=0}^{k} \gamma^{i} r_{t+i}\right) \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | x_{t}\right)
$$

其中 $$\rho_{t}=\frac{\pi\left(a_{t} | x_{t}\right)}{\mu\left(a_{t} | x_{t}\right)}$$ 即重要性权重。这个估计量是无偏的，但是它有很高的方差，因为它涉及许多潜在的无限的重要性权重的乘积。为了防止重要性权重的累成的结果爆炸，可截断此权重，整个轨迹上的截断重要性采样，尽管在方差上是有界的，但可能会有明显的偏差。

最近[Degris等人](https://arxiv.org/pdf/1205.4839.pdf)利用极限分布上的边际值函数对该问题进行了研究，得到了以下近似梯度：

$$
g^{\mathrm{marg}}=\mathbb{E}_{x_{t} \sim \beta, a_{t} \sim \mu}\left[\rho_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | x_{t}\right) Q^{\pi}\left(x_{t}, a_{t}\right)\right]
$$

其中 $$\mathbb{E}_{\boldsymbol{x}_{\boldsymbol{t}}} \sim \beta, a_{t} \sim \mu[\cdot]$$ 是关于极限分布 $$\beta(x)=\lim _{t \rightarrow \infty} P\left(x_{t}=x | x_{0}, \mu\right)$$ 和行为策略 $$μ$$ 的期望。为了保持符号简洁，我们用 $$\mathbb{E}_{x_{t} a_{t}}[\cdot]$$ 替换。

关于等式\( 4 \)，必须强调两个重要事实：首先，请注意，它依赖于 $$Q^π$$ ，而不是 $$Q^μ$$ ，因此我们必须能够估计$$Q^π$$ 。第二，我们不再有重要权重的乘积，而是只需要估计边缘重要性权重 $$ρ_t$$ 。重要性采样在这个较低维度空间中\(相对于轨迹而言，在边缘上\)预计会显示出较低的方差。

[Degris等人](https://arxiv.org/pdf/1205.4839.pdf)使用 $$λ-return: R_{t}^{\lambda}=r_{t}+(1-\lambda) \gamma V\left(x_{t+1}\right)+\lambda \gamma \rho_{t+1} R_{t+1}^{\lambda}$$ （即资格迹方法）估计 $$Q^π$$。这个估计要求我们知道如何提前选择 $$λ$$ 来权衡偏差和方差。此外，当使用小值 $$λ$$的来减少方差时，偶尔的大重要性权重仍然会导致不稳定性。

在下面的小节中，我们采用了Munos等人的 $$Retrace$$ [算法](http://arxiv.org/abs/1606.02647)估计$$Q^π$$。随后，我们提出了一种重要性权重截断技术，以提高[Degris等人](https://arxiv.org/pdf/1205.4839.pdf)的off-policy actor critic的稳定性。并为策略优化引入计算有效的信任区域方案。

#### **MULTI-STEP ESTIMATION OF THE STATE-ACTION VALUE FUNCTION**

给定在行为策略 $$μ$$ 下生成的轨迹，可以如下递归地表示 $$Retrace$$ 估计（为了便于演示，我们只考虑 $$λ= 1$$ ） :

$$
Q^{\mathrm{ret}}\left(x_{t}, a_{t}\right)=r_{t}+\gamma \overline{\rho}_{t+1}\left[Q^{\mathrm{ret}}\left(x_{t+1}, a_{t+1}\right)-Q\left(x_{t+1}, a_{t+1}\right)\right]+\gamma V\left(x_{t+1}\right)
$$

$$\overline{\rho}_{t}$$ 是截断的重要性采样系数，即 $$\overline{\rho}_{t}=\min \left\{c, \rho_{t}\right\} \text { with } \rho_{t}=\frac{\pi\left(a_{t} | x_{t}\right)}{\mu\left(a_{t} | x_{t}\right)}$$ ， $$Q$$ 是 $$Q_π$$ 的当前值估计，并且 $$V(x)=\mathbb{E}_{a \sim \pi} Q(x, a)$$ 

递归 $$Retrace$$ 方程需要估计 $$Q$$。为了计算它，在离散动作空间中，我们采用具有“双头”的卷积神经网络，其输出估计 $$Q_{θv}(x_t, a_t)$$ ，以及策略 $$\pi_θ(a_t | x_t)$$ 。

$$Retrace$$ 是一种off-policy、基于回报的[算法](http://arxiv.org/abs/1606.02647)，并具有低方差，被证明\(在表格中\)收敛于任何行为策略的目标策略的值函数。

为了近似策略梯度 $$g^{\mathrm{marg}}$$ ， $$ACER$$ 使用 $$Q^{\mathrm{ret}}$$ 估计 $$Q^{\pi}$$ 。由于Retrace使用多步回报，它可以显著减少策略梯度估计中的偏差。

为了学习Critic $$Q_{\theta_{v}}\left(x_{t}, a_{t}\right)$$ ，我们再次使用 $$Q^{\mathrm{ret}}\left(x_{t}, a_{t}\right)$$ 作为均方误差损失的目标函数，并使用以下标准梯度更新其参数 $$θ_v$$ 

$$
\left(Q^{\mathrm{ret}}\left(x_{t}, a_{t}\right)-Q_{\theta_{v}}\left(x_{t}, a_{t}\right)\right) \nabla_{\theta_{v}} Q_{\theta_{v}}\left(x_{t}, a_{t}\right) )
$$

因为Retrace是基于回报的，它还可以更快地学习critic。 因此，我们设定的多步估计 $$Q^{\mathrm{ret}}$$ 的目的是双重的：减少策略梯度的偏差，并使critic能够更快地学习，从而进一步减少偏差。

#### **IMPORTANCE WEIGHT TRUNCATION WITH BIAS CORRECTION**

为了防止高方差，我们建议通过以下对 ****$$g^{\text { marg }}$$ 的分解来截断重要性权重并引入校正项：

$$
g^{\operatorname{marg}}=\mathbb{E}_{x_{t} a_{t}}\left[\rho_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | x_{t}\right) Q^{\pi}\left(x_{t}, a_{t}\right)\right] \\
=\mathbb{E}_{x_{t}}\left[\mathbb{E}_{a_{t}}\left[\overline{\rho}_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | x_{t}\right) Q^{\pi}\left(x_{t}, a_{t}\right)\right]+\mathbb{E}_{a \sim \pi}\left(\left[\frac{\rho_{t}(a)-c}{\rho_{t}(a)}\right]_{+} \nabla_{\theta} \log \pi_{\theta}(a | x_{t}) Q^{\pi}\left(x_{t}, a\right)\right)\right]
$$

其中 $$\overline{\rho}_{t}=\min \left\{c, \rho_{t}\right\} \text { with } \rho_{t}=\frac{\pi\left(a_{t} | x_{t}\right)}{\mu\left(a_{t} | x_{t}\right)}$$ ，并且 $$[x]_{+}=x \text { if } x>0$$ 其他情况取0。左起第一项为截断权重，第二项为校正项。我们提醒读者，上述期望是关于行为策略下的极限状态分布: $$x_{t} \sim \beta$$ 和 $$a_{t} \sim \mu$$ 。

我们用神经网络 $$Q_{θ_v}(x_t,a_t)$$ 拟合逼近校正项中的 $$Q^π(x_t,a)$$ ，这种修改我们称之为truncation with bias correction trick，在这种情况下应用于函数 $$\nabla_{\theta} \log \pi_{\theta}\left(a_{t} | x_{t}\right) Q^{\pi}\left(x_{t}, a_{t}\right) $$ ：

 $$\widehat{g}^{\operatorname{marg}}= \mathbb{E}_{x_{t}}\left[\mathbb{E}_{a_{t}}\left[\overline{\rho}_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | x_{t}\right) Q^{r e t}\left(x_{t}, a_{t}\right)\right]+\mathbb{E}_{a \sim \pi}\left(\left[\frac{\rho_{t}(a)-c}{\rho_{t}(a)}\right]_{+}^{ } \nabla_{\theta} \log \pi_{\theta}(a | x_{t}) Q_{\theta_{v}}\left(x_{t}, a\right)\right)\right]$$

利用行为策略 $$\mu$$ 采样的样本轨迹 $$\left\{x_{0}, a_{0}, r_{0}, \mu(\cdot | x_{0}), \cdots, x_{k}, a_{k}, r_{k}, \mu(\cdot | x_{k})\right\}$$ ，可以近似得到 off-policy ACER梯度:

$$
\widehat{g}_{t}^{\operatorname{acer}}=\overline{\rho}_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | x_{t}\right)\left[Q^{\mathrm{ret}}\left(x_{t}, a_{t}\right)-V_{\theta_{v}}\left(x_{t}\right)\right]
\\
+\underset{a \sim \pi}{\mathbb{E}}\left(\left[\frac{\rho_{t}(a)-c}{\rho_{t}(a)}\right]_{+} \nabla_{\theta} \log \pi_{\theta}(a | x_{t})\left[Q_{\theta_{v}}\left(x_{t}, a\right)-V_{\theta_{v}}\left(x_{t}\right)\right]\right)
$$

在上述表达式中，我们减去了经典的基线 $$V_{θ_v}(x_t )$$ 以减少方差。

#### **EFFICIENT TRUST REGION POLICY OPTIMIZATION**

Actor-Critic的策略更新经常表现出很大的方差。因此，为了确保稳定，我们必须限制策略的每一步变化。仅仅使用较小的学习速率是不够的，因为他们不能在保持期望的学习速度的同时防止偶尔的大规模更新。信任区域政策优化\( TRPO \) 提供了更完善的解决方案。

尽管TRPO方法有效，但每次更新都需要重复计算Fisher向量乘积。这在大规模问题下被证明是非常费时的。

在本节中，我们将介绍一种新的信任域策略优化方法，该方法可以很好地扩展到大规模问题。 我们建议维护一个average policy network代表过去策略的运行平均值，并强制更新的策略不偏离这一平均水平，而不是将更新后的策略限制在接近当前策略（如TRPO）。

我们将策略网络分解为两部分:一个概率分布$$f$$，一个深度神经网络，产生这个分布的统计参数 $$φ_θ(x )$$ 。也就是说，给定 $$f$$ ，策略完全由网络 $$\phi_{\theta} : \pi(\cdot | x)=f(\cdot | \phi_{\theta}(x))$$ 表示。

我们将average policy network表示为 $$φ_{θ_a}$$ ，并在每次更新策略参数 $$θ$$ 后“柔和地”更新其参数 $$\theta_{a} \leftarrow \alpha \theta_{a}+(1-\alpha) \theta$$

例如，考虑前面定义的ACER策略梯度，但是关于 $$\phi$$的 

$$
\begin{aligned} \widehat{g}_{t}^{\operatorname{acer}}=& \overline{\rho}_{t} \nabla_{\phi_{\theta}\left(x_{t}\right)} \log f\left(a_{t} | \phi_{\theta}(x)\right)\left[Q^{\mathrm{ret}}\left(x_{t}, a_{t}\right)-V_{\theta_{v}}\left(x_{t}\right)\right] \\ &+\underset{a \sim \pi}{\mathbb{E}}\left(\left[\frac{\rho_{t}(a)-c}{\rho_{t}(a)}\right]_{+} \nabla_{\phi_{\theta}\left(x_{t}\right)} \log f\left(a_{t} | \phi_{\theta}(x)\right)\left[Q_{\theta_{v}}\left(x_{t}, a\right)-V_{\theta_{v}}\left(x_{t}\right)\right]\right) \end{aligned}
$$

给定average policy network，我们建议的信任区域更新包括两个阶段。在第一阶段，我们用线性化KL散度约束来解决以下优化问题

$$
\begin{array}{ll}{\underset{z}{\operatorname{minimize}}} & {\frac{1}{2}\left\|\hat{g}_{t}^{\text { acer }}-z\right\|_{2}^{2}} \\ {\text { subject to }} & {\nabla_{\phi_{\theta}\left(x_{t}\right)} D_{K L}[f(\cdot | \phi_{\theta_{a}}\left(x_{t}\right)) \| f(\cdot | \phi_{\theta}\left(x_{t}\right))]^{T} z \leq \delta}\end{array}
$$

由于约束是线性的，整个优化问题简化为简单的二次规划问题，利用KKT条件可以很容易地以封闭形式导出其解，令 $$k=\nabla_{\phi_{\theta}\left(x_{t}\right)} D_{K L}\left[f\left(\cdot\left|\phi_{\theta_{a}}\left(x_{t}\right)\|f(\cdot | \phi_{\theta}\left(x_{t}\right)]\right.\right.\right.$$ 

$$
z^{*}=\hat{g}_{t}^{\mathrm{acer}}-\max \left\{0, \frac{k^{T} \hat{g}_{t}^{\mathrm{acer}}-\delta}{\|k\|_{2}^{2}}\right\} k
$$

在第二阶段，我们利用反向传播。具体地，关于 $$φ_θ$$ 的更新的梯度，即 $$z^*$$ ，通过网络反向传播，以计算与参数相关的导数。 策略网络的参数更新遵循链规则： $$\frac{\partial \phi_{\theta}(x)}{\partial \theta} z^{*}$$ 

信任区域步骤在分布的统计空间中执行，而不是在策略参数的空间中执行。这样做是故意的，以避免通过策略网络进行额外的反向传播。

ACER算法源于上述想法的组合，所以想要深入理解原理，需参阅上面引用的论文。

### CONTINUOUS ACTOR CRITIC WITH EXPERIENCE REPLAY

Retrace需要估计Q和V，但是我们不能轻易连续的动作空间中利用积分求解Q和V。 在本节中，我们以RL的新颖表示形式提出了这个问题的解决方案，以及信任区域更新所需的修改。

#### POLICY EVALUATION

![](../../.gitbook/assets/image%20%288%29.png)

我们提出了一个SDN网络（借鉴Dueling Deep-Q Network）解决这个问题，在每个时间步，SDN输出 $$Q_π$$ 的随机估计 $$\widetilde{Q}_{\theta_{v}}$$ 和 $$V_π$$ 的确定性估计 $$V_θ$$ ，使得

$$
\widetilde{Q}_{\theta_{v}}\left(x_{t}, a_{t}\right) \sim V_{\theta_{v}}\left(x_{t}\right)+A_{\theta_{v}}\left(x_{t}, a_{t}\right)-\frac{1}{n} \sum_{i=1}^{n} A_{\theta_{v}}\left(x_{t}, u_{i}\right), \text { and } u_{i} \sim \pi_{\theta}(\cdot | x_{t})
$$

然而，除了SDN之外，我们还构建了以下用于估计的新目标

$$
V^{\text {target}}\left(x_{t}\right)=\min \left\{1, \frac{\pi\left(a_{t} | x_{t}\right)}{\mu\left(a_{t} | x_{t}\right)}\right\}\left(Q^{\mathrm{ret}}\left(x_{t}, a_{t}\right)-Q_{\theta_{v}}\left(x_{t}, a_{t}\right)\right)+V_{\theta_{v}}\left(x_{t}\right)
$$

最后，当估计在连续域估计 $$Q^{\mathrm{ret}}$$时，我们实现了一个稍微不同的截断重要性权重公式， $$\overline{\rho}_{t}=\min \left\{1,\left(\frac{\pi\left(a_{t} | x_{t}\right)}{\mu\left(a_{t} | x_{t}\right)}\right)^{\frac{1}{d}}\right\} $$ ，d是动作空间的维度。虽然不是必需的，但我们发现这种配方可以加快学习速度。

#### TRUST REGION UPDATING

对于分布 $$f $$ ，我们选择具有固定对角协方差和均值的高斯分布 $$\phi_{\theta}(x) $$ 

考虑关于随机Deuling Network的ACER策略梯度

$$
\begin{aligned} g_{t}^{\text { acer }}=& \mathbb{E}_{x_{t}}\left[\mathbb{E}_{a_{t}}\left[\overline{\rho}_{t} \nabla_{\phi_{\theta}\left(x_{t}\right)} \log f\left(a_{t} | \phi_{\theta}\left(x_{t}\right)\right)\left(Q^{\mathrm{opc}}\left(x_{t}, a_{t}\right)-V_{\theta_{v}}\left(x_{t}\right)\right)\right]\right.\\ &+\underset{a \sim \pi}{\mathbb{E}}\left(\left[\frac{\rho_{t}(a)-c}{\rho_{t}(a)}\right]_{+}\left(\widetilde{Q}_{\theta_{v}}\left(x_{t}, a\right)-V_{\theta_{v}}\left(x_{t}\right)\right) \nabla_{\phi_{\theta}\left(x_{t}\right)} \log f(a | \phi_{\theta}\left(x_{t}\right))\right) ] \end{aligned}
$$

这里，我们使用 $$Q^{\mathrm{opc}}$$ 代替 $$Q^{\mathrm{ret}}$$ ， $$Q^{\mathrm{opc}}\left(x_{t}, a_{t}\right)$$ 与Retrace相同，但截断的重要性比率替换为1。给定状态 $$x_t$$ ，可以从 $$a_{t}^{\prime} \sim \pi_{\theta}(\cdot | x_{t})$$ 采样通过蒙特卡洛近似得到：

$$
\begin{aligned} \hat{g}_{t}^{\text { acer }}=& \overline{\rho}_{t} \nabla_{\phi_{\theta}\left(x_{t}\right)} \log f\left(a_{t} | \phi_{\theta}\left(x_{t}\right)\right)\left(Q^{\mathrm{opc}}\left(x_{t}, a_{t}\right)-V_{\theta_{v}}\left(x_{t}\right)\right) \\ &+\left[\frac{\rho_{t}\left(a_{t}^{\prime}\right)-c}{\rho_{t}\left(a_{t}^{\prime}\right)}\right]\left(\widetilde{Q}_{\theta_{v}}\left(x_{t}, a_{t}^{\prime}\right)-V_{\theta_{v}}\left(x_{t}\right)\right) \nabla_{\phi_{\theta}\left(x_{t}\right)} \log f\left(a_{t}^{\prime} | \phi_{\theta}\left(x_{t}\right)\right) \end{aligned}
$$

接下来就和离散的情况一样了。

## 伪代码

![](../../.gitbook/assets/image%20%2839%29.png)

![](../../.gitbook/assets/image%20%2844%29.png)

![](../../.gitbook/assets/image%20%281%29.png)

## 实验

#### 雅达利游戏机

![](../../.gitbook/assets/image%20%283%29.png)

#### MuJoCo

![](../../.gitbook/assets/image%20%2830%29.png)

