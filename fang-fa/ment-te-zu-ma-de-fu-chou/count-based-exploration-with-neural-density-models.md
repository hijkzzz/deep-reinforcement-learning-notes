# DQN-PixelCNN

## 介绍

> [Count-Based Exploration with Neural Density Models](https://arxiv.org/pdf/1703.01310.pdf)

Bellmare等人\( 2016 \)。引入了从密度模型导出的伪计数的概念，将基于计数的探索推广到非表格强化学习。这个伪计数被用来为DQN代理生成一个探测奖励，并与蒙特卡洛更新相结合，足以在Atari 2600游戏Montezuma的复仇中达到最先进的水平。我们认为他们的工作揭示了两个问题:第一，密度模型的质量对勘探有多重要？第二，蒙特卡罗更新在探索中扮演什么角色？我们通过演示PixelCNN的使用来回答第一个问题，PixelCCNN是一种先进的图像神经密度模型，用于支持伪计数。特别是，当对模型的假设进行违反时，我们研究了调整Bellemare等人的方法所面临的内在困难。结果是更实用和一般的算法，不需要特殊设备。 我们将PixelCNN伪计数与不同的体系结构相结合，以显着改善几个硬Atari游戏的艺术状态。 令人惊讶的发现是混合的蒙特卡洛更新在最稀疏的环境中是一个强大的探索促进者，包括Montezuma的复仇。

## 算法

### Pseudo-Count and Prediction Gain

令 $$\rho$$ 为输入空间的密度模型， $$\rho_{n}(x)$$ 为由序列 $$x_{1}, \dots, x_{n}$$ 训练的密度模型。 $$\rho_{n}^{\prime}(x)$$ 是模型用 $$x$$再训练一次后，对同一个 $$x$$ 的密度概率。 $$ρ$$ learning-positive即 $$\rho_{n}^{\prime}(x) \geq \rho_{n}(x) \text { for all } x_{1}, \ldots, x_{n}, x \in \mathcal{X}$$ 。$$\rho$$ 的prediction gain\(PG\) 即：

$$
\mathrm{PG}_{n}(x)=\log \rho_{n}^{\prime}(x)-\log \rho_{n}(x)
$$

learning-positive 意味着 $$\mathrm{PG}_{n}(x) \geq 0 \text { for all } x \in \mathcal{X}$$ ，对于 $$ρ$$ learning-positive，我们定义伪计数为：

$$
\hat{\mathrm{N}}_{n}(x)=\frac{\rho_{n}(x)\left(1-\rho_{n}^{\prime}(x)\right)}{\rho_{n}^{\prime}(x)-\rho_{n}(x)}
$$

从假设单个观察 $$x$$ 应当导致伪计数的单位增加得出：

$$
\rho_{n}(x)=\frac{\hat{\mathrm{N}}_{n}(x)}{\hat{n}}, \quad \rho_{n}^{\prime}(x)=\frac{\hat{\mathrm{N}}_{n}(x)+1}{\hat{n}+1}
$$

其中 $$\hat{n}$$ 表示总的伪计数。在 $$\rho_{n}$$的某些假设下，伪计数与真实计数大致线性增长。 至关重要的是，可以使用密度模型的PG来近似伪装计数：

$$
\hat{\mathrm{N}}_{n}(x) \approx\left(e^{\mathrm{PG}_{n}(x)}-1\right)^{-1}
$$

它的主要用途是定义一个探险奖金。我们考虑强化学习代理与提供观察和外在回报的环境相互作用。加上探索奖励：

$$
r^{+}(x) :=\left(\hat{\mathrm{N}}_{n}(x)\right)^{-1 / 2}
$$

这就激励代理人去尝试重新体验令人惊讶的情况。与PG相关的量化方法在内在动机文献中被用于类似目的，其中它们度量了代理的learning progress。 虽然伪计数奖励接近PG，但它渐渐变得更加保守并得到更强的理论保证的支持。

### Density Models for Images

CTS密度模型基于算法Context Tree Switching，一种Bayesian variable-order Markov模型。在最简单的形式中，该模型将2D图像作为输入，并根据位置相关的L形滤波器的乘积为其分配一个概率，其中每个滤波器的预测由在过去图像上训练的CTS算法给出。CTS模型在简单性和性能方面具有优势，但在表达能力、可扩展性和数据效率方面受到限制。

近年来，图像的神经生成模型在各种领域中产生多样性图像的能力方面取得了令人瞩目的成功。[PixelCNN](https://arxiv.org/pdf/1601.06759.pdf)，一个完全卷积的神经网络，由具有多重门控单元的残差网络组成，通过使用掩蔽卷积核对先前像素（通常的左上角到右下光栅扫描顺序）上的像素概率进行建模。该模型在标准数据集上实现了最先进的建模性能，并配以卷积前馈网络的计算效率。

### Multi-Step RL Methods

Q-Learning

![](../../.gitbook/assets/image%20%2881%29.png)

众所周知，在学习效率和近似误差方面，通过多步方法可以获得更好的性能。这些方法在一步学习和蒙特卡罗更新之间进行插值：

$$
Q(x, a) \leftarrow Q(x, a)+\alpha \underbrace{\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(x_{t}, a_{t}\right)-Q(x, a)\right]}_{\delta_{\mathrm{mc}(x, a)}}
$$

混合蒙特卡洛更新

$$
Q(x, a) \leftarrow Q(x, a)+\alpha\left[(1-\beta) \delta(x, a)+\beta \delta_{\mathrm{MC}}(x, a)\right]
$$

更好的多步法是最近的 $$Retrace(λ)$$ 算法，它使用截断的重要性采样比例乘积 $$c_{1}, c_{2}, \dots$$ 得到误差：

$$
\delta_{\mathrm{RETRACB}}(x, a) :=\sum_{t=0}^{\infty} \gamma^{t}\left(\prod_{s=1}^{t} c_{s}\right) \delta\left(x_{t}, a_{t}\right)
$$

### Using PixelCNN for Exploration

正如引言中提到的，使用密度模型进行探索的理论做出了几个假设，这些假设转化为实现的具体要求：

\(a\) 密度模型应该完全在线训练，即按照给定的顺序在代理经历的每个状态上精确地训练一次

\(b\) prediction gain（PG）应该以 $$n^{-1}$$ 的速率衰减，以确保伪计数的增长线性近似实际计数

\(c\) 密度模型应该是learning positve的

同时，训练神经密度模型并将其用作RL代理的一部分的实用性提出了一组部分竞争的要求。

\(d\) 为了稳定性，效率并避免在漂移数据分布的背景下进行灾难性的遗忘，这对从多样化的数据集中随机抽取小批量训练神经模型是有利的。

\(e\) 对于有效的训练，必须遵循某种优化方案（例如，固定的学习速率）

\(f\) 密度模型必须是计算轻量级，以允许计算PG（两个模型评估和一个更新）作为RL代理的每个训练步骤的一部分。

#### Designing a Suitable Density Model

为了应对要求\(f\)，为了我们设计了PixelCNN网络的超薄变体。 它的核心是一堆2个门控残差块，带有16个特征图（与15个残差块相比，在标准PixelCNN中有128个特征图）。 与CTS模型一样，图像被下采样到42×42并量化为3位灰度。

#### Training the Density Model

我们不是使用随机小批量，而是在经验状态序列上完全在线训练密度模型。经验上，我们发现，通过优化超参数的微调，我们可以在具有随机顺序的序列中的时间相关的状态序列上稳健地训练模型。

除了满足要求\(a\)，密度模型的完全在线训练具有 $$ρ'_n=ρ_n+ 1$$ 的优点，因此用于计算PG的模型更新无需恢复。

避免密度模型的小批量更新的另一个更微妙的原因（尽管有要求\(d\)）是一个实际的优化问题。PG的计算\(必须在线\)包括模型更新，与深度神经网络一起使用的高级优化器，如本工作中使用的RMSProp优化器，是有状态的，跟踪运行中的例如模型参数的平均值和方差。如果模型是从小批量处理的，那么两个更新流可能会显示不同的统计特征，使优化算法的假设无效，导致训练速度变慢或不稳定。

![](../../.gitbook/assets/image%20%2823%29.png)

我们探索了不同的学习速率

![](../../.gitbook/assets/image%20%2820%29.png)

#### Computing the Pseudo-Count

为了达到要求的PG衰减\(b\)，我们用 $$c_{n} \cdot \mathrm{PG}_{n}$$ 代替 $$\mathrm{PG}_{n}$$ ，其中 $$c_{n}$$ 是一个合适的衰减序列。在比较实际代理性能的实验中，我们通过实际上确定了常量学习率0.001，与PG衰减 $$c_{n}=c \cdot n^{-1 / 2}$$ 配对，在像蒙特祖玛的复仇这样的难探索游戏中获得最好的探索结果。

关于\(c\)，很难确保深度神经模型的learning positive，并且当优化器“超过”局部损失最小值时，可能会出现负PG。 作为解决方法，我们将PG值阈值设置为0。总结一下，计算出的伪计数是：

$$
\hat{\mathrm{N}}_{n}(x)=\left(\exp \left(c \cdot n^{-1 / 2} \cdot\left(\mathrm{PG}_{n}(x)\right)_{+}\right)-1\right)^{-1}
$$

## 实验

### DQN

![](../../.gitbook/assets/image%20%2888%29.png)

### Reactor

![](../../.gitbook/assets/image%20%2842%29.png)

### Quality of the Density Model

![](../../.gitbook/assets/image%20%281%29.png)

### Importance of the Monte Carlo Return

![](../../.gitbook/assets/image%20%2818%29.png)





















