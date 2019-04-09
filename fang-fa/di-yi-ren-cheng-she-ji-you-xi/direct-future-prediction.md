# Direct Future Prediction

## 介绍

> [LEARNING TOACT BYPREDICTING THEFUTURE](https://arxiv.org/pdf/1611.01779.pdf)

我们提出了一种沉浸式环境中感觉运动控制的方法。我们的方法利用高维感应流和低维测量流。这些流的同时结构提供了丰富的监控信号，可以通过与环境相互作用来训练感觉运动控制模型。该模型使用监督学习技术进行培训，但没有外部监督。它学会在一个复杂的三维环境中，根据原始的感觉输入采取行动。所提出的公式使得学习在训练时没有固定的目标，而在测试时追求动态变化的目标。我们基于经典的第一人称游戏Doom在三维模拟中进行了大量实验。结果表明，所提出的方法优于复杂的先验公式，特别是在具有挑战性的任务上。他的结果还表明，训练模型成功地在环境和目标中推广。使用这种方法训练的模型赢得了Visual Doom AI Competition 中的 Full Deathmatch track，并且是在以前从未见过的环境中。

## 方法

### Model

在本方法中观察状态可以分为两个变量： $$\mathbf{o}_{t}=\left\langle\mathbf{s}_{t}, \mathbf{m}_{t}\right\rangle$$ ，其中 $$s_t$$ 是原始的图像输入， $$m_t$$ 是一些测量指标，如子弹数、血量。 不同的未来状态与当前测量的差分可以表示为： $$\mathbf{f}=\left\langle\mathbf{m}_{t+\tau_{1}}-\mathbf{m}_{t}, \dots, \mathbf{m}_{t+\tau_{n}}-\mathbf{m}_{t}\right\rangle$$ 。设任何目标 $$g$$可以用函数 $$u(\mathbf{f} ; \mathbf{g})$$ 表达：

$$
u(\mathbf{f} ; \mathbf{g})=\mathbf{g}^{\top} \mathbf{f}
$$

目标即不同测量指标的权重，如生命值权重为1，其它为0.5。

为了预测未来的测量值，我们使用一个函数近似：

$$
\mathbf{p}_{t}^{a}=F\left(\mathbf{o}_{t}, a, \mathbf{g} ; \boldsymbol{\theta}\right)
$$

然后选择函数 $$u$$ 最大的动作执行：

$$
a_{t}=\underset{a \in \mathcal{A}}{\arg \max } \mathbf{g}^{\top} F\left(\mathbf{o}_{t}, a, \mathbf{g} ; \boldsymbol{\theta}\right)
$$

很显然，这里的测量值就类似于标准强化学习里面的回报

#### Training

目标函数被定义为：

$$
\mathcal{L}(\boldsymbol{\theta})=\sum_{i=1}^{N}\left\|F\left(\mathbf{o}_{i}, a_{i}, \mathbf{g}_{i} ; \boldsymbol{\theta}\right)-\mathbf{f}_{i}\right\|^{2}
$$

我们评估了两种训练方法：

* 单一目标:目标向量在整个训练过程中是固定的
* 随机目标：每集的目标向量是随机生成的

### ARCHITECTURE

![](../../.gitbook/assets/image%20%28146%29.png)

这里使用了类似于Dueling的网络结构，首先通过卷积网络、全连接网络得到 $$s, m, g$$ 。然后预测一个 $$E$$ 

作为未来测量值的期望， $$A$$ 为动作优势值，并且归一化成均值0。这样做的好处是让网络更容易学习预测，可以参考Dueling DQN。

$$
\overline{A^{i}}(\mathbf{j})=A^{i}(\mathbf{j})-\frac{1}{w} \sum_{k=1}^{w} A^{k}(\mathbf{j})
$$

$$
\mathbf{p}=\left\langle\mathbf{p}^{a_{1}}, \ldots, \mathbf{p}^{a_{w}}\right\rangle=\left\langle\overline{A^{1}}(\mathbf{j})+E(\mathbf{j}), \ldots, \overline{A^{w}}(\mathbf{j})+E(\mathbf{j})\right\rangle
$$

## 实验

### 固定场景

![](../../.gitbook/assets/image%20%2882%29.png)

![](../../.gitbook/assets/image%20%2894%29.png)

### 与目标无关的训练

![](../../.gitbook/assets/image%20%2888%29.png)



