# H-DRLN

## 介绍

> [A Deep Hierarchical Approach to Lifelong Learning in Minecraft](https://arxiv.org/pdf/1604.07255.pdf)

我们提出了一种终身学习系统，它能够重用并将知识从一个任务转移到另一个任务，同时有效地保留以前学过的知识库。在《我的世界》，通过学习可重复使用的技能来传递知识，这是一个流行的视频游戏，是一个尚未解决的高维终身学习问题。这些可重用的技能，我们称之为深度技能网络，然后结合到我们新颖的分层深度强化学习网络\( H-DRLN \)体系结构中，使用两种技术：（1）深度技能数组（2）技能蒸馏，我们用于学习技能的策略蒸馏新变体。技能蒸馏使H-DRLN通过积累知识和将多种可重用技能封装到一个单一的蒸馏网络中，有效地保留知识，从而在终身学习中扩展规模。与常规深度Q网络相比，H-DRLN表现出优越的性能和较低的学习样本复杂度在《我的世界》的中。

## 方法

### Life long learning

![](../../.gitbook/assets/image%20%28115%29.png)

《我的世界》是一个终身学习问题

![](../../.gitbook/assets/image%20%28124%29.png)

### Deep Skill Module

预先学习的技能被表示为深层网络，被称为Deep Skill Networks（DSNs）。即使用我们的DQN算法版本以及常规经验重放（ER）对不同的任务进行了先验训练。我们定义了两种类型的DSN，如图3中的ModuleA和ModuleB。前者即神经网络版，后者为策略蒸馏的版本。

![](../../.gitbook/assets/image%20%2850%29.png)

### H-DRLN architecture

图3上部即H-DRLN的整体架构。在这里，H-DRLN的输出包括原始动作和技能。 H-DRLN学习一种策略，该策略确定何时执行原始动作以及何时重新使用预先学习的技能。当选择原始动作的时候，仅运行一步；而选择技能的时候，会运行直到技能 $$\pi_{D S N_{i}}(s)$$ 终止。

### Skill Objective Function

如前所述，H-DRLN扩展了vanilla DQN架构，以学习原始动作和技能之间的控制。所以学习目标如下：

$$
Q_{\Sigma}^{*}(s, \sigma)=\mathbb{E}\left[R_{s}^{\sigma}+\gamma^{k} \max _{\sigma^{\prime} \in \Sigma} Q_{\Sigma}^{*}\left(s^{\prime}, \sigma^{\prime}\right)\right]
$$

其中 $$R_{s}^{\sigma}$$ 是技能 $${\sigma}$$ 获得的回报。具体来说的目标函数如下

$$
y_{t}=\left\{\begin{array}{ll}{\sum_{j=0}^{k-1}\left[\gamma^{j} r_{j+t}\right]} & {\text { if } s_{t+k} \text { terminal }} \\ {\sum_{j=0}^{k-1}\left[\gamma^{j} r_{j+t}\right]+\gamma^{k} \max _{\sigma^{\prime}} Q_{\theta_{\text {target}}}\left(s_{t+k}, \sigma^{\prime}\right)} & {\text { else }}\end{array}\right.
$$

### Skill - Experience Replay

我们扩展了常规的经验回放\(ER\)来整合技能，并将其称为技能经验回放\( S-ER \)。S-ER保存 $$\left(s_{t}, \sigma_{t}, \tilde{r}_{t}, s_{t+k}\right)$$ ，即保存技能的回报总和和最后一个状态。

## 实验

### State space

状态空间表示为来自最后四个图像帧的原始图像像素，这些像素被组合并下采样为84×84像素图像。动作 - DSN的原始动作空间由六个动作组成：（1）向前移动，（2）向左旋转30°，（3）向右旋转30°，（4）向内旋转，（5）拾取项目和（ 6）放置它。奖励——在所有领域，代理人在每一步后都会得到一个小的负面奖励信号，在达到最终目标时会得到一个非负面的奖励。见图4和图5。

![](../../.gitbook/assets/image%20%2830%29.png)

### Training a DSN

单个DSN、两个房间域和复杂域的剧集长度分别为30、60和100步。代理在每个DSN中的随机位置以及两个空间和复杂域的第一个空间中初始化。

为了训练不同的DSN，我们使用VanillaDQN架构（Mnih等人2015）并执行网格搜索以找到Minecraft中学习DSN的最佳超参数。

### Training an H-DRLN with a DSN

在这个实验中，我们训练H-DRLN代理通过重用单个DSN（在navigation1 domain上预先训练）来解决两个房间的复杂任务。

我们在两个房间领域（图4）训练了H-DRLN架构以及标准DQN。

![](../../.gitbook/assets/image%20%2870%29.png)

### Training an H-DRLN with a Deep Skill Module

在本节中，我们将讨论我们的培训结果，并使用深度技能模块来利用H-DRLN来解决复杂的Minecraft领域。

![](../../.gitbook/assets/image%20%28127%29.png)

这里的Deep Skill Module即多技能网络，通过多个Teacher网络指导一个Student学习多种技能。

![](../../.gitbook/assets/image%20%28174%29.png)





### 



