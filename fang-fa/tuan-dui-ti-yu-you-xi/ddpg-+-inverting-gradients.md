# DDPG + Inverting Gradients

## 介绍

> [Deep reinforcement learning in parameterized action space](https://arxiv.org/abs/1511.04143)

最近的研究表明，深度神经网络能够逼近以连续状态和动作空间为特征的强化学习领域中的价值函数和策略。然而，据我们所知，在结构化的\(参数化的\)连续动作空间中，还没有先前的工作成功地使用深度神经网络。为了填补这一空白，本文重点研究模拟RoboCup足球领域的学习，其中包含一小组离散动作类型，每个类型都用连续变量参数化。 最好的代理人可以比2012年RoboCup冠军更可靠地进球。 因此，本文将深度强化学习成功扩展到参数化动作空间MDP类。

## 方法

### HALF FIELD OFFENSE DOMAIN

RoboCup是一项国际机器人足球比赛，旨在促进人工智能和机器人技术的研究。在RoboCup中，2D模拟联盟与足球的抽象，其中球员，球和场都是二维物体。

该代理使用一个低级别，以自我为中心的观点，使用58个连续值特征进行编码。这些特征是通过Helios-Agent2D（Akiyama，2010）世界模型得出的，并提供与各种重要现场对象（如球， 目标，以及其他球员。

![](../../.gitbook/assets/image%20%2845%29.png)

Half Field Offense具有低级参数化动作空间。 有四种相互排斥的离散动作：Dash，Turn，Tackle和Kick。 在每个时间步，代理必须选择这四个中的一个来执行。每个动作都有1-2个连续值参数，这些参数也必须指定。 代理必须选择它希望执行的离散动作以及该动作所需的连续值参数。

HFO领域的真正奖励来自赢得完整的比赛。 然而，这种奖励信号对于学习代理人获得牵引力来说太过稀疏。相反，我们引入了一个手工制作的奖励信号，其中包含四个部分：Move To Ball Reward提供与代理和balld之间距离变化成比例的标量奖励 $$d(a, b)$$ 。Kick To Goal Reward与球和球门中心之间的距离变化成比例 $$d(b, g)$$ 。 为进球而得分的额外奖励。

![](../../.gitbook/assets/image%20%28138%29.png)

### Actor-Critic architecture

![](../../.gitbook/assets/image%20%28104%29.png)

### PARAMETERIZED ACTION SPACE ARCHITECTURE

参数化的动作空间可表示为 $$\left(a, p_{1}^{a}, \ldots, p_{m_{a}}^{a}\right)$$ ，即一个离散动作a与m个连续值p。对于HFO，整个动作空间可表示为：

$$
\left(\text { Dash }, p_{1}^{\text { dash }}, p_{2}^{\text { dash }}\right) \cup\left(\text { Turn, } p_{3}^{\text { turn }}\right) \cup\left(\text { Tackle, } p_{4}^{\text { tackle }}\right) \cup\left(\text { Kick }, p_{5}^{\text { kick }}, p_{6}^{\text { kick }}\right)
$$

对应图2架构的输出参数

#### ACTIONSELECTION ANDEXPLORATION

在训练期间，评论网络接收所有四个离散动作和所有六个动作参数的输出节点的值作为输入。 我们没有向评论家表明在HFO环境中实际应用了哪些离散动作，或者哪些连续参数与该离散动作相关联。类似地，当更新演员时，评论家为所有四个离散动作和所有六个连续参数提供梯度。虽然评论家似乎缺乏关于动作空间结构的基本信息，但我们在后面的实验结果表明，评论家学会了为每个离散动作的正确参数提供梯度。

连续作用空间的探索不同于离散空间。 我们将 $$\epsilon-greedy$$探索参数化的行动空间，连续值使用使用均匀随机分布进行采样 。并且探索幅度采用模拟退火。

### BOUNDED PARAMETER SPACE LEARNING

HFO限制了每个连续参数的范围。 参数指示方向（例如转弯和踢动方向）以\[-180,180\]为界，并且功率参数（例如踢和力量）以\[0,100\]为界。在没有强制执行这些限制的情况下，经过几百次更新后，我们观察到常规超出边界的连续参数。 如果允许继续更新，参数将迅速趋向于天文数字大的值。这个问题源于评论家提供的渐变，鼓励演员网络继续增加已超出界限的参数。 我们探索了三种在预期范围内保留参数的方法：

#### Zeroing Gradients

也许最简单的方法是检查每一个参数的批评家的梯度，并且将那些建议增加/减少已经在其范围的上限/下限的参数值的梯度归零：

$$
\nabla_{p}=\left\{\begin{array}{ll}{\nabla_{p}} & {\text { if } p_{\min }<p<p_{\max }} \\ {0} & {\text { otherwise }}\end{array}\right.
$$

#### Squashing Gradients

双曲正切\(tanh\)等挤压函数用于限制每个参数的激活。随后，参数被重新缩放到它们的预定范围。这种方法的优点是不需要人工进行梯度修补，但是如果挤压函数饱和，就会出现问题。

#### Inverting Gradients

这种方法捕获了归零和挤压梯度的最佳方面，同时最大限度地减少了缺点。当参数接近其范围的边界时，梯度会缩减，如果参数超出值范围，则会反转。这样可以在避免饱和问题的同时将参数保持在界限范围内。

![](../../.gitbook/assets/image%20%28120%29.png)

## 实验

![](../../.gitbook/assets/image%20%2841%29.png)



