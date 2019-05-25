# BiC-Net

> [Multiagent Bidirectionally-Coordinated Network](https://arxiv.org/abs/1703.10069)

许多人工智能应用通常需要多个智能代理协同工作。智能体内部交流和协调的有效学习是实现通用人工智能不可或缺的一步。在本文中，我们以星际争霸战斗游戏为例，其任务是协调多个代理作为一个团队来打败他们的敌人。为了维护一个可扩展但有效的通信协议，我们引入了一个多智能体双向协调网络\(BiCNet\['bIknet\]\)，它具有行为者-批评者表述的矢量扩展。我们表明，BiCNet可以处理不同类型的战斗，双方都有任意数量的AI代理。我们的分析表明，没有任何监督，如人类演示或标记数据，BiCNet可以学习各种类型的高级协调策略，这些策略已被经验丰富的游戏玩家普遍使用。在我们的实验中，我们根据不同场景下的多个基线评估我们的方法；它显示了最先进的性能，并具有大规模实际应用的潜力。

## 方法

### StartCraft Combat as Stochastic Games

回报函数可以定义为我方平均血量和对方平均血量的差

![](../../.gitbook/assets/image%20%2873%29.png)

于是求解Q函数可以定义为一个极大极小问题

![](../../.gitbook/assets/image%20%28176%29.png)

为了简化问题，我们只考虑极大

![](../../.gitbook/assets/image%20%2891%29.png)

### Local, Individual Rewards

只考虑整体的回报显然无法有效分配信用，所以考虑每个单位自己的回报，即与该单位接触的top-k个单位的血量之差

![](../../.gitbook/assets/image%20%28123%29.png)

### Communication w/ Bidirectional Backpropagation

我们需要一种机制来实现多智能体的通信/协同训练，于是有了Bic-Net这种网络结构：

![](../../.gitbook/assets/image%20%2828%29.png)

策略网络与本地视图一起接受共享观察，返回单个代理的操作。 由于双向复现结构不仅可以作为通信渠道，还可以作为本地存储器，每个代理人都能够维持自己的内部状态，并与合作者共享信息。

对于BiCNet的学习，直观地说，我们可以考虑通过展开网络长度N（受控代理的数量）然后通过时间应用反向传播来计算后向梯度（BPTT）。

梯度同时传递给每个 $$Q_{i}$$ 函数和策略函数。它们是从所有代理和他们的动作中聚集起来的。换句话说，首先传播来自所有代理商奖励的梯度以影响每个代理商的活动，并且得到的梯度进一步传播回更新参数。

令 $$J_{i}(\theta)=\mathbb{E}_{\mathbf{s} \sim \rho_{a_{\theta}}^{T}}\left[r_{i}\left(\mathbf{s}, \mathbf{a}_{\theta}(\mathbf{s})\right)\right]$$ 为每个单位的最大化目标，其中 $$\rho_{\mathbf{a}_{o}}^{\mathcal{T}}(\mathbf{s})$$ 对应于策略的折扣状态分布，如 $$\rho_{\mathbf{a}_{\theta}}^{\mathcal{T}}(\mathbf{s}) :=\int_{\mathcal{S}} \sum_{t=1}^{\infty} \lambda^{\iota-1} p_{1}(\mathbf{s}) \mathbb{1}\left(\mathbf{s}^{\prime}=\mathcal{T}_{\mathbf{a}_{\theta}, \mathbf{b}_{\phi}}^{1}(\mathbf{s})\right) \mathrm{d} \mathbf{s}$$ ，它也可以作为遍历MDP的平稳分布。所以整体的目标函数为：

![](../../.gitbook/assets/image%20%2812%29.png)

#### Theorem 1 \(Multiagent Deterministic PG Theorem\)

接下来，我们将一个多智能体引入确定性策略梯度定理：

![](../../.gitbook/assets/image%20%28187%29.png)

为了确保充分的探索，我们应用Ornstein-Uhlenbeck过程在每个时间步骤中增加行动网络输出的噪声。在这里，我们进一步考虑了off-policy确定性的行动者-批评者算法\(Lillicrap等人\)来降低方差。在训练Critic时，我们使用平方损失的总和并对参数化的Critic$$Q^{\xi}(\mathbf{s}, \mathbf{a})$$具有以下梯度 。

![](../../.gitbook/assets/image%20%28167%29.png)

BiCNet与贪婪的MDP明显不同，代理的依赖性嵌入在潜在的层面，而不是直接在行动中。

## 实验

![](../../.gitbook/assets/image%20%28118%29.png)

