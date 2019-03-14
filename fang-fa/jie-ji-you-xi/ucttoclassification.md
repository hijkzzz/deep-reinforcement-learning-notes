# UCTtoClassification

## 介绍

> [Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](https://web.eecs.umich.edu/~baveja/Papers/UCTtoCNNsAtariGames-FinalVersion.pdf)

现代强化学习和深度学习方法的结合有望在挑战性应用方面取得重大进展，这些应用需要丰富的认知和政策选择。街机学习环境\( ALE \)提供了一套雅达利游戏，代表了一套有用的这类应用的基准点。最近在将无模型强化学习与深度学习相结合方面的突破，被称为DQN，实现了迄今为止最好的实时代理。基于规划的方法比最好的无模型方法得分高得多，但是它们利用了人类玩家无法获得的信息，并且比实时播放所需的速度慢几个数量级。我们这项工作的主要目标是建立一个比DQN更好的实时雅达利游戏玩家。中心思想是使用基于缓慢规划的代理为能够实时的深度学习架构提供训练数据。我们基于这一想法提出了新的代理，并表明它们的性能优于DQN。

## 算法

### Planning Agents for Atari Games based on UCT

这些方法从仿真器访问游戏状态，因此面对确定性MDP（除了初始状态的随机选择）。他们使用UCT逐步规划动作以采用当前状态，UCT是一种广泛用于游戏的算法。 UCT有三个参数，轨迹数量，最大深度（均匀的对于每个轨迹）和探测参数（我们所有实验中的标量设置为1）。通常，轨迹和深度参数越大，UCT越慢但是越好。 UCT使用模拟器作为模型来模拟轨迹如下。

假设正在生成第 $$k$$ 个轨迹并且当前节点处于深度 $$d$$ 、当前状态为 $$s$$ 。它将状态-深度 $$(s, d)$$ 的每一个可能的动作的得分计算为两个项的和：第一个是利用项，是在前一段轨迹中使用状态-深度 $$(s, d)$$ 所获得的折扣后的奖励总和的蒙特卡罗平均值；第二个是探索项， $$\sqrt{\log (n(s, d)) / n(s, a, d)}$$ ， $$n$$表示访问次数 ， $$a表示动作$$ ，旨在促进探索不同的动作。

### Combining UCT-based RL with DL

#### Baseline UCT agent that provides training data

该代理不需要训练。 但是，它确实需要指定其两个参数，即轨迹的数量和最大深度。 回想一下，我们提议的新代理将全部使用来自该UCT代理的数据来训练基于CNN的策略，因此我们提出的代理的最终性能将比UCT代理的性能更差是合理的。 因此，在实验中，我们将这两个参数设置得足够大，以确保它们超过公布的DQN分数，但不会太大，以至于它们使我们的计算实验不合理地慢。具体来说，我们选择使用300作为所有游戏的最大深度和10000个轨迹数量但是两个 。 Pong原本是一个更简单的游戏，我们可以将轨迹数量减少到500，而Enduro的远端奖励比其他游戏更多，而且使用最大深度为400的sowe。 从第5节的结果可以明显看出，这允许theUCT代理在所有游戏中明显优于DQN，但DQN已经完美地完成了Pong。 我们强调UCT代理不符合我们的实时游戏目标。 例如，使用UCT代理玩游戏只需800次（我们这样做是为了收集我们代理商的培训数据）在最近的多核计算机上花费几天来完成每个游戏。

#### UCTtoRegression

关键思想是使用UCT代理计算的动作值来训练基于回归的CNN。 以下是针对每个游戏进行的。 收集800次运行样本，即使用上面的UCTagent从开始到结束玩游戏800次。 从这些运行中构建数据集（表），如下所示。 将每个状态的最后四个帧按每个轨迹映射到由UCT计算的所有动作的动作值。 该训练数据用于通过回归训练CNN（参见下面的CNN详细信息）。 UCTtoRegression-agent使用该训练过程学习的CNN来选择评估期间的动作。

#### UCTtoClassification

关键思想是使用由UCT代理计算的动作选择（从动作值中贪婪地选择）来训练基于分类器的CNN。 每场比赛都进行以下操作。 收集800次运行样本如上。 这些运行产生一个表，其中行对应于沿每个轨迹的每个状态的最后四个帧，并且单个列是根据轨迹状态下的UCT-agent最佳的动作选择。 该训练数据用于通过多项分类训练CNN（参见下面的CNN细节）。 UCTtoClassification-agent用该训练过程学习的CNN分类器来选择评估期间的动作。

![](../../.gitbook/assets/image%20%2862%29.png)

上面的方法有一个潜在的问题是，CNN的的决策可能与UCT不同，这就造成的了输入分布与实际游戏不一致。所以我们提出了下面的方法：

#### UCTtoClassification-Interleaved

收集200 UCT-agent运行如上; 这些显然会有相同的输入分布问题。 来自这些运​​行的数据用于通过多项分类来训练CNN，就像在UCTtoClassification-agent的方法中一样（我们不对UCTtoRegression-agent执行此操作，因为我们在下面显示它比UCTtoClassification-agent更糟糕）。 然后，训练有素的CNN用于决定收集另外200个运行中的动作选择（尽管选择随机动作的5％以确保一些探索）。 在每个轨道的游戏的每个状态下，UCT被要求计算其动作选择，并且原始数据集被增加，每个状态的最后四个帧作为行和列作为UCT的动作选择。 此400轨迹数据集的输入分布现在可能与UCT代理的输入分布不同。 该数据集用于通过多项分类再次训练CNN。 重复该交错过程，直到CNN的最后一轮训练的数据集中总共有800转的数据。 UCTtoClassification-Interleaved代理使用此训练过程学习的最终CNN分类器来选择测试期间的操作。


