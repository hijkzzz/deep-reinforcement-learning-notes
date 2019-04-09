# Macro-action PPO

## 介绍

> [TStarBots: Defeating the Cheating Level Builtin AI in StarCraft II in the Full Game](https://arxiv.org/pdf/1809.07193)

星际争霸2\(SC2\)被广泛认为是最具挑战性的即时战略游戏游戏。潜在的挑战包括巨大的观察空间，巨大的\(连续的和无限的\)行动空间，部分观察，所有玩家同时移动，以及长期的延迟对当地决策的奖励。为了推动人工智能研究的前沿，Deepmind和Blizzard联合开发了星际争霸II学习环境（SC2LE）作为复杂决策系统的测试平台。SC2LE提供了一些迷你游戏，如MOToBeacon，CollectMineralShards和DefeatRoaches，其中一些AI代理已经达到了人类职业玩家的性能水平。然而，对于完整的游戏，当前的AI代理仍然远未达到人类专业水平的表现。对于bridgethis差距，我们在本文中提出了两个完整的游戏AI代理 - AI TStarBot1基于对平面动作结构的深度强化学习，AI TStarBot2基于分层动作结构上的硬编码规则。THtarBot1和TStarBot2能够在完整游戏中击败内置AI代理从1级到10级（AbyssalReef地图上的1v1Zerg-vs-Zerggame），注意到8级，9级和10级作弊代理具有不公平的优势，如 整个地图的全景和资源收获提升1。据我们所知，这是第一个调查可以在星际争霸II完整游戏中击败内置AI的AI代理的公开工作。

## 方法

### PySC2 Extension

SC2LE \[15\]是由DeepMind和Blizzard联合开发的平台。 暴雪提供的游戏内容暴露了原始界面和功能图界面。 DeepMind PySC2环境进一步在Python中包装核心库并完全公开了特征映射接口。 目的是密切模仿人类控制（例如，鼠标点击或按下某些键盘按钮），由于SC2的复杂性，这引入了大量的动作。 因此，它为基础决策系统带来了困难。而且，这种“玩家级”建模对于设计“单元级”模型是不方便的，尤其是在考虑多个代理时。 在这项工作中，我们进一步努力实现单元级控制。 此外，我们将上述建筑依赖项编码为技术树。

1. Expose unit control 在我们的PySC2扩展中，我们公开了SC2核心库的原始接口，它支持每单位的观察和操作。
2. Encode the technology tree 在星际争霸II中，玩家可能需要特定单位（建筑物或技术人员）作为其他先进单位（或建筑物或技术人员）的先决条件。在UrAlbertaBot \[19\]之后，我们将这些依赖关系形式化为一个技术树，缩写为TechTreein我们的PySC2扩展。

### TStarBot1

#### A Macro Action Based Reinforcement Learning Agent

![](../../.gitbook/assets/image%20%2810%29.png)

#### Macro Actions

我们为Zerg-vs-ZergSC2完整游戏设计了165个宏动作，如表1所示（完整列表请参阅附录-I）。 如上所述，宏观行动的目的是双重的：

1. 仅使用试错法对游戏的内在规则进行编码，这些规则难以学习
2. 通过硬编码决策隐藏学习算法中的不重要决策

![](../../.gitbook/assets/image%20%2813%29.png)

#### Observations and Rewards

观察结果被表示为一组空间二维特征图和一组非空间标量特征，这些特征是从SC2 游戏内核提供的单位信息中提取的，这些信息在我们的PySC2扩展中公开。

1. Spatial Feature Maps 提取的特征图的大小为N×N，其中N小于屏幕尺寸。 特征图的每个像素对应于整个世界地图中的小区域，表示某个统计量，例如该区域中某种类型的单位数。 这些数量包括玩家和对手常用单位类型的数量，以及具有某些属性的单位数，例如“can-attack-ground”和“can-attack-air”
2. Non-spatial Features 标量特征包括收集的天然气和矿物质的数量，剩余的食物量，每种单位类型的数量等。它们还包括最近采取的跟踪过去信息的方法
3. Rewards 我们使用三元值奖励函数：在游戏结束时收到1（赢）/ 0（平局）/ -1（失败）。 在比赛期间奖励总是为零。 虽然奖励信号非常稀疏，并且具有很长的时间范围延迟，但它仍然适用于这项工作中提出的宏观结构。

#### Learning Algorithms and Neural Network Architectures

1. Dueling Double Deep Q-learning \(DDQN\) 参考DDQN算法
2. Proximal Policy Optimization \(PPO\) 参考PPO算法
3. Neural Network Architecture 我们采用多层感知神经网络来参数化状态 - 动作值函数，状态值函数和策略函数。可以考虑更复杂的网络体系结构（例如，提取空间特征的卷积层，或补偿部分观察的循环层） ，我们将把它们留给未来的工作。
4. Distributed Rollout Infrastructure SC2游戏核心是CPU密集型的，并且推出速度慢，导致RL训练期间出现瓶颈。 为了解决这个问题，我们构建一个分布式部署基础设施，其中使用了一组CPU机器（称为actor）来并行执行部署过程。 高速缓存在每个actor的显示内存中的推出体验被随机采样并定期发送到基于GPU的机器（称为学习者）。我们目前使用1920个并行参与者\(80台机器上有3840个处理器\)以每秒大约16000帧的速度生成重放转换。这显著减少了训练时间\(从几周减少到几天\)，并且由于探索轨迹的多样性增加而提高了学习稳定性。

### TstartBot2

#### Hierarchical Macro-Micro Action Based Agent

上面描述的基于宏动作的代理有一些限制。 尽管宏操作可以根据功能进行分组，但是单个控制器必须修复所有操作组，其中不同组的操作在每个决策步骤中是互斥的。 此外，在预测要采取的操作时，控制器采用不知道操作组的共同观察。 这相当于训练控制器的不必要的困难，因为不期望的信息可能会引发观察和反应。 此外，宏观行动对单个单位（即，每单位控制）没有任何控制，当我们想要采用多代理方式时，这是不灵活的。

为了提高灵活性，我们创建了一组不同的动作，如图3所示。我们既包括宏观行动，也包括微观行动，组织在两层结构中。

![](../../.gitbook/assets/image%20%28118%29.png)

这种层次结构有两个优点。 1）每个控制器都有自己的观察/动作空间，这样就可以更容易地过滤出无关信息;在对子任务Q头进行建模时，\[32\]也采用并讨论了这一点。 2）层次结构更好地捕获游戏的动作结构，同时来自不同控制器的动作。

虽然理想情况下控制器应该单独或联合使用RL进行训练，但在这项工作中我们只使用专家规则，目的是验证提出的分层动作集方法。

如图4所示，每个控制器代表一个模块，以类似于UrbertaBot的方式组织。 第一层模块（CombatStrategy，ProductionStrategy）仅发出高级命令（宏操作），而第二层模块（Combat，Scout，Resourceand Building）发出低级命令（微动作）。

![](../../.gitbook/assets/image%20%2811%29.png)

#### Data Context







