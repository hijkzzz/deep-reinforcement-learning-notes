# Double Deep Q-Learning

## 介绍

> [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)

众所周知，流行的Q-learning算法会过估计某些条件下的动作值。在实践中，这种过高估计是否普遍存在，是否会损害性能，是否能够从根本上加以预防，这些都是人们以前所不知道的。在这篇论文中，我们肯定地回答了所有这些问题。特别地，我们首先展示了现有的DQN算法，它将Q-learning与adeep神经网络相结合，在雅达利2600领域的一些游戏中存在严重的过估计。然后，我们展示了双Q学习算法背后的思想，它是在表格中介绍的，可以推广到大规模函数逼近。我们对DQN算法提出了一种特殊的适应方法，结果表明算法不仅减少了所观察到的过估计，正如假设的那样，而且这也在了几款游戏的带来了更好性能。

## 算法

### 双Q学习

最优贝尔曼方程

$$Y_{t}^{\mathrm{Q}} \equiv R_{t+1}+\gamma \max _{a} Q\left(S_{t+1}, a ; \boldsymbol{\theta}_{t}\right)$$ 

上式中的最大运算符使用相同的值来选择和评估动作。这使得选择过度估计的值的可能性更大，从而导致过于乐观的值估计。为了阻止这一点，我们可以将选择从评估中分离出来。这就是Double Q-Learning背后的思想。

在原始的双Q学习算法中，通过将经验样本随意分配给两个Q网络中的一个，训练得到两组权重θ和θ'。 对于每次更新，一组权重用于确定贪婪策略，另一组用于确定其值。 为了进行清晰的比较，我们可以首先分离Q学习中的选择和评估过程，并将其目标写为

$$Y_{t}^{\mathrm{Q}}=R_{t+1}+\gamma Q\left(S_{t+1}, \underset{a}{\operatorname{argmax}} Q\left(S_{t+1}, a ; \boldsymbol{\theta}_{t}\right) ; \boldsymbol{\theta}_{t}\right)$$ 

然后可以将Double Q-learning训练误差写为

$$Y_{t}^{\text { Double } Q} \equiv R_{t+1}+\gamma Q\left(S_{t+1}, \operatorname{argmax} Q\left(S_{t+1}, a ; \boldsymbol{\theta}_{t}\right) ; \boldsymbol{\theta}_{t}^{\prime}\right)$$ 

请注意，在 $$argmax$$ 中，动作是根据在线权重$$θ_t$$选择的。这意味着，与Q学习一样，我们仍然根据 $$θ_t$$ 定义的值来估计贪婪策略的值。然而，我们使用第二组权重 $$θ′$$ 来公平地评估该策略的值。通过切换θ和θ的角色，可以对称地更新第二组权重。

### 由于估计误差导致的过度乐观

![](../../.gitbook/assets/image%20%2813%29.png)

请注意，我们不需要假设不同动作的估计误差是独立的。这个定理表明，即使估计值平均正确，任何的估计误差都可以推动估计值远离真正的最优值。

定理1的下界随着动作的数量而减少， 这是考虑下限的假象。更典型的是，过度乐观随着动作的数量而增加，如图1所示。

![](../../.gitbook/assets/image%20%288%29.png)

Q学习的过估计的确会随着行动的增多而增加，而双Q学习是不偏不倚的。

![](../../.gitbook/assets/image%20%287%29.png)

![](../../.gitbook/assets/image%20%282%29.png)

### Double DQN

$$Y_{t}^{\text { DoubleDQN } } \equiv R_{t+1}+\gamma Q\left(S_{t+1}, \operatorname{argmax}_{a} Q\left(S_{t+1}, a ; \boldsymbol{\theta}_{t}\right), \boldsymbol{\theta}_{t}^{-}\right)$$ 

与Double Q-learning相比，第二个网络的权重被替换为target网络的权重 $$θ_t$$ ，用于评估当前的贪婪策略。 目标网络的更新与DQN保持不变，并且仍然是当前网络的定期副本。

