# Ape-X  DQfD

## 介绍

> [Observe and Look Further: Achieving Consistent Performance on Atari](https://arxiv.org/pdf/1805.11593.pdf)

尽管深度强化学习（RL）领域取得了重大进展，但今天的算法仍然无法在Atari 2600游戏等一系列多项任务中持续学习人类级别的策略。我们确定了任何算法都需要掌握的三个关键挑战，才能在所有游戏中表现出色:处理各种奖励分配、长期推理和高效探索。在这篇文章中，我们提出了一个算法来解决每一个挑战，并且能够学习几乎所有雅达利游戏中的人类层面的策略。一个新的transformed Bellman算子允许我们的算法处理不同密度和尺度的奖励; 辅助时间一致性损失使我们能够使用 $$\gamma$$ = 0.999（而不是 $$\gamma$$ = 0.99）的贴现因子稳定地训练，将有效规划的范围扩大一个数量级;我们通过使用人类演示来引导代理人奖励状态，从而缓解探索问题。

## 方法

### Transformed Bellman Operator

### Temporal consistency \(TC\) loss

### Ape-X DQfD

在本节中，我们将描述如何将变换后的Bellman算子和TC损失与DQfD算法和分布式优先级经验重放相结合。

![](../../.gitbook/assets/image%20%287%29.png)

![](../../.gitbook/assets/image%20%2834%29.png)





