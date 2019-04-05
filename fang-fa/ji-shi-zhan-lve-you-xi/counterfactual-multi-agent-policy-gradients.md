# Counterfactual multi-agent policy gradients

## 介绍

> [Counterfactual multi-agent policy gradients](https://arxiv.org/abs/1705.08926)

许多现实世界的问题，如网络数据包路由和自治车辆的协调，自然被建模为协作多智能体系统。非常需要新的强化学习方法，能够有效地学习这些系统的分散策略。为此，我们提出了一种新的多代理行为者-批评者方法，称为counterfactual multi-agent\(COMA\)policy gradient。COMA使用集中评论家来评估Q-函数和分散的参与者，以优化代理的策略。 此外，为了应对多智能体信用分配的挑战，它使用了一个counterfactual baseline，即边缘化一个代理人的行为，同时保持其他代理人的行为不变。COMA还使用了一种批评表示，它可以在一次正向传递中有效地计算counterfactual baseline。我们使用具有显著部分可观测性的分散变量在测试平台StarCraft unit micromanagement上评估COMA。在这种情况下，COMA显著提高了与其他多代理参与者-批评者方法相比的平均性能，并且最佳性能代理与能够访问完整状态的最先进的集中式控制器竞争。

## 方法





