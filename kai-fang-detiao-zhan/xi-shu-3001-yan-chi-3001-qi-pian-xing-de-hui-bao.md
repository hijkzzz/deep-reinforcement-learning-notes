# 稀疏、延迟、欺骗性的回报

Montezuma's Revenge等以稀疏奖励为特征的游戏仍然是大多数Deep RL方法的挑战。虽然结合DQN内在动机或专家演示，的最新进展可以提供帮助，但是具有稀疏奖励的游戏仍然是当前深度RL方法的挑战。内在动机的强化学习，以及分层强化学习方面有着悠久的研究历史，这在这里可能是有用的。基于Minecraft的Project Malmo环境提供了一个很好的场所，可以创建具有非常稀疏奖励的任务，代理需要设置他们的目标。无导数和无梯度方法，如进化策略和遗传算法，通过局部采样来探索参数空间，在这些游戏中有着广阔的应用前景，特别是结合novelty search。
