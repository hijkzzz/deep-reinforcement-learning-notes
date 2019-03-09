# ES

## 介绍

> [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf)

我们探索使用Evolution Strategies（ES），一类黑盒优化算法，作为基于MDP的流行RL技术的替代方案，如Q-learning和Policy Gradients。 MuJoCo和Atari上的实验表明，ES是一种可行的解决方案策略，可以根据可用的CPU数量进行非常好的扩展：通过使用基于常见随机数的新型通信策略，我们的ES实现只需要传达标量，使其可以扩展到超过一千个并行进程。 这使我们能够在10分钟内解决3D人体行走，并在训练一小时后获得大多数Atari游戏的竞争结果。 此外，我们强调了作为黑盒优化技术的ES的几个优点：它对动作频率和延迟奖励不变，容忍极长的视野，并且不需要temporal discounting或值函数逼近。

## 算法



