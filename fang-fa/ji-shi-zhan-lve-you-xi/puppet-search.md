# Puppet Search

## 介绍

> [Puppet Search: Enhancing Scripted Behavior by Look-Ahead Search with Applications to Real-Time Strategy Games](https://pdfs.semanticscholar.org/3902/521ee719f296423c33e79f99949af1c7445b.pdf)

实时策略游戏已经被证明非常适合标准的对抗树搜索技术。最近，出现了几种解决它们复杂性的方法，它们使用游戏状态或移动抽象，或者两者兼有。不幸的是，支持性实验要么局限于更简单的即时战略环境\(μRTS，SparCraft\)，要么缺乏对最先进的游戏代理的测试。

在这里，我们提出了Puppet Search，一种基于脚本的新的对抗性搜索框架，可以将选择点暴露给前瞻搜索程序。为其选择点选择script和decision的组合表示接下来要应用的动作。这种移动可以在实际游戏中执行，从而让脚本运行，或者在游戏状态的抽象表示中执行，该抽象表示可以被敌对搜索算法使用。Puppet Search返回代理在特定时间内要执行的脚本和选项的主要变化。我们在一个完整的星际争霸机器人中实现了这个算法。实验表明，它与2014年亚投行星际争霸赛中最先进的机器人对战时使用的所有独立脚本都匹配或优于它们





