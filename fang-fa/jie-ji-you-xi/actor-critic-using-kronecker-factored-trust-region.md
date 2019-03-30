# ACKTR

## 介绍

> [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/pdf/1708.05144.pdf)

在这项工作中，我们建议用最近提出的Kronecker-factored approximation curvature将信赖域优化应用于深度强化学习。我们扩展了自然策略梯度的框架，提出了利用带信任区域的Kronecker-factored approximation curvature\(K-FAC\)来优化actor和critic；因此，我们此算法其称为Actor Critic using Kronecker-Factored Trust Region \(ACKTR\)，据我们所知，这是Actor-Critic方法中第一个可扩展的信任区域自然梯度方法。它也是一种直接从原始像素输入中学习连续控制中的非平凡任务以及离散控制策略的方法。我们在Atari游戏中的离散域以及MuJoCo环境中的连续域中测试了我们的方法。使用所提出的方法，我们能够获得更高的回报，并比之前的最佳的on-policy的actor-critic样本效率平均提高2 - 3倍。

## 算法

### Natural gradient using Kronecker-factored approximation

假设神经网络输出一个分布 $$f(\theta)$$ ，为了最小化该分布相关的目标函数 $$\mathcal{J}(\theta)$$ ，最速下降法在有约束 $$\|\Delta \theta\|_{B}<1$$ 的情况下计算一个 $$\Delta \theta$$ 使得 $$\mathcal{J}(\theta+\Delta \theta)$$ 具有最小，其中 $$\|\cdot\| B$$ 是一个范数，即 $$\|x\|_{B}=\left(x^{T} B x\right)^{\frac{1}{2}}$$ （ $$B$$ 是一个半正定矩阵）。可以得到这个问题的解是： $$\Delta \theta \propto-B^{-1} \nabla_{\theta} \mathcal{J}$$ 。当$$B$$ 是单位矩阵 $$I$$ （即欧式范数）时，此方法又被称为梯度下降。然而这种欧式范数是依赖于参数 $$\theta$$ 的，即它体现的不是分布的距离而是参数 $$\theta$$ 的距离，如下图：

![](../../.gitbook/assets/image-106.png)

上下两图中，左右分布的参数距离均为2，但是他们的分布距离却截然不同，即上图重叠区小，下图重叠区大。因为这个范数是参数 $$\theta$$ 相关的，所以参数 $$\theta$$ 会影响优化的轨迹，这是不合理的，实际上应当只允许分布影响优化轨迹。

而费雪信息矩阵 $$F$$ 是 $$KL$$ 散度的二阶近似，他是独立于参数 $$\theta$$的，只与分布有关，所以利用费雪信息矩阵构建上面的范数约束，可以使得优化更加稳定和有效，这也被称为自然梯度。然而费雪矩阵的求逆是一个复杂所以不实际的操作，因此我们必须使用某种近似方法。

Kronecker-factored approximate curvature \(K-FAC\)就是这样的一种方法。假设 $$p(y | x)$$ 是神经网络拟合的分布， $$L=\log p(y | x)$$ 即其似然函数。定义 $$W \in \mathbb{R}^{C_{\text {out}} \times C_{i n}}$$ 是神经网络第L层的权重参数，且 $$a \in \mathbb{R}^{C_{i n}}$$ 是L层的输入，有输出 $$s=W a$$ 。根据矩阵求导术可以得到标准梯度 $$\nabla_{W} L=\left(\nabla_{s} L\right) a^{\top}$$ ,K-FAC使用下面的近似方法计算神经网络第L层参数的费雪信息矩阵：

$$\begin{aligned} F_{\ell} &=\mathbb{E}\left[\operatorname{vec}\left\{\nabla_{W} L\right\} \operatorname{vec}\left\{\nabla_{W} L\right\}^{\top}\right]=\mathbb{E}\left[a a^{\top} \otimes \nabla_{s} L\left(\nabla_{s} L\right)^{\top}\right] \\ & \approx \mathbb{E}\left[a a^{\top}\right] \otimes \mathbb{E}\left[\nabla_{s} L\left(\nabla_{s} L\right)^{\top}\right] :=A \otimes S :=\hat{F}_{\ell} \end{aligned}$$

其中$$\otimes$$ 是Kronecker product： $$\mathbf{A} \otimes \mathbf{B}=\left[ \begin{array}{ccc}{a_{11} \mathbf{B}} & {\cdots} & {a_{1 n} \mathbf{B}} \\ {\vdots} & {\ddots} & {\vdots} \\ {a_{m 1} \mathbf{B}} & {\cdots} & {a_{m n} \mathbf{B}}\end{array}\right]$$

又根据Kronecker product的性质 $$(P \otimes Q)^{-1}=P^{-1} \otimes Q^{-1} \text { and }(P \otimes Q) \operatorname{vec}(T)=P T Q^{\top}$$ ，可得自然梯度近似公式

$$\operatorname{vec}(\Delta W)=\hat{F}_{\ell}^{-1} \operatorname{vec}\left\{\nabla_{W} \mathcal{J}\right\}=\operatorname{vec}\left(A^{-1} \nabla_{W} \mathcal{J} S^{-1}\right)$$

[从流形的角度理解自然梯度](https://www.cnblogs.com/tiny-player/p/3323973.html)

[K-FAC的近似理论证明](https://arxiv.org/pdf/1503.05671.pdf)

### Natural gradient in actor-critic

本节介绍如何将自然梯度引入 actor-critic 算法中。actor网络的费雪信息矩阵如下：

$$F=\mathbb{E}_{p(\tau)}\left[\nabla_{\theta} \log \pi\left(a_{t} | s_{t}\right)\left(\nabla_{\theta} \log \pi\left(a_{t} | s_{t}\right)\right)^{\top}\right]$$

其中 $$p(\tau)$$是样本轨迹的分布，即 $$p\left(s_{0}\right) \prod_{t=0}^{T} \pi\left(a_{t} | s_{t}\right) p\left(s_{t+1} | s_{t}, a_{t}\right)$$ 。

而对于标准的critic网络，其输出值是一个标量而非分布，无法定义费雪信息矩阵，所以我们引入高斯分布来解决这个问题：假设critic的输出由 $$p(v | s_{t}) \sim \mathcal{N}\left(v ; V\left(s_{t}\right), \sigma^{2}\right)$$ 定义，于是我们便可以基于高斯分布定义相关的费雪信息矩阵。

如果actor和critic共用一个网络，我们假设网络的输出是一个联合分布 $$p(a, v | s)=\pi(a | s) p(v | s)$$ ，然后定义费雪信息矩阵为：

$$\mathbb{E}_{p(\tau)}\left[\nabla \log p(a, v | s) \nabla \log p(a, v | s)^{T}\right]$$

然后同步更新actor和critic

### Step-size Selection and trust-region optimization

对于随机梯度下降，参数的更新方式为$$\theta \leftarrow \theta-\eta F^{-1} \nabla_{\theta} L$$ ，但是在RL的环境中，有时候会出现大的更新步伐，导致算法过早收敛到接近确定性的策略。所以就出现了TRPO这种信任区域更新的方法，这里我们选择 $$\min \left(\eta_{\max }, \sqrt{\frac{2 \delta}{\Delta \theta \tau F \Delta \theta}}\right)$$ 作为学习速率 $$\eta$$ ，其中 $$\delta$$ 是半径超参。

