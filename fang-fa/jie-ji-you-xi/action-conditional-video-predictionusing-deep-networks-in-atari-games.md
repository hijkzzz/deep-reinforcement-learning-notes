# Action-Conditional Video Prediction

## 方法

> [Action-Conditional Video Predictionusing Deep Networks in Atari Games](http://papers.nips.cc/paper/5859-action-conditional-video-prediction-using-deep-networks-in-atari-games.pdf)

受基于视觉的强化学习问题\(特别是最近基准 ALE 中的Atari游戏\)的驱动，我们考虑未来图像帧依赖于控制变量或动作以及先前帧的时空预测问题。虽然雅达利游戏中的画面不是由自然场景构成的，但它在尺寸上是高维的，可以包含数十个物体，其中一个或多个物体被动作直接控制，许多其他物体受到间接影响，可能涉及物体的进入和分离，也可能涉及深度部分可观察性。我们提出并评估了两种基于卷积神经网络和递归神经网络的深层神经网络结构，包括编码层、动作条件变换层和解码层。实验结果表明，所提出的体系结构能够生成视觉上真实的帧，这些帧对于在某些游戏中控制大约100步动作条件预期也是有用的。据我们所知，本文是第一个对控制输入条件下的高维视频进行长期预测和评估的论文。

## 方法

![](../../.gitbook/assets/image-14.png)

我们架构的目标是学习一个函数： $$f : \mathbf{x}_{1 : t}, \mathbf{a}_{t} \rightarrow \mathbf{x}_{t+1}$$，其中 $$x$$ 是帧， $$a$$ 是动作。上图即我们提出的两种帧预测网络架构。每个编码层由输入帧提取时空特征，动作条件变换层通过引入动作变量作为附加输入，将编码特征转换为高级特征空间中下一帧的预测，最后解码映射预测的高层特征转换为输出像素。

### encoding

Feedforward encoding 将先前帧的固定历史作为输入，通过通道连接\(图1a \)，堆叠卷积层从连接的帧中直接提取时空特征。

$$
\mathbf{h}_{t}^{e n c}=\mathrm{CNN}\left(\mathbf{x}_{t-m+1 : t}\right)
$$

其中 $$\mathbf{x}_{t-m+1 : t} \in \mathbb{R}(m \times c) \times h \times w$$

Recurrent encoding

$$
\left[\mathbf{h}_{t}^{e n c}, \mathbf{c}_{t}\right]=\operatorname{LSTM}\left(\mathrm{CNN}\left(\mathrm{x}_{t}\right), \mathrm{h}_{t-1}^{e n c}, \mathbf{c}_{t-1}\right)
$$

### transformation

我们使用编码特征向量和控制变量之间的乘法交互

$$
h_{t, i}^{d e c}=\sum_{j, l} W_{i j l} h_{t, j}^{e n c} a_{t, l}+b_{i}
$$

使用因子分解近似

$$
\mathbf{h}_{t}^{d e c}=\mathbf{W}^{d e c}\left(\mathbf{W}^{e n c} \mathbf{h}_{t}^{e n c} \odot \mathbf{W}^{a} \mathbf{a}_{t}\right)+\mathbf{b}
$$

### decoding

$$
\hat{\mathbf{x}}_{t+1}=\text { Deconv (Reshape }\left(\mathbf{h}^{d e c}\right) )
$$

训练方式使用1-step到n-step的课程学习

