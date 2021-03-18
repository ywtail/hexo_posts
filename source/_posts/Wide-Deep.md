---
title: Wide&Deep
date: 2021-03-18 20:27:39
tags: [DeepRecommendationModel]
categories:
---

Wide & Deep 是 Google 发表在 DLRS 2016 上的《[Wide & Deep Learning for Recommender System](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1606.07792)》。Wide & Deep 模型的核心思想是结合线性模型的记忆能力和 DNN 模型的泛化能力，从而提升整体模型性能。Wide & Deep 已成功应用到了 Google Play 的app推荐业务，并于TensorFlow中封装。该结构被提出后即引起热捧，在业界影响力非常大，很多公司纷纷仿照该结构并成功应用于自身的推荐等相关业务。

## 模型

Wide&Deep模型的初衷是融合高阶和低阶特征。原始模型里面：wide部分是特征工程+LR，deep部分是MLP(多层感知器（Multilayer Perceptron,缩写MLP）)。
Wide&Deep是一类模型的统称，将LR换成FM同样也是一个Wide&Deep模型。

后续的改进如下：

- DCN/DeepFM是对wide部分的改进，将LR替换为CN，FM等。
- NFM是对Deep部分的改进，加入特征交叉层Bi-interaction。

本文讨论最原始的模型，结构如下

<img src="/images/Wide-Deep_1.jpg" width = "600" height = "500" alt="Wide-Deep_1" align=center/>

Wide&Deep全文围绕着“**记忆”(Memorization)**与“**扩展(Generalization)**”两个词展开。实际上，它们在推荐系统中有两个更响亮的名字，Exploitation & Exploration，即著名的EE问题。

- **记忆（memorization）即从历史数据中发现item或者特征之间的相关性；哪些特征更重要——Wide部分**。
- **泛化（generalization）即相关性的传递，发现在历史数据中很少或者没有出现的新的特征组合；——Deep部分**。

在推荐系统中，记忆体现的准确性，而泛化体现的是新颖性。

### Wide

wide部分是一个广义的线性模型，输入的特征主要有两部分组成，一部分是原始的部分特征，另一部分是原始特征的交叉特征(cross-product transformation)。

Wide侧记住的是历史数据中那些**常见、高频**的模式。Wide侧没有发现新的模式，只是学习到这些模式之间的权重，做一些模式的筛选。正因为Wide侧不能发现新模式，因此我们需要**根据人工经验、业务背景，将我们认为有价值的、显而易见的特征及特征组合，喂入Wide侧**。

![[公式]](https://www.zhihu.com/equation?tex=y%3D%5Cbm%7Bw%7D%5ET%5B%5Cbm%7Bx%7D%2C%5Cphi%28%5Cbm%7Bx%7D%29%5D%2Bb)

### Deep

Deep部分是一个DNN模型，输入的特征主要分为两大类，一类是数值特征(可直接输入DNN)，一类是类别特征(需要经过Embedding之后才能输入到DNN中)

Deep侧，通过embedding+深层交互，能够学交叉特征。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbm%7Ba%7D%5E%7Bl%2B1%7D%3Df%28%5Cbm%7BW%7D%5El+%5Cbm%7Ba%7D%5El+%2B+%5Cbm%7Bb%7D%5El+%29)

###Wide&Deep

W&D模型是将两部分输出的结果结合起来联合训练，将deep和wide部分的输出重新使用一个逻辑回归模型做最终的预测，输出概率值。

**损失函数** 模型选取logistic loss作为损失函数，此时Wide & Deep最后的预测输出为：

![[公式]](https://www.zhihu.com/equation?tex=p%28y%3D1%7C%5Cbm%7Bx%7D%29%3D%5Csigma%28%5Cbm%7Bw%7D%5E%7BT%7D_%7Bwide%7D%5B%5Cbm%7Bx%7D%2C%5Cphi%28%5Cbm%7Bx%7D%29%5D%2B%5Cbm%7Bw%7D%5E%7BT%7D_%7Bdeep%7D%5Cbm%7Ba%7D%5E%7Bl_f%7D%2Bb%29)

## 优缺点

### 优点

1. 简单有效。结构简单易于理解，效果优异。目前仍在工业界广泛使用，也证明了该模型的有效性。
2. 结构新颖。使用不同于以往的线性模型与DNN串行连接的方式，而将线性模型与DNN并行连接，同时兼顾模型的Memorization与Generalization。

### 缺点

Wide侧的特征工程仍无法避免。

## 参考

- [详解 Wide & Deep 结构背后的动机]()
- https://github.com/datawhalechina/team-learning-rs/blob/master/DeepRecommendationModel/Wide%26Deep.md
- [看Google如何实现Wide & Deep模型(1)](https://zhuanlan.zhihu.com/p/47293765)
- [推荐系统CTR实战——Wide & Deep](https://fuhailin.github.io/Wide-Deep/)

