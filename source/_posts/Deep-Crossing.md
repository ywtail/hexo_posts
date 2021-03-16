---
title: Deep Crossing
date: 2021-03-16 20:49:38
tags: [DeepRecommendationModel]
categories:
---

Deep Crossing模型是由微软研究院在论文[《Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features》](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)中提出的，它主要是用来解决大规模特征自动组合问题，从而减轻或者避免手工进行特征组合的开销。Deep Crossing可以说是深度学习CTR模型的最典型和基础性的模型。

## 模型

 <img src="/images/Deep_Crossing_1.jpg" width = "500" height = "400" alt="Deep_Crossing" align=center/>

模型的输入是一系列的独立特征，模型总共包含4层，分别是Embedding层、Stacking层、Residual Unit层、Scoring层，模型的输出是用户点击率预测值。
注意上图中红色方框部分，输入特征没有经过Embedding层就直接连接到了Stacking层了。这是因为输入特征可能是稠密的也可能是稀疏的，论文中指出，对于维度小于256的特征直接连接到Stacking层。



### 损失函数

论文中使用的是交叉熵损失函数，但是也可以使用Softmax或者其他损失函数：

![[公式]](https://www.zhihu.com/equation?tex=logloss%3D-%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28y_ilog%28p_i%29%2B%281-y_i%29log%281-p_i%29%29)



### Embedding层

将高维稀疏特征转化为低维稠密特征，公式如下：

![[公式]](https://www.zhihu.com/equation?tex=X_j%5EO%3Dmax%280%2CW_jX_j%5EI%2Bb_j%29)

### Stacking层

将特征聚合起来，形成一个向量

![[公式]](https://www.zhihu.com/equation?tex=X%5EO%3D%5BX_0%5EO%2CX_1%5EO%2C%5Cdots%2CX_K%5EO%5D)

### Residual层

Deep Crossing模型使用稍微修改过的残差单元，它不使用卷积内核，改为了两层神经网络。

残差层是由下图所示的残差单元构建成的。残差单元如下所示：

<img src="/images/Deep_Crossing_2.jpg" width = "500" height = "400" alt="Deep_Crossing_2" align=center/>

公式定义为：

![[公式]](https://www.zhihu.com/equation?tex=X%5EO%3DF%28X%5EI%2C%5C%7BW_0%2CW_1%5C%7D%2C%5C%7Bb_0%2Cb_1%5C%7D%29%2BX%5EI)

将X<sup>I</sup>移项到等式左侧，可以看出 F函数拟合的是输入与输出之间的残差。对输入进行全连接变换之后，经过relu激活函数送入第二个全连接层，将输出结果与原始输入进行 element-wise add 操作，再经过relu激活输出

### Scoring层

Residual层的输出首先连接到全连接层，其次再经过Sigmoid激活函数，最后输出的是一个广告的预测点击率。



## 参考

- [datawhalechina github](https://github.com/datawhalechina/team-learning-rs/blob/master/DeepRecommendationModel/DeepCrossing.md)
- 知乎论文介绍：https://zhuanlan.zhihu.com/p/91057914
- 简书：https://www.jianshu.com/p/e1873e9a97ad
- tf实现：https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/Deep_Crossing