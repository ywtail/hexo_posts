---
title: 'TensorFlow (3): 多层感知机识别手写数字'
date: 2017-06-03 21:05:12
tags: [TensorFlow,MachineLearning,python]
categories: TensorFlow
top: 2
mathjax: true
---

**本文内容主要来自图书：TensorFlow实战 / 黄文坚，唐源著**
在 [TensorFlow (2): Softmax Regression识别手写数字](http://ywtail.github.io/2017/06/02/TensorFlow-2-Softmax-Regression%E8%AF%86%E5%88%AB%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97/) 中使用 TensorFlow 实现了`Softmax Regression` (无隐含层)，并在 `MNIST` 数据集上取得了 92% 的正确率。在这里将给神经网络加上隐含层，使用 TensorFlow 实现多层感知机，并对 `MNIST` 数据集中的手写数字进行识别。
实现多层感知机中使用了 `Dropout`、`Adagrad`、`ReLU` 等辅助性组件。

### 多层感知机
`Softmax Regression` 和传统意义上的神经网络的最大区别是没有隐含层。本文实现的多层感知机实际上是在 `Softmax Regression` 的基础上加上一个隐含层。
隐含层是神经网络的一个重要概念，它是指除输入、输出层外，中间的那些层。输入层和输出层是对外可见的，通常也被称作可视层，而中间层不直接暴露出来，是模型的黑箱部分，通常也比较难具有可解释性，所以一般被称作隐含层。
理论上只要隐含层节点足够多，即使只有一个隐含层的神经网络也可以拟合任意函数。同时，隐含层越多，越容易拟合复杂函数。有理论研究表明，拟合复杂函数需要的隐含节点的数目，基本上随着隐含层的数量增多而呈指数下降趋势。即层数越多，神经网络需要的隐含节点可以越少。这也是深度学习的特点之一，层数越深，概念越抽象，神经网络隐含节点就越少。
不过在实际应用中，使用层数较深的神经网络会遇到许多困难，例如：过拟合、参数难以调试、帝都弥散等。这些问题需要很多策略来解决，在最近几年的研究中，越来越多的方法来帮助我们解决问题，例如：`Dropout`、`Adagrad`、`ReLU`等。

### Dropout
过拟合是机器学习中一个常见的问题，尤其是在神经网络中，由于参数众多，非常容易出现过拟合。为了解决这个问题，`Hinton` 教授团队提出了一个思路非常简单但是非常有效的方法 —— `Dropout`。
它的大致思路是 **在训练时，将神经网络某一层的输出节点数据随机丢弃一部分**。
可以理解为随机把一张图片的 50% 的点删掉，此时人依然很可能识别这张图片，机器也一样。这种做法实质上相当于随机创造了很多新的样本，通过增大样本量、减少特征数来防止过拟合。
`Dropout` 其实也相当于是一种 `Bagging` 方法，可以理解成每次丢弃节点数据是对特征的一种采样。相当于我们训练了一个 `ensemble` 的神经网络模型，对每个样本都做特征采样，只不过没有训练多个神经网络模型，只有一个融合的神经网络。

### Adagrad
参数难以调试是神经网络的另一大难点，尤其是随机梯度下降（`Stochastic Gradient Descent`，`SGD`）的参数，对 `SGD` 设置不同的学习速率，最后得到的结果可能差异巨大。神经网络通常不是一个凸优化的问题，它处处充满了局部最优。`SGD` 本身不是一个比较稳定的算法，结果可能会在最优解附近波动，而不同的学习速率可能导致神经网络落入截然不同的局部最优之中。
对 `SGD`，一开始我们希望学习速率大一些，可以加速收敛，但训练的后期又希望学习速率可以小一点，这样可以比较稳定地落入一个局部最优解。
不同的机器学习所需要的学习速率也不太好设置，需要反复调试，因此就有像 `Adagrad`、`Adam`、`Adadelta` 等自适应的方法可以减轻调试参数的负担。对于这些优化算法，通常我们使用它默认的参数设置就可以取得一个比较好的效果。

### ReLU
梯度弥散（`Gradient Vanishment`）是另一个影响神经网络训练的问题，在 `ReLU` 激活函数出现之前，神经网络的训练全部是用 `Sigmoid` 作为激活函数。这可能是因为 `Sigmoid` 函数具有限制性，输出数值在 0~1，最符合概率输出的定义。
非线性的 `Sigmoid` 函数在信号的特征空间映射上，对中央区的信号增益较大，对两侧区的信号增益小。因而在神经网络训练时，可以将重要特征置于中央区，而非重要的特征置于两侧区。
但是当神经网络层数较多时，`Sigmoid` 函数在反向传播中梯度值会逐渐减小，经过多层的传递后会呈指数级急剧减小，因此梯度值在传递到前面几层时就会变得非常小了。在这种情况下，根据训练数据的反馈来更新神经网络参数将会非常缓慢，基本起不到训练的作用。
直到 `ReLU` 出现，才比较完美地解决了梯度弥散的问题。
**`ReLU` 是一个简单的非线性函数 y=max(0,x) ，它在坐标轴上是一条折线，当 x <= 0 时， y=0；当 x > 0 时，y = x。**
`ReLU` 非常类似于人脑的阈值响应机制，信号在超过某个阈值时，神经元才会进入兴奋和激活的状态，平时则出于抑制状态。
`ReLU` 可以很好地传递梯度，经过多层的反向传播，梯度依旧不会大幅缩小，因此非常适合训练很深的神经网络。`ReLU` 从正面解决了梯度弥散的问题，而不需要通过无监督的逐层训练初始化权重。`ReLU` 对比 `Sigmoid` 的主要变化有如下三点：
1. 单侧抑制
2. 相对宽阔的兴奋边界
3. 稀疏激活性

目前，`ReLU` 及其变种（`EIU`，`PReLU`，`RReLU`）已经成为了最主流的激活函数。实践中大部分情况下（包括 `MLP` 和 `CNN`，`RNN` 内部主要还是使用 `Sigmoid`、`Tanh`、`Hard Sigmoid`）将隐含层的激活函数从 `Sigmoid` 替换为 `ReLU`都可以代练训练速度及模型准确率的提升。当然神经网络的输出层一般还是 `Sigmoid` 函数，因为它最接近概率输出分布。

### TensorFlow 实现机器学习算法通用步骤
使用 `TensorFlow` 实现了简单的机器学习算法整个流程可以分为4个部分：
>1. 定义算法公式，也就是神经网络的 `forward` 时的计算
2. 定义 `loss`，选定优化器（这里选的是梯度下降），并指定优化器优化 `loss`
3. 迭代地对数据进行训练
4. 在测试集或验证集上对准确率进行测评

以上几个步骤是使用 `TensorFlow` 进行算法设计、训练的核心步骤。
`TensorFlow` 和 `Spark` 类似，我们定义的各个公式其实只是 `Computation Graph`，在执行这行代码时，计算还没有实际发生，只有等调用 `run` 方法，并 `feed` 数据时计算才真正执行。比如 `cross_entropy`、`train_step`、`accuracy` 等都是计算图中的节点，而不是数据结果，我们可以通过调用 `run` 方法执行这些结点或者说运算操作来获取结果。

### 实现多层感知机
#### 加载数据
首先加载 `MNIST` 数据集，并创建一个 `TensorFlow` 默认的 `InteractiveSession`，这样在后续的各项操作中就无需指定 `Session` 了。
```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
sess=tf.InteractiveSession()
```
#### 定义各个参数
- 这里 `in_units` 是输入节点数，`h1_units` 是隐含层的输出节点数，在这里设置为 300。
- `W1`、`b1` 是隐含层的权重和偏置，这里将偏置赋值为 0，并将权重初始化为截断的正态分布，其标准差为 0.1。
  因为模型使用的激活函数是 `ReLU`，所以需要使用正态分布给参数加一点噪声来打破完全对称并且避免 0 梯度。
  在其他的一些模型中，有时还需要给偏置赋上一些小的非零值来避免 `dead neuron`（死亡神经元）。
- 对于输出层的 `Softmax`，直接将权重 `W2` 和偏置 `b2` 全部初始化为 0 即可（对于 `Sigmoid`，在 0 附近最敏感、梯度最大）

```python
in_units=784
h1_units=300
W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))
```

定义输入 `x`，由于 `Dropout` 的比率 `keep_prob` 是变化的（训练时小于1，预测时等于1），所以也定义成一个 `placeholder`。
```python
x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)
```

#### 定义模型结构
- 隐含层命名为 `hidden1`，激活函数为 `ReLU`，所以这个隐含层的计算公式就是 $y=relu(W1x+b1)$
- 接下来调用 `tf.nn.dropout` 实现 `Dropout` 功能，这里的 `keep_prob` 是保留数据的比例，在训练时应小于 1，用以制造随机性，防止过拟合；在预测时应等于 1，即使用全部特征来预测样本的类别。
- 最后是输出层，这一层依旧使用 `softmax` 作激活函数。

```python
hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
```
#### 定义损失函数和选择优化器
目前已经完成第一步：定义计算公式，即神经网络的 `forward` 计算。
接下来进行第 2 步：定义损失函数和选择优化器来优化`loss`，这里的损失函数使用交叉信息熵，优化器选择自适应的优化器 `Adagrad`，并把学习速率设为 0.01.

```python
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)
```
#### 训练模型
这里加入了 `keep_prob`，在训练时设置为 0.75。一般来说，对越复杂越大规模的神经网络，`Dropout` 的效果越显著。
为了达到一个比较好的效果，一共采用 5000 个 `batch`，每个 `batch` 包含 100 条的样本，一共 50 万的样本，相当于对全数据及进行了 9 轮（`epoch`）迭代。

```python
tf.global_variables_initializer().run()
for i in range(5000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})
```
#### 对模型进行准确率测评
在这一步 `keep_prob` 设置为 1，这样可以达到模型最好的预测效果。

```python
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
```

### 总结
最终达到了 97% 的准确度，相比之前的 `Softmax`， 误差率由最初的 8% 下降到 3%，这个提升仅靠增加一个隐含层就实现了，可见多层神经网络的效果是十分显著的。当前，其中也使用了一些 Trick 进行辅助，例如 `Dropout`、`Adagrad`、`ReLU`等，但起决定性作用的还是隐含层本身，它能对特征进行抽象和转化。


### 完整的代码及运行结果
```python
# coding: utf-8
# 多层感知机识别手写数字
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载 MNIST 数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()


# 定义各个参数
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)


# 定义模型结构
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)


# 定义损失函数和选择优化器来优化loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)


# 训练模型
tf.global_variables_initializer().run()
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})


# 对模型进行准确率测评
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
```

输出：
```
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
0.97
```

