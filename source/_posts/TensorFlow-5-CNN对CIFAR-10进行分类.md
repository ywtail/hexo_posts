---
title: 'TensorFlow (5): CNN对CIFAR-10进行分类'
date: 2017-06-06 19:30:29
tags: [TensorFlow,MachineLearning,python]
categories: TensorFlow
top: 2
---

本文将逐步实现一个稍微复杂一些的卷积网络，简单的 `MNIST` 数据集已经不适合用来评测其性能，在这里我们将使用 `CIFAR-10` 数据集来进行训练。
本文的结构安排如下
- 介绍 `CIFAR-10` 数据集
- 实现 CNN
- 在网络中加入 LRN
- 对权重进行 L2 正则化
- 总结和对比
- 给出完整代码和运行结果
- 列出参考资料

本文中涉及的所有代码均在 [github.com/ywtail](https://github.com/ywtail/TensorFlow/tree/master/5_CNN%E5%AF%B9CIFAR-10%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB) 中。

### CIFAR-10
本文使用的数据集是 `CIFAR-10`。这是一个经典的数据集，包含 60000 张 32x32 的彩色图像，其中训练集是 50000 张，测试集 10000 张。`CIFAR-10` 如同其名，一共标注为 10 类，每一类图片 6000 张，这 10 类分别是 `飞机， 汽车， 鸟， 猫， 鹿， 狗， 青蛙， 马， 船， 卡车`，其中没有任何重叠的情况，也不会在一张图片中同时出现两类物体。它还有一个兄弟版本 `CIFAR-100`，其中标注了 100 类。

`CIFAR-10` 数据集非常通用，对 `CIFAR-10` 数据集的分类是机器学习中一个公开的基准测试问题，其任务是对一组 32x32 RGB 的图像进行分类。许多论文中都在这个数据集上进行了测试，目前 `state-of-the-art` 的工作已经可以达到 3.5% 的错误率了，但是需要训练很久，即使在 GPU 上也需要十几个小时。据深度学习三巨头之一 LeCun 说，现有的卷积神经网络已经可以对 `CIFAR-10` 进行很好的学习，这个数据集的问题已经解决了。

### 实现 CNN
在这里我们要实现的 CNN 网络结构如下表

| Layer 名称 | 描述            |
| :------: | :------------ |
|  conv1   | 卷积层，ReLU激活函数  |
|  pool1   | 最大池化          |
|  conv2   | 卷积层，ReLU激活函数  |
|  pool2   | 最大池化          |
|  local3  | 全连接层，ReLU激活函数 |
|  local4  | 全连接层，ReLU激活函数 |
|  logits  | 模型的输出         |


#### 下载 TesorFlow Models 库
首先下载 TesorFlow Models 库，以便使用其中提供的 `CIFAR-10` 数据的类。在本文中构建模型的过程中，实际上只使用了 `cifar10_input.py` 和 `cifar10.py` 这两个文件。
```bash
git clone https://github.com/tensorflow/models.git
```
下载完后是一个名为 `models` 的文件夹，代码位于 `models/image/cifar10/`，文件组织结构如下

| 文件                                       | 作用                       |
| ---------------------------------------- | ------------------------ |
| [`cifar10_input.py`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_input.py) | 读取本地CIFAR-10的二进制文件格式的内容。 |
| [`cifar10.py`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10.py) | 建立CIFAR-10的模型。           |
| [`cifar10_train.py`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_train.py) | 在CPU或GPU上训练CIFAR-10的模型。  |
| [`cifar10_multi_gpu_train.py`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py) | 在多GPU上训练CIFAR-10的模型。     |
| [`cifar10_eval.py`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_eval.py) | 评估CIFAR-10模型的预测性能。       |


可以通过 `cd models/tutorials/image/cifar10/` 在cifar10 文件夹下编写代码，也可以只将 cifar10 中的  `cifar10_input.py` 和 `cifar10.py` 拷贝出来。

#### 加载数据
这里需要调用下载的 `cifar10` 和 `cifar10_input` 类来对数据进行下载和处理，得到训练数据和测试数据。
```python
# 载入需要用的库
import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import math
import time

data_dir = 'cifar10_data/cifar-10-batches-bin'  # 下载 CIFAR-10 的默认路径
cifar10.maybe_download_and_extract()  # 下载数据集，并解压、展开到其默认位置
```

接下来使用 `cifar10_input` 类中的 `distorted_inputs` 函数产生训练需要使用的数据，包括特征及其对应的 label，这里返回的是已经封装好的 tensor，每次执行都会生成一个 batch_size 的数量的样本。 batch_size 需要作为参数输入，所以先设定 batch_size，并使用 `distorted_inputs` 产生训练需要使用的数据。
>`distorted_inputs` 对数据进行了 `Data Augmentation`（数据增强），即采取了一系列随机变换的方法来人为的增加数据集的大小：
- 对图像进行随机的左右翻转；
- 随机变换图像的亮度；
- 随机变换图像的对比度；

>通过这些操作可以获取更多的样本（带噪声的），原来的一张图片可以变为多张图片，相当于扩大样本量，对提高准确率非常有帮助。
>从磁盘上加载图像并进行变换需要花费不少的处理时间。为了避免这些操作减慢训练过程，我们在 16 个独立的线程中并行进行这些操作，这16个线程被连续的安排在一个 TensorFlow 队列中（在训练过程中会启动线程队列）。

```python
batch_size=128
images_train,labels_train=cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
```
再使用 `cifar10_input.inputs` 生成测试数据，这里不需要进行太多处理，不需要对图片进行翻转或者修改亮度、对比度，不过需要剪裁图片正中间的 24x24 大小的区块，并进行数据标准化操作。
```python
images_test,labels_test=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)
```
到此为止数据加载完毕，下面进行参数设置。

#### 参数设置
有些数据需要多次使用，所以写成函数。
```python
def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev)  # stddev=stddev！！！
    return tf.Variable(initial)

def bias_variable(cons, shape):
    initial = tf.constant(cons, shape=shape)  # 必须是 shape=shape
    return tf.Variable(initial)

def conv(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
```

#### 模型实现
首先创建输入数据的 `placeholder`，在设定时需要注意，因为 `batch_size` 在之后定义的网络结构时被用到了，所以数据尺寸中的第一个值（样本条数）需要被预先设定，而不是像之前那样设定为 None。而数据尺寸中的图片尺寸为剪裁后的 24x24，因为图片是彩色 RGB 图片，所以通道数为 3。
```python
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一层
weight1 = weight_variable([5, 5, 3, 64], 5e-2)
bias1 = bias_variable(0.0, [64])

conv1 = tf.nn.relu(conv(image_holder, weight1) + bias1)
pool1 = max_pool_3x3(conv1)

# 第二层
weight2 = weight_variable([5, 5, 64, 64], 5e-2)
bias2 = bias_variable(0.1, [64])

conv2 = tf.nn.relu(conv(pool1, weight2) + bias2)
pool2 = max_pool_3x3(conv2)

reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value

# 全连接层
weight3 = weight_variable([dim, 384], 0.04)
bias3 = bias_variable(0.1, [384])

local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 全连接层
weight4 = weight_variable([384, 192], 0.04)
bias4 = bias_variable(0.1, [192])

local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 输出
weight5 = weight_variable([192, 10], 1 / 192.0)
bias5 = bias_variable(0.0, [10])
logits = tf.matmul(local4, weight5) + bias5
```

#### 损失函数
依然使用 `cross_entropy` 作为损失函数，不同的是在这里将 `cross_entropy` 的计算和 `softmax` 的计算混合在了一起，使用 `tf.nn.sparse_softmax_cross_entropy_with_logits`。然后将 cross_entropy 的均值添加到 `total_loss` 的 `collection` 中（后面会加入 L2 正则，所以计算 `total_loss`）。最后，使用 `tf.add_n` 将 `collection` 中的 loss 全部求和，得到最终的 loss。

优化器使用 `Adam Optimizer`。

使用 `tf.nn.in_top_k` 函数输出结果中 top k 的准确率，默认使用 top 1，也就是输出分数最高的那一类的准确率。
```python
# 损失函数
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
```

#### 训练
如下代码中的第三行启动前面提到的图片数据增强的线程队列，这里一共使用了 16 个线程来进行加速。注意，如果这里不启动线程，那么后续的 inference 及训练的操作都是无法开启的。
```python
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

max_steps = 3000
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        print 'step {},loss={},({} examples/sec; {} sec/batch)'.format(step, loss_value, examples_per_sec, sec_per_batch)
```

#### 评测准确率
测试集一共有 10000 个样本，但是需要注意的是，我们依然要像训练时哪样使用固定的 batch_size，然后一个 batch 一个batch 地输入测试数据。
- 计算一共要多少个 batch 才能将全部样本测试完
- 在每一个 step 中使用 session 的 run 方法获取 `images_test`、`labels_test` 的 batch，再执行 `top_k_op` 计算模型在这个 batch 的 top 1 上预测正确的样本数。
- 汇总所有预测正确的结果，求得全部测试样本中预测正确的数量
- 求准确率的评测结果并打印

```python
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))  # 计算一共有多少组
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print 'precision = ', precision
```
运行结果为准确率 74.56%。
实现的代码见 [github.com/ywtail](https://github.com/ywtail/TensorFlow/tree/master/5_CNN%E5%AF%B9CIFAR-10%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB) 中的 `cnn.py` 文件。

### 加入 LRN
LRN 最早见于 Alex 那篇用 CNN 参加 `ImageNet` 比赛的论文，Alex 在论文中解释 LRN 层模仿了生物神经系统的“侧抑制”机制，对局部神经元的活动创建竞争环境，使得其中相应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。Alex 在 `ImageNet` 数据集上的实验表明，使用 LRN 后 CNN 在 Top1 的错误率可以降低 1.4%，因此在其经典的 `AlexNet` 中使用 LRN 层。LRN 对 ReLU 这种没有上限边界的激活函数会比较有用，因为它会从附近的多个卷积核的相应（Response）中挑选比较大的反馈，但不适合 `Sigmoid` 这种有固定边界并且能抑制过大值的激活函数。
尝试加入 LRN，增强模型泛化能力。现在网络结构如下表

| Layer 名称 | 描述            |
| :------: | :------------ |
|  conv1   | 卷积层，ReLU激活函数  |
|  pool1   | 最大池化          |
|  norm1   | LRN           |
|  conv2   | 卷积层，ReLU激活函数  |
|  norm2   | LRN           |
|  pool2   | 最大池化          |
|  local3  | 全连接层，ReLU激活函数 |
|  local4  | 全连接层，ReLU激活函数 |
|  logits  | 模型的输出         |

在上述 CNN 中加入 LRN，准确率 73.90%。
实现的代码见 [github.com/ywtail](https://github.com/ywtail/TensorFlow/tree/master/5_CNN%E5%AF%B9CIFAR-10%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB) 中的 `cnn_lrn.py` 文件。

### 对权重进行 L2 正则化
为了避免过拟合，在神经网络中使用 L2 正则化。

在机器学习中，不管是分类还是回归任务，都可能因特征过多而导致过拟合，一般可以通过减少特征或者惩罚不重要特征的权重来缓解这个问题。但是通常我们并不知道该惩罚哪些特征的权重，而正则化就是帮助我们惩罚特征权重的，即 **特征的权重也会成为模型损失函数的一部分**。这样我们就可以筛选出最有效的特征，减少特征权重防止过拟合。

一般来说，L1 正则会制造稀疏的特征，大部分无用特征的权重会被置为 0，而L2 正则会让特征的权重不过大，使得特征的权重比较平均。

在定义初始化 weight 的函数时，像之前一样使用 `tf.truncated_normal` 截断的正态分布来处理初始化权重。与之前不同的是，给权重设置函数加一个参数 w1，如果 w1 不等于 0，就给 `weight` 加上一个 L2 的 loss，相当于做了一个 L2 的正则化处理。

```python
def weight_variable(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))  # stddev=stddev！！！
    if w1:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var
```

修改好这个函数后，将两个全连接层 weight 设置的第三个参数改为不为 0 的数。例如：`weight3 = weight_variable([dim, 384], 0.04, 0.004)`
实现的代码见 [github.com/ywtail](https://github.com/ywtail/TensorFlow/tree/master/5_CNN%E5%AF%B9CIFAR-10%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB) 中的 `cnn_l2.py` 文件。

### 总结

本文中实现的卷积神经网络没有那么复杂，在只使用 3000 个batch（每个batch 包含 128 个样本）时，设计的 CNN 模型在 CIFAR-10 数据集中分类的准确率为 74.56%；在 CNN 的基础上增加了 LRN，准确率 73.90%；如果对全连接层的权重进行 L2 正则化，准确率 70.30%；同时增加了 LRN，并对全连接层的权重进行了 L2 正则化，准确率 71.90%。具体见下表

| 网络结构       | 说明                         |  准确率   |
| :--------- | :------------------------- | :----: |
| CNN        | 只有卷积层、池化层、全连接层             | 74.56% |
| CNN+L2     | 对全连接层的 weights 进行了 L2 的正则化 | 70.30% |
| CNN+LRN    | 在每个卷积-最大池化层中使用了 LRN 层      | 73.90% |
| CNN+LRN+L2 | 同时使用 LRN 层和 L2 的正则化        | 71.90% |

在这个卷积神经网络中，我们使用了一些新的技巧。

- 对 weights 进行了 L2 的正则化
- 对图片进行了翻转、随机剪切等数据增强，制造了更多样本（下载的 TesorFlow Models 库中的 `cifar10_input.distorted_inputs` 函数）
- 在每个卷积-最大池化层中使用了 LRN 层，增强了模型的泛化能力

卷积层一般需要和一个池化层连接，卷积加池化的组合目前已经是做图像处理时的一个标准组件了。卷积网络最后的几个全连接层的作用是输出分类结果，前面的卷积层主要做特征提取的工作，直到最后的全连接层才开始对特征进行组合匹配，并进行分类。

可以观察到，其实设计 CNN 主要就是安排卷积层、池化层、全连接层的分布和顺序，以及其中超参数的设置、Trick 的使用等。设计性能良好的 CNN 是有一定规律可循的，但是想要针对某个问题设计最合适的网络结构，是需要大量实践摸索的。

### 完整代码
以下是在 CNN 中同时增加了 LRN，并对全连接层的权重进行了 L2 正则化的代码，其他相关代码参见 [github.com/ywtail](https://github.com/ywtail/TensorFlow/tree/master/5_CNN%E5%AF%B9CIFAR-10%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB) 
```python
# coding=utf-8
# cnn_l2_lrn
from __future__ import division

import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import math
import time

data_dir = 'cifar10_data/cifar-10-batches-bin'  # 下载 CIFAR-10 的默认路径
cifar10.maybe_download_and_extract()  # 下载数据集，并解压、展开到其默认位置

batch_size = 128
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)


def weight_variable(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))  # stddev=stddev！！！
    if w1:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def bias_variable(cons, shape):
    initial = tf.constant(cons, shape=shape)  # 必须是 shape=shape
    return tf.Variable(initial)


def conv(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一层
weight1 = weight_variable([5, 5, 3, 64], 5e-2, 0.0)
bias1 = bias_variable(0.0, [64])

conv1 = tf.nn.relu(conv(image_holder, weight1) + bias1)
pool1 = max_pool_3x3(conv1)
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 第二层
weight2 = weight_variable([5, 5, 64, 64], 5e-2, 0.0)
bias2 = bias_variable(0.1, [64])

conv2 = tf.nn.relu(conv(norm1, weight2) + bias2)
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = max_pool_3x3(norm2)

reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value

# 全连接层
weight3 = weight_variable([dim, 384], 0.04, 0.004)
bias3 = bias_variable(0.1, [384])

local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 全连接层
weight4 = weight_variable([384, 192], 0.04, 0.004)
bias4 = bias_variable(0.1, [192])

local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 输出
weight5 = weight_variable([192, 10], 1 / 192.0, 0.0)
bias5 = bias_variable(0.0, [10])
logits = tf.matmul(local4, weight5) + bias5


# 损失函数
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

max_steps = 3000
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        print 'step {},loss={},({} examples/sec; {} sec/batch)'.format(step, loss_value, examples_per_sec,
                                                                       sec_per_batch)
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))  # 计算一共有多少组
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print 'precision = ', precision
```
### 参考

- 图书：TensorFlow实战 / 黄文坚，唐源著
- TensorFlow 中文社区：[卷积神经网络](http://www.tensorfly.cn/tfdoc/tutorials/deep_cnn.html)