---
title: 'TensorFlow (4): 卷积神经网络识别手写数字'
date: 2017-06-05 12:33:34
tags: [TensorFlow,MachineLearning,python]
categories: TensorFlow
top: 2
---

使用全连接神经网络（`Fully Connected Netword`,` FCN`,` MLP` 的另一种说法）也是有局限的，即使我们使用很深的网络，很多的隐藏节点，很大的迭代轮数，也很难在 `MNIST` 数据集上达到 99% 以上的准确率。

因此接下来我们介绍卷积神经网络，以及如何在 `MNIST` 数据及上使用 `CNN` 达到 99% 以上的准确率。

### 卷积神经网络简介
卷积神经网络（`Convolutional Neural Network`，`CNN`）最初是为解决图像识别等问题设计的，当然现在的应用不仅限于图像和视频，也可以用于时间序列信号，例如音频信号、文本数据等。

在早期的图像识别研究中，最大的挑战是如何组织特征，因为图像数据不像其他类型的数据那样可以通过人工理解来提取特征。在深度学习出现之前，必须借助`SIFT` 、`HoG` 等算法提取具有良好区分性的特征，再集合 `SVM` 等机器学习算法进行图像识别。然而 `SIFT` 这类算法提取的特征还是有局限性的，在 `ImageNet ILSVRC` 比赛的最好结果错误率也有 26% 以上，而且常年难以产生突破。卷积神经网络提取的特征则可以达到更好地效果，同时它不需要将特征提取和分类训练两个过程分开，它在训练时就自动提取了最有效的特征。

`CNN` 作为一个深度学习架构被提出的最初诉求，是降低对图像预处理的要求，以避免复杂的特征工程。`CNN` 可以直接使用图像的原始像素作为输入，而不必先使用 `SIFT` 等算法提取特征，减轻了使用传统算法如 `SVM` 时必须要做的大量重复、繁琐的数据预处理。和 SIFT 算法类似，CNN 训练的模型同样对缩放、平移、旋转等畸变具有不变形，有着很强的泛化性。

CNN 的最大特点在于卷积的权值共享结构，可以大幅度减少神经网络的参数量，防止过拟合的同时又降低了神经网络模型的复杂度。

总的来说，全连接神经网络之所以不太适合图像识别任务，主要有以下几个方面的问题：

- **参数数量太多** 考虑一个输入`1000*1000`像素的图片(一百万像素，现在已经不能算大图了)，输入层有`1000*1000`=100 万节点。假设第一个隐藏层有 100 个节点(这个数量并不多)，那么仅这一层就有`(1000*1000+1)*100`=1 亿参数，这实在是太多了！我们看到图像只扩大一点，参数数量就会多很多，因此它的扩展性很差。
- **没有利用像素之间的位置信息** 对于图像识别任务来说，每个像素和其周围像素的联系是比较紧密的，和离得很远的像素的联系可能就很小了。如果一个神经元和上一层所有神经元相连，那么就相当于对于一个像素来说，把图像的所有像素都等同看待，这不符合前面的假设。当我们完成每个连接权重的学习之后，最终可能会发现，有大量的权重，它们的值都是很小的(也就是这些连接其实无关紧要)。努力学习大量并不重要的权重，这样的学习必将是非常低效的。
- **网络层数限制** 我们知道网络层数越多其表达能力越强，但是通过梯度下降方法训练深度全连接神经网络很困难，因为全连接神经网络的梯度很难传递超过 3 层。因此，我们不可能得到一个很深的全连接神经网络，也就限制了它的能力。

卷积神经网络解决这个问题主要有三个思路：

- **局部连接** 这个是最容易想到的，每个神经元不再和上一层的所有神经元相连，而只和一小部分神经元相连。这样就减少了很多参数。
- **权值共享** 一组连接可以共享同一个权重，而不是每个连接有一个不同的权重，这样又减少了很多参数。
- **下采样** 可以使用 `Pooling` 来减少每层的样本数，进一步减少参数数量，同时还可以提升模型的鲁棒性。

对于图像识别任务来说，卷积神经网络通过尽可能保留重要的参数，去掉大量不重要的参数，来达到更好的学习效果。

**关于卷积神经网络更详细的解读参见博客 [零基础入门深度学习(4) - 卷积神经网络](https://www.zybuluo.com/hanbingtao/note/485480)**，作者总结的非常详细。

### 实现卷积神经网络
#### 加载数据
首先加载 `MNIST` 数据集，并创建一个 `TensorFlow` 默认的 `InteractiveSession`，这样在后续的各项操作中就无需指定 `Session` 了。
```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
sess=tf.InteractiveSession()
```
#### 权重初始化

为了创建这个模型，我们需要创建大量的权重和偏置项，为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。

- 权重：这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免 0 梯度，因此标准差设为 0.1。
- 偏置：由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为 0 的问题（`dead neurons`）。

```python
def weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
```

#### 卷积和池化

`TensorFlow` 在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多大？在这个实例里，我们会一直使用 `vanilla` 版本。我们的卷积使用1步长（`stride size`），0边距（`padding size`）的模板，保证输出和输入是同一个大小。我们的池化用简单传统的2x2大小的模板做 `max pooling`。为了代码更简洁，我们把这部分抽象成一个函数。

- `tf.nn.conv2d` 是 `TensorFlow` 中的 2 维卷积函数，其中 `x` 是输入，`W` 是卷积的参数，`Strides` 代表卷积模板移动的步长，`Padding` 代表边界的处理方式，`padding='SAME'` 表明不再给输入元素的周围补充元素，让卷积的输入和输出保持同样的尺寸。具体示例参见[零基础入门深度学习(4) - 卷积神经网络](https://www.zybuluo.com/hanbingtao/note/485480)。
- `tf.nn.max_pool` 是 `TensorFlow` 中的最大池化函数，在这里使用 2x2 的最大池化，即将一个 2x2 的像素块降为 1x1 的像素。最大池化会保留原始像素块中灰度值最高的那一个像素，即保留最显著特征。池化层的 `strides` 设为横竖两个方向以 2 为步长。

```python
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
```

#### 第一层卷积

首先定义输入的 `placeholder`，`x` 是特征，`y_` 是真实的 `label`。因为卷积神经网络会利用到空间结构信息，因此需要将 1D 的输入向量转为 2D 的图片结构，即从 1x784 的形式转为原始的 28x28 的结构。同时因为只有一个颜色通道，故最终尺寸为` [-1,28,28,1]`，前面的 -1 代表样本数量不固定，最后的 1 代表颜色通道数为 1（因为是灰度图所以这里的通道数为 1，如果是 `rgb` 彩色图，则为 3）。这里我们使用的 tensor 变形函数是 `tf.reshape`。

```python
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])
```

现在我们可以开始实现第一层了。首先使用前面写好的函数进行参数初始化，包括 weights 和 bias。

- `weights`：卷积的权重张量形状是 `[5, 5, 1, 32]`，前两个维度是`patch`的大小，接着是输入的通道数目，最后是输出的通道数目，即卷积核尺寸是 5x5，颜色通道是 1，有 32 个不同的卷积核。
- `bias`：卷积在每个 5x5 的 `patch` 中算出 32 个特征，而对于每一个输出通道都有一个对应的偏置量。

第一层卷积由一个卷积接一个 `max pooling` 完成：

- 首先使用 `conv2d` 函数进行卷及操作，并加上偏置，接着再使用 `ReLU` 激活函数进行非线性处理。
- 然后使用最大池化函数 `max_pool_2x2` 对卷积的输出结果进行池化操作。

```python
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
```

#### 第二层卷积

这个卷积层基本和第一个卷积层一样，唯一不同的是，卷积核的数量变成了 64，也就是说每个 5x5 的`patch` 会得到 64 个特征。

```python
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
```

#### 密集连接层

因为前面经历了两次步长为 2x2 的最大池化，所以边长只有 1/4 了，即图片尺寸由 28x28 变为 7x7。并且由于第二个卷积层的卷积核数量为 64，所以输出的 `tensor` 尺寸是 7x7x64。

我们加入一个有 1024 个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量 `reshape` 成一些向量，乘上权重矩阵，加上偏置，然后对其使用 `ReLU`。

```python
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
```

#### Dropout

为了减少过拟合，我们在输出层之前加入 `Dropout`。

我们用一个 `placeholder` 来代表一个神经元的输出在 `dropout` 中保持不变的概率。这样我们可以在训练过程中启用 `dropout`，在测试过程中关闭 `dropout`。`TensorFlow` 的 `tf.nn.dropout` 操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的 `scale`。所以用 `dropout` 的时候可以不用考虑 `scale`。

```python
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
```

#### 输出层

我们添加一个 `softmax` 层，就像前面的单层 `softmax regression` 一样，得到最后的概率输出。

```python
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
```

这里的损失函数依然使用交叉信息熵，优化器使用 `Adam`，并把学习速率设为较小的 1e-4。

```python
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```

再继续定义评测准确率的操作，这里和之前一样。

```python
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
```

下面开始训练过程，首先依然是初始化所有参数。

 `keep_prob` 在训练时设置为 0.5。这里采用 5000 个 `batch`，每个 `batch` 包含 50 条的样本，参与训练的样本量共 25 万。其中每 500 次训练，会对准确率进行一次测评（测评时  `keep_prob` 设为 1），用以检测模型的性能。

```python
tf.global_variables_initializer().run()
for i in range(5000): # 20000次训练需要耗时30min，为了节省时间这次运行改为5000次
    batch = mnist.train.next_batch(50)
    if i % 500 == 0:
        train_accuracy = accuracy.eval({x: batch[0], y_: batch[1], keep_prob: 1.0})
        print 'step {},training accuracy {}'.format(i, train_accuracy)
    train_step.run({x: batch[0], y_: batch[1], keep_prob: 0.5})
```

全部训练完成后，在测试集上进行全面的测试，得到分类的准确率。

```python
test_accuracy=accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
print 'test accuracy',test_accuracy
```

### 总结

当进行 5000 次训练时，这个 `CNN` 模型可以得到 98.7% 的准确率，当进行 20000 次训练时，这个模型可以达到 99.18% 的准确率，基本可以满足对手写数字识别准确率的要求。

`CNN` 主要的性能提升都来自于更优秀的网络设计，即卷积网络对图像特征的提取和抽象能力。依靠卷积核的权值共享，`CNN` 的参数量并没有爆炸，减低计算量的同时也减轻了过拟合，因此整个模型的性能有较大的提升。


### 完整的代码及运行结果

在 [github](https://github.com/ywtail/TensorFlow/tree/master/4_%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AF%86%E5%88%AB%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97) 可以下载完整代码和数据集。

```python
# coding: utf-8
# 卷积神经网络识别手写数字
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# 定义准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 训练
tf.global_variables_initializer().run()
for i in range(5000): # 20000次训练需要耗时30min，为了节省时间这次运行改为5000次
    batch = mnist.train.next_batch(50)
    if i % 500 == 0:
        train_accuracy = accuracy.eval({x: batch[0], y_: batch[1], keep_prob: 1.0})
        print 'step {},training accuracy {}'.format(i, train_accuracy)
    train_step.run({x: batch[0], y_: batch[1], keep_prob: 0.5})


# 测试准确率
test_accuracy = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print 'test accuracy', test_accuracy
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
step 0,training accuracy 0.20000000298
step 500,training accuracy 0.899999976158
step 1000,training accuracy 0.959999978542
step 1500,training accuracy 0.939999997616
step 2000,training accuracy 1.0
step 2500,training accuracy 0.939999997616
step 3000,training accuracy 0.980000019073
step 3500,training accuracy 0.980000019073
step 4000,training accuracy 1.0
step 4500,training accuracy 0.980000019073
test accuracy 0.987
```

### 参考

- 图书：TensorFlow实战 / 黄文坚，唐源著
- TensorFlow 中文社区：[MNIST 进阶](http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html)
- 博客：[零基础入门深度学习(4) - 卷积神经网络](https://www.zybuluo.com/hanbingtao/note/485480)