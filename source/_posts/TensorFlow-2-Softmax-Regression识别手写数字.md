---
title: 'TensorFlow (2): Softmax Regression识别手写数字'
date: 2017-06-02 18:47:34
tags: [TensorFlow,MachineLearning,python]
categories: TensorFlow
top: 2
mathjax: true
---

MNIST手写数字识别是机器学习领域的Hello World任务。
MNIST(Mixed National Institute of Standards and Techenology database)是一个非常简单的机器视觉数据集。由几万张28像素 x 28像素的手写数字组成，这些图片只包含灰度值信息。MNIST也包含每一张图片对应的标签，告诉我们这个是数字几。

### 数据集下载
TensorFlow为我们提供了一个方便的封装，可以直接将MNIST数据集加载成我们期望的格式。只需要使用以下两行代码
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
```
**注意**
如果运行这两行代码报错：`IOError: [Errno socket error] [Errno 60] Operation timed out`
可以尝试以下解决方法：
- 自行从[极客学院](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_download.html)下载数据集
- 新建`MNIST_data`文件夹，并将下载的4个.gz文件放入`MNIST_data`
- 将`MNIST_data`放入代码所在目录，运行上面的代码，成功输出提示信息，表示已经将MNIST数据集加载成我们期望的格式

MNIST数据集一共包含三个部分，打印一下数据的信息
```python
print 'train:',mnist.train.images.shape,mnist.train.labels.shape
print 'test:',mnist.test.images.shape,mnist.test.labels.shape
print 'validation:',mnist.validation.images.shape,mnist.validation.labels.shape
```
输出是：
>train: (55000, 784) (55000, 10)
test: (10000, 784) (10000, 10)
validation: (5000, 784) (5000, 10)

这个结果表明：
- 训练数据集(55,000份，`mnist.train`)，测试数据集(10,000份，`mnist.test`)，验证数据集(5,000份，`mnist.validation`)。
- `mnist.train.images` （训练集的图片）是一个形状为 `[55000, 784]`的张量，第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。
每一张图片包含28像素X28像素。我们可以用一个数字数组来表示这张图片：把这个数组展开成一个向量，长度是 `28x28 = 784`。如何展开这个数组（数字间的顺序）不重要，只要保持各个图片采用相同的方式展开。
- `mnist.train.labels` （训练集的标签）是一个 `[55000, 10]` 的数字矩阵，使用`one-hot vectors`，将数字n表示成一个只有在第n维度（从0开始）数字为1的10维向量。比如，标签0将表示成([1,0,0,0,0,0,0,0,0,0,0])。 

### 实现Softmax Regression模型
这里使用一个很简单的数学模型Softmax Regression(更详细的信息参见TensorFlow 中文社区：[MNIST机器学习入门](http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html))：

$$y=softmax(Wx+b)$$

使用TensorFlow实现实现Softmax Regression：
```python
import tensorflow as tf
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b) #tf.matmul实现了 x 和 W 的乘积
```
- 这里的输入 x 是一个占位符 `Placeholder`，在需要时指定。`Placeholder` 的第一个参数是数据类型，第二个参数`[None,784]`代表 tensor 的 shape，也就是数据的尺寸，这里 None 代表不限条数的输入，784 代表每条输入是一个 784 维的向量。
- 为模型中的 W 和 b 创建 `Variable` 对象，`Variable` 对象是用来存储模型参数的，在模型训练迭代中是持久化的（比如一直放在显存中），它可以长期存在并且在每轮迭代中被更新。
- 这里将 W 和 b 全部初始化为 0，因为模型训练时会自动学习合适的值，所以对这个简单模型来说初始值不太重要。不过对于复杂的卷积网络、循环网络或者比较深的全连接网络，初始化的方法就比较重要，甚至至关重要。
- 这里 W 的 shape 是`[784,10]`，784 是特征的维数，10 代表有 10 类（Label 在 one-hot 编码后是 10 维的向量）

### 模型训练及评测
为了训练模型，我们需要定义一个 loss function 损失函数来描述模型对问题的分类精度。loss function 越小代表模型的分类结果与真实值偏差越小，即模型越精确。
这里，我们使用交叉熵函数（cross-entropy）作为代价函数，交叉熵是一个源于信息论中信息压缩领域的概念，但是现在已经应用在多个领域。它的定义如下：

$$H_{y′}(y)=−\sum_{i}y_i'log(y_i)$$

这里 y 是所预测的概率分布，而 y′ 是真实的分布(one-hot vector表示的图片label)。直观上，交叉熵函数的输出值表示了预测的概率分布与真实的分布的符合程度。
交叉熵函数的实现如下（`reduce_sum`表示 ∑ 求和）
```python
y_=tf.placeholder(tf.float32,[None,10]) # y_存储实际lable
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
```

接下来以损失函数最小化为目标，来训练模型以得到参数 W 和 b 的值。
这里使用梯度下降算法来最小化代价函数。
```python
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init=tf.global_variables_initializer() # 对所有的参数进行初始化
sess=tf.Session() # 在一个Session里运行模型
sess.run(init) # 执行初始化

for i in range(1000): 
    batch_xs,batch_ys=mnist.train.next_batch(100) # 每次随机取100个样本进行训练
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
```

现在已经完成了训练，接下来对模型的准确率进行验证。
```python
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})) # 输出模型在测试及上的准确率
```

- `tf.argmax`是从一个 tensor 中寻找最大值的序号，`tf.argmax(y,1)`就是求各个预测的数字中概率最大的那一个，而`tf.argmax(y_,1)`是找样本的真实数字的类别。
- `tf.equal`方法用来判断预测值与真实值是否相等，返回的`correct_prediction`是一个布尔值的列表，例如 `[True, False, True, True]`。
- `tf.cast`将`correct_prediction`输出的 bool 值转换为 float，再求平均。
- 最后输出模型在测试集上的准确率，这里求得的准确率为 0.912。（目前最好的模型的准确率为99.7%）

### 完整的代码及运行结果
```python
# coding: utf-8
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 打印数据集基本本信息
print '=' * 30
print 'train:', mnist.train.images.shape, mnist.train.labels.shape
print 'test:', mnist.test.images.shape, mnist.test.labels.shape
print 'validation:', mnist.validation.images.shape, mnist.validation.labels.shape
print '=' * 30

# 实现softmax regression模型
x = tf.placeholder(tf.float32, [None, 784])  # x 使用占位符，在后续输入时填充
W = tf.Variable(tf.zeros([784, 10]))  # W 和 b 参数使用Variable，在迭代过程中不断更新
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 即 y = softmax(xW+b)

# 用cross_entropy作损失函数
y_ = tf.placeholder(tf.float32, [None, 10])  # y_表示 label的实际值
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 实现交叉熵函数

# 训练模型
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()  # 对所有的参数进行初始化
sess = tf.Session()  # 在一个Session里运行模型
sess.run(init)  # 执行初始化

for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次随机取100个样本进行训练
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 判断预测值与真实值是否相等，返回的correct_prediction是一个布尔值的列表，例如 [True, False, True, True]。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 将correct_prediction输出的bool值转换为float，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# 输出模型在测试及上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

如果使用`sess = tf.InteractiveSession()`将当前这个`session`注册为默认的`session`，那么后续的有些写法能够简洁一些，例如`sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})`可以直接写为`train_step.run({x: batch_xs, y_: batch_ys})`，完整代码见 [https://github.com/ywtail/](https://github.com/ywtail/TensorFlow/blob/master/2_SoftmaxRegression%E8%AF%86%E5%88%AB%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97/main_2.py)

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
==============================
train: (55000, 784) (55000, 10)
test: (10000, 784) (10000, 10)
validation: (5000, 784) (5000, 10)
==============================
0.922
```

### 总结

上面使用 `TensorFlow` 实现了一个简单的机器学习算法 `Softmax Regression`，这可以算作是一个没有隐含层的最浅的神经网络。整个流程可以分为4个部分：
>1. 定义算法公式，也就是神经网络的 `forward` 时的计算
2. 定义 `loss`，选定优化器（这里选的是梯度下降），并指定优化器优化 `loss`
3. 迭代地对数据进行训练
4. 在测试集或验证集上对准确率进行测评

以上几个步骤是使用 `TensorFlow` 进行算法设计、训练的核心步骤。
`TensorFlow` 和 `Spark` 类似，我们定义的各个公式其实只是 `Computation Graph`，在执行这行代码时，计算还没有实际发生，只有等调用 `run` 方法，并 `feed` 数据时计算才真正执行。比如 `cross_entropy`、`train_step`、`accuracy` 等都是计算图中的节点，而不是数据结果，我们可以通过调用 `run` 方法执行这些结点或者说运算操作来获取结果。

### 参考
- 图书：TensorFlow实战 / 黄文坚，唐源著
- Jey Zhang 的博客：[TensorFlow学习笔记1：入门](http://www.jeyzhang.com/tensorflow-learning-notes.html)
- TensorFlow 中文社区：[MNIST机器学习入门](http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html)





