---
title: 'TensorFlow (1): 对评论进行分类'
date: 2017-03-24 20:42:42
tags: [TensorFlow,python]
categories: TensorFlow
top: 2
---

对评论进行分类是按照[TensorFlow练习1: 对评论进行分类](http://blog.topspeedsnail.com/archives/10399)来做的，所做的只是代码理解+重新实现。

原作者：@斗大的熊猫，地址：http://blog.topspeedsnail.com/archives/10399

### 关于 TensorFlow

引用[TensorFlow中文社区](http://www.tensorfly.cn/)中的解释：

> TensorFlow™ 是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。它灵活的架构让你可以在多种平台上展开计算，例如台式计算机中的一个或多个CPU（或GPU），服务器，移动设备等等。TensorFlow 最初由Google大脑小组（隶属于Google机器智能研究机构）的研究员和工程师们开发出来，用于机器学习和深度神经网络方面的研究，但这个系统的通用性使其也可广泛用于其他计算领域。

### 流程总结

所使用的是`python 3.6` + `tensorflow 1.0.0`，具体实现流程如下

- 下载数据集
  - neg.txt：5331条负面电影评论[下载]([http://blog.topspeedsnail.com/wp-content/uploads/2016/11/neg.txt](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/neg.txt))
  - pos.txt：5331条正面电影评论 [下载]([http://blog.topspeedsnail.com/wp-content/uploads/2016/11/pos.txt](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/pos.txt))
- 构建`lex`词汇表
  - 使用`nltk.tokenize.word_tokenize()`对数据集中数据进行分词，结果保存到`lex`
  - 使用`nltk.stem.WordNetLemmatizer`对lex中词汇进行词形还原（例如cats还原为cat），结果更新到`lex`
  - 使用`nltk.FreqDist()`进行词频统计（统计完成也相当于去除了重复词汇），结果保存到`lex_freq`
  - 筛选`lex_freq`中词频在20到2000之间的词汇，结果更新到lex
- 评论向量化，结果保存到`dataset`
  - 对于每条评论，向量features的长度为lex长度，初始化为全0。如果评论中的的词出现在`lex`中，则这个词对应位置的值改为1
  - 使用clf表示评论的分类，[0,1]代表负面评论 [1,0]代表正面评论
  - 对每条评论，将[features,clf]作为向量加入`dataset`
- 构建神经网络
  - 构建具有两层hidden layer的前馈神经网络
  - 每一层的w和b均由`tf.random_normal()`生成
- 使用`dataset`中数据训练神经网络
  - 取70%作为训练数据集，30%作为测试数据集
  - `dataset`中，features作为x，clf作为y，每次取50条数据训练，共训练15次

### 参考资料

对于TensorFlow看不懂的部分，参考[TensorFlow学习笔记1：入门](http://www.jeyzhang.com/tensorflow-learning-notes.html)

- Sessions最后需要关闭，以释放相关的资源；你也可以使用`with`模块，session在`with`模块中自动会关闭

- 抓取(Fetches)：为了抓取`ops`的输出，需要先执行`session`的`run`函数。然后，通过`print`函数打印状态信息。

  ```python
  input1 = tf.constant(3.0)
  input2 = tf.constant(2.0)
  input3 = tf.constant(5.0)
  intermed = tf.add(input2, input3)
  mul = tf.mul(input1, intermed)

  with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)

  # output:
  # [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
  ```

  所有tensors的输出都是一次性 [连贯] 执行的。

- 填充(Feeds):TensorFlow也提供这样的机制：先创建特定数据类型的占位符(placeholder)，之后再进行数据的填充。例如下面的程序：

  ```python
  input1 = tf.placeholder(tf.float32)
  input2 = tf.placeholder(tf.float32)
  output = tf.mul(input1, input2)

  with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

  # output:
  # [array([ 14.], dtype=float32)]
  ```

  如果不对`placeholder()`的变量进行数据填充，将会引发错误，更多的例子可参考[MNIST fully-connected feed tutorial (source code)](https://www.tensorflow.org/versions/r0.7/tutorials/mnist/tf/index.html)。

### 完整代码及执行结果

根据个人习惯对原作者的代码进行了调整，具体如下

```python
# coding=utf-8
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import random
import numpy as np


# 获得分词结果
def create_tokens(filename):
    tokens = []
    lines = open(filename, 'r').readlines()
    for line in lines:
        tokens += word_tokenize(line)
    return tokens


# 整理词汇
# 使用lemmatize进行词性还原
# 使用FreqDist进行词频统计，统计完成相当于去重完成
# 沿用参考博客的词汇筛选方法：选词频>20且<2000的词放入词汇表lex中
def create_lexicon(lex):
    wnl = WordNetLemmatizer()
    lex = [wnl.lemmatize(w) for w in lex]
    lex_freq = nltk.FreqDist(lex)
    print('词性还原及去重后词汇数：', len(lex_freq))
    lex = [w for w in lex_freq if lex_freq[w] > 20 and lex_freq[w] < 2000]
    print('词频在20到2000之间的词汇数：', len(lex))
    return lex


# 评论向量化
# 每条评论向量的维数是len(lex),初始化为全0，若评论中的词在lex中存在，则词汇对应位置为1
# lex是词汇表，clf是评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
def create_dataset(filename, lex, clf):
    lines = open(filename, 'r').readlines()
    dataset = []
    for line in lines:
        features = [0 for i in range(len(lex))]
        words = word_tokenize(line)
        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(w) for w in words]
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1
        dataset.append([features, clf])
    return dataset


# 构造神经网络
# 此处构建的是具有两层hidden layer的前馈神经网络
def neural_network(data, n_input_layer, n_layer_1, n_layer_2, n_output_layer):
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


# 使用数据训练神经网络
# epochs=15，训练15次
def train_neural_network(X, Y, train_dataset, test_dataset, batch_size, predict):
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    epochs = 15
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        random.shuffle(train_dataset)
        train_x = train_dataset[:, 0]
        train_y = train_dataset[:, 1]

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(train_x) - batch_size)[::batch_size]:
                batch_x = train_x[i:i + batch_size]
                batch_y = train_y[i:i + batch_size]

                _, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
                epoch_loss += c
            print(epoch, 'epoch_loss :', epoch_loss)

        test_x = test_dataset[:, 0]
        test_y = test_dataset[:, 1]
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率：', accuracy.eval({X: list(test_x), Y: list(test_y)}))


def main():
    pos_file = 'pos.txt'
    neg_file = 'neg.txt'

    lex = []  
    lex += create_tokens(pos_file) #正面评论分词
    lex += create_tokens(neg_file)
    print('分词后词汇数：', len(lex))

    lex = create_lexicon(lex) # 词汇整理

    dataset = []  # 保存评论向量化结果
    dataset += create_dataset(pos_file, lex, [1, 0])  # 正面评论
    dataset += create_dataset(neg_file, lex, [0, 1])  # 负面评论
    print('总评论数：', len(dataset))

    random.shuffle(dataset)
    dataset = np.array(dataset)

    test_size = int(len(dataset) * 0.3)  # 取30%的数据作为测试数据集
    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]

    n_input_layer = len(lex)
    n_layer_1 = 1000
    n_layer_2 = 1000
    n_output_layer = 2
    batch_size = 50  # 每次取50条评论进行训练

    X = tf.placeholder('float', [None, len(train_dataset[0][0])])
    Y = tf.placeholder('float')
    predict = neural_network(X, n_input_layer, n_layer_1, n_layer_2, n_output_layer)
    train_neural_network(X, Y, train_dataset, test_dataset, batch_size, predict)


if __name__ == '__main__':
    main()
```

**执行结果**

```
分词后词汇数： 230193
词性还原及去重后词汇数： 18643
词频在20到2000之间的词汇数： 1065
总评论数： 10662
0 epoch_loss : 48756.8896103
1 epoch_loss : 11542.5044079
2 epoch_loss : 4327.35995594
3 epoch_loss : 2536.12286408
4 epoch_loss : 1304.26391755
5 epoch_loss : 366.168846184
6 epoch_loss : 237.654407166
7 epoch_loss : 78.9821858138
8 epoch_loss : 28.4057260391
9 epoch_loss : 37.6864250147
10 epoch_loss : 40.7936543619
11 epoch_loss : 65.4782509211
12 epoch_loss : 56.0021716377
13 epoch_loss : 81.3493594176
14 epoch_loss : 113.843462383
准确率： 0.605378
```