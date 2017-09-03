---
title: 'Machine Learning (1): kNN'
date: 2017-09-03 18:39:36
tags: [Machine Learning,python]
categories: Machine Learning
---



## kNN简介

### 算法流程

- 基本分类和回归方法。大多用于分类。
- 分类时，根据k个最近邻实例的类别，通过多数表决等方式进行预测。
- 不具有显示的学习过程。（训练过程）
- KNN即最近邻算法，其主要过程为：
  - 计算训练样本和测试样本中每个样本点的距离（常见的距离度量有欧式距离，马氏距离等）；
  - 对上面所有的距离值进行排序；
  - 选前k个最小距离的样本；
  - 根据这k个样本的标签进行投票，得到最后的分类类别；

### 三个基本要素
- 这三个基本要素是：
  - k值的选择
  - 距离度量
  - 分类决策规则

#### k值的选择
- k=1时称 最近邻法
- k值的选择（不具有显示学习过程，所以下方学习打引号）
  - k不同，有可能导致分类不同
  - 在应用中，k值一般取一个比较小的数值。通常采用**交叉验证法**来选取最优的k值
  - k过小
    - 过拟合
    - 相当于用较小的领域中的训练实例进行预测
    - “学习”的近似误差（approximation error）会减小（与训练集误差小）
    - “学习”的估计误差（estimation error）会增大（与测试集误差大）
    - 预测结果会对近邻实例点非常敏感，如果近邻的实例点恰好是噪声，预测就会出错
    - 即，k值的减小意味着整体模型变得复杂，容易发生过拟合
  - k过大
    - 欠拟合
    - 相当于用较大的领域中的训练实例进行预测
    - “学习”的近似误差（approximation error）会增大（与训练集误差大）
    - “学习”的估计误差（estimation error）会减小（与测试集误差小）
    - 与输入实例较远的（不相似的）训练实例也会对预测起作用，使预测发生错误
    - 即，k值得增大意味着整体的模型变得简单
    - 如果k=N，则不论输入什么，都简单地预测它属于实例中最多的类。这时，模型过于简单，完全忽略了训练实例中的大量有用信息，是不可取的

#### 距离度量
  - 欧式距离 Euclidean distance（Lp距离，p=2的情况）
  - p=1，曼哈顿距离 Manhattan distance
  - p=无穷，各坐标距离的最大值（无求和过程）
  - p越大，Lp距离越小
  - 距离度量不同，x1与x2最近，可能变为x1与x3最近。
  - 从图中直接观察的距离是欧式距离。
  - 所以，并不是在图上看起来最近，使用Lp算出来的距离就最近，这完全要看p的取值。

#### 分类决策规则
  - 往往是多数表决。多数表决规则（majority voting rule）

### kNN实现
- 简单实现方法：线性扫描（linear scan）
  - 计算输入实例与每一个训练实例的距离。
  - 当训练集很大时，非常耗时，不可行
- kd树（kd tree），这里k与k-NN的k意义不同
  - 实现时，主要考虑如何对训练数据快速地进行k近邻搜索。在特征空间维数大及训练数据容量大时尤其必要。
  - 为了提高搜索效率，考虑使用特殊的结构存储训练数据，以减少计算距离的次数
  - 具体方法很多，介绍kd树

### 优缺点
- KNN算法的优点
  - 思想简单，理论成熟，既可以用来做分类也可以用来做回归；
  - 可用于非线性分类；
  - 训练时间复杂度为O(n)；
  - 准确度高，对数据没有假设，对outlier不敏感；
- 缺点
  - 计算量大；
  - 样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
  - 需要大量的内存；

## Python实现

环境：
>Python 2.7.13
numpy 1.12.0
pandas 0.19.2
skleran 0.18.1

为了验证效果，使用的数据集是kaggle上digit-recognizer给的数据集，取了前3000行来验证实现的kNN效果。

### 线性扫描法（Linear Scan）实现
```python
# coding:utf-8
from __future__ import division
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def knnClassify(newinput, datas, labels, k, p):
    # 计算距离
    diff = abs(np.tile(newinput, (len(datas), 1)) - datas)
    distances = (np.sum(diff ** p, axis=1)) ** (1 / p)
    sort_distances = np.argsort(distances)
    #print distances
    # print sort_distances

    # 投票法决定分类
    classCount = {}
    for i in range(k):
        label = labels[sort_distances[i]]
        classCount[label] = classCount.get(label, 0) + 1

    maxCount = 0
    maxIndex = -1
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex


if __name__ == '__main__':
    '''
    # example in book
    group = [[5, 1], [4, 4]]
    lables = ['a', 'b']
    print knnClassify([1, 1], group, lables, 1, 1)
    print knnClassify([1, 1], group, lables, 1, 2)
    print knnClassify([1, 1], group, lables, 1, 3)
    print knnClassify([1, 1], group, lables, 1, 4)
    '''

    # example for digit recognizer
    all_data = pd.read_csv('/Users/liuyue/workspace/pythonstudy/test/digit_datas.csv')
    y = all_data['label'].values
    x = all_data.drop(['label'], axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print 'Data Infomation:'
    print 'Train Data Information: ',x_train.shape, len(y_train)
    print 'Test Data Information: ',x_test.shape, len(y_test)

    matchCount = 0
    accuracy = 0
    print '识别错误的数如下：'
    for i in range(len(x_test)):
        predict = knnClassify(x_test[i], x_train, y_train, 3, 2)
        # print predict,y_test[i]
        if predict == y_test[i]:
            matchCount += 1
        else: # 打印识别错误的数
            print predict, y_test[i]
        accuracy = float(matchCount) / len(x_test)
    print 'accuracy:',accuracy
```

运行结果
```
Data Infomation:
Train Data Information:  (2100, 784) 2100
Test Data Information:  (900, 784) 900
识别错误的数如下：
6 0
9 5
4 9
6 8
0 8
5 8
8 2
1 7
8 3
7 9
1 3
1 4
1 7
8 5
3 8
1 8
0 6
0 5
1 8
9 8
7 3
9 4
5 9
3 2
1 7
8 2
8 5
1 7
7 8
6 0
8 2
5 3
9 4
1 2
0 5
9 4
9 4
7 2
9 4
7 9
7 8
8 3
4 9
9 4
9 8
9 3
4 8
9 7
5 8
5 8
1 7
1 8
1 2
9 5
1 7
0 5
6 0
8 5
3 8
2 8
1 4
9 4
9 4
0 9
9 4
5 9
7 2
8 5
9 5
9 3
8 3
2 3
9 8
4 2
5 8
9 0
5 8
1 8
9 8
accuracy: 0.912222222222
```


## 参考

- 李航. 统计学习方法[M]. 清华大学出版社, 2012.
- [机器学习&数据挖掘笔记_16（常见面试之机器学习算法思想简单梳理）](http://www.cnblogs.com/tornadomeet/p/3395593.html)
- [机器学习算法与Python实践之（一）k近邻（KNN）](http://blog.csdn.net/zouxy09/article/details/16955347)

