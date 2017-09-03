---
title: Kaggle记录
date: 2017-06-19 21:48:41
tags: [Kaggle,MachineLearning,python]
categories: Kaggle
---

## Titanic
代码地址：https://github.com/ywtail/kaggle/tree/master/1_Titanic
### 2017.4.7：Titanic-1
score：`0.76555`
使用模型：`RandomForestClassifier`
运行过程展示地址：https://ywtail.github.io/kaggle/1_Titanic/Titanic-1.html
说明：titanic 初探，只简单对缺失数据进行了填充。使用 `GridSearchCV` 对参数进行了调整。

### 2017.6.22：Titanic-2
score：`0.76555` (后来提交分数越来越低)
使用模型：`RandomForestClassifier`
运行过程展示地址：https://ywtail.github.io/kaggle/1_Titanic/Titanic-2.html
说明：简单特征工程，从 Name 得到 Title、从 Parch、SibSp 得到家庭人数，从 Age 等信息得到是不是 Child，是不是有 Child（预测时使用分数反而降低了），随机（[average - std, average + std]）生成了缺失的年龄。
参考：https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic

### 2017.6.23：Titanic-3
score: `0.77033`
运行过程展示地址：https://ywtail.github.io/kaggle/1_Titanic/Titanic-3.html
大体流程如下：
- 删除不需要的项：PassengerId(训练数据中的)，Name，Ticket
- 处理缺失数据：Embarked，Fare，Age，Cabin（缺失太多直接删除）
- 特征工程：Family，Person，Pclass

使用模型及提交后得分：
- `LogisticRegression`：0.76077
- `SVC`：0.61722
- `RandomForestClassifier`：0.73206
- `KNeighborsClassifier`：0.62201
- `GaussianNB`：0.73206
- `GradientBoostingClassifier`：0.77033

参考：https://www.kaggle.com/omarelgabry/a-journey-through-titanic

### 2017.7.10：Titanic-4
score: `0.79426` (排名`1841/7167(26%)`)
运行过程展示地址：https://ywtail.github.io/kaggle/1_Titanic/Titanic-4.html
Age的缺失值填充参考了 Pclass 和 Sex；Age 和 Fare 分段。
使用模型及提交后得分：
- `LogisticRegression`：0.76555
- `SVC`：0.77990
- `LinearSVC`：0.76555
- `KNeighborsClassifier`：0.77033
- `GaussianNB`：0.74163
- `Perceptron`：0.75598
- `SGDClassifier`：0.79426
- `DecisionTreeClassifier`：0.78469
- `RandomForestClassifier`：0.77990
- `GradientBoostingClassifier`：0.79426

参考：https://www.kaggle.com/startupsci/titanic-data-science-solutions

## Digit Recognizer
Kaggle地址：https://www.kaggle.com/c/digit-recognizer
代码地址：https://github.com/ywtail/kaggle/tree/master/2_Digit_Recognizer

### 2017.7.26：1-SVC【0.90929】.ipynb
运行过程展示：点击[这里](https://ywtail.github.io/kaggle/2_Digit_Recognizer/1-SVC%E3%80%900.90929%E3%80%91.html)

探索数据，流程如下：
- 取5000个样本进行训练。
- 特征缩放：大于1的特征取1。
- 使用SVC，提交得分 0.90929


### 2017.8.3：2-softmax【0.90971】.ipynb
运行过程展示：点击[这里](https://ywtail.github.io/kaggle/2_Digit_Recognizer/2-Softmax+Regression%E3%80%900.90971%E3%80%91.html)

y=softmax(xW+b)（特征缩放：特征/255）

代价函数：交叉熵

最小化代价函数：梯度下降 GradientDescentOptimizer，学习率0.01

详细分析见：[TensorFlow (2): Softmax Regression识别手写数字](http://ywtail.github.io/2017/06/02/TensorFlow-2-Softmax-Regression%E8%AF%86%E5%88%AB%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97/)


### 2017.8.3：3-多层感知机 【0.96486】.ipynb
运行过程展示：点击[这里](https://ywtail.github.io/kaggle/2_Digit_Recognizer/3-%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA+%E3%80%900.96486%E3%80%91.html)

Softmax Regression 和传统意义上的神经网络的最大区别是没有隐含层。
这里实现的多层感知机实际上是在 Softmax Regression 的基础上加上一个隐含层。
结构如下：
- x=tf.placeholder(tf.float32,[None,784])
- hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
- hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
- y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
- 代价函数：交叉熵
- 最小化代价函数：AdagradOptimizer，学习率0.01

详细分析见：[TensorFlow (3): 多层感知机识别手写数字](http://ywtail.github.io/2017/06/03/TensorFlow-3-%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA%E8%AF%86%E5%88%AB%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97/)

### 2017.8.5：4-CNN-Tensorflow【0.98957】.ipynb
运行过程展示：点击[这里](https://ywtail.github.io/kaggle/2_Digit_Recognizer/4-CNN-Tensorflow%E3%80%900.98957%E3%80%91.html)

结构如下：
- x_image=tf.reshape(x,[-1,28,28,1])
- h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
- h_pool1=max_pool_2x2(h_conv1)
- h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
- h_pool2=max_pool_2x2(h_conv2)
- h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
- h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
- h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
- y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
- 代价函数：交叉熵
- 最小化代价函数：AdamOptimizer，学习率1e-4

详细分析见：[TensorFlow (4): 卷积神经网络识别手写数字](http://ywtail.github.io/2017/06/05/TensorFlow-4-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AF%86%E5%88%AB%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97/)

### 2017.8.6：5-CNN-Keras【0.99514】.ipynb
运行过程展示：点击[这里](https://ywtail.github.io/kaggle/2_Digit_Recognizer/5-CNN-Keras%E3%80%900.99514%E3%80%91.html)

Top: 12%，198/1789

参考：https://www.kaggle.com/toregil/welcome-to-deep-learning-cnn-99

结构如下：
```python
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
```