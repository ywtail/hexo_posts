---
title: Kaggle记录
date: 2017-06-19 21:48:41
tags: [Kaggle,MachineLearning,python]
categories: Kaggle
top: 2
---

## Titanic
代码地址：https://github.com/ywtail/kaggle/tree/master/1_Titanic
- 2017.4.7：Titanic-1
> 
score：`0.76555`
使用模型：`RandomForestClassifier`
运行过程展示地址：https://ywtail.github.io/kaggle/1_Titanic/Titanic-1.html
说明：titanic 初探，只简单对缺失数据进行了填充。使用 `GridSearchCV` 对参数进行了调整

- 2017.6.22：Titanic-2
> 
score：`0.76555` (后来提交分数越来越低)
使用模型：`RandomForestClassifier`
运行过程展示地址：https://ywtail.github.io/kaggle/1_Titanic/Titanic-2.html
说明：简单特征工程，从 Name 得到 Title、从 Parch、SibSp 得到家庭人数，从 Age 等信息得到是不是 Child，是不是有 Child（预测时使用分数反而降低了），随机（[average - std, average + std]）生成了缺失的年龄。
参考：https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic