---
title: sklearn决策树可视化
date: 2017-06-08 10:07:48
tags: [python,可视化]
categories: python
---

scikit-learn 中决策树的可视化一般需要安装 graphviz：主要包括 graphviz 的安装和 python 的 graphviz 插件的安装。

- `brew install graphviz` 安装graphviz
- `pip install graphviz` 安装python中的graphviz
- `pip install pydotplus` 安装python中的pydotplus

以下示例来自 scikit learn 官方文档 [1.10. Decision Trees](http://scikit-learn.org/stable/modules/tree.html#decision-trees)

**代码的运行效果见 [这个链接](https://ywtail.github.io/python/%E5%8F%AF%E8%A7%86%E5%8C%96/2_sklearn%E5%86%B3%E7%AD%96%E6%A0%91%E5%8F%AF%E8%A7%86%E5%8C%96.html)。**


### 方法一：export_graphviz 将树导出为 Graphviz 格式

```python
from sklearn import tree
from sklearn.datasets import load_iris

# 载入sklearn中自带的数据Iris，构造决策树

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# 训练完成后，我们可以用 export_graphviz 将树导出为 Graphviz 格式，存到文件iris.dot中
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
```

此时已经在本地生成了 `iris.dot` 文件，在命令行输入`dot -Tpdf iris.dot -o iris.pdf`生成决策树的PDF可视化文件，打开 `iris.pdf` （`open iris.pdf`）就能够看到生成的图片了。

官网还提供了删除 `iris.dot` 文件的方法：（如果想要删除，也可以直接在命令行`rm iris.dot`）

```python
import os
os.unlink('iris.dot') #os.unlink() 方法用于删除文件
```

### 方法二：使用 pydotplus 直接生成 iris.pdf

按如下代码生成 iris_2.pdf，`open iris_2.pdf` 就能够看到决策树。
```python
import pydotplus

dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('iris.pdf')
```

### 方法三：直接在 jupyter notebook 中生成

代码的运行效果见 [这个链接](https://ywtail.github.io/python/%E5%8F%AF%E8%A7%86%E5%8C%96/2_sklearn%E5%86%B3%E7%AD%96%E6%A0%91%E5%8F%AF%E8%A7%86%E5%8C%96.html)。

```python
from IPython.display import Image  
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  
```

### 参考
- 博客园：[scikit-learn决策树算法类库使用小结](http://www.cnblogs.com/pinard/p/6056319.html)
- scikit learn官方文档 [1.10. Decision Trees](http://scikit-learn.org/stable/modules/tree.html#decision-trees)