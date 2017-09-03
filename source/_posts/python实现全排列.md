---
title: python实现全排列
date: 2017-08-10 10:56:44
tags:
categories:
---

## 使用permutations



```python
itertools.permutations(iterable[, r])
#创建一个迭代器，返回iterable中所有长度为r的项目序列，如果省略了r，那么序列的长度与iterable中的项目数量相同： 返回p中任意取r个元素做排列的元组的迭代器
```

例如：

```python
# coding:utf-8
from itertools import permutations

s = raw_input()
print permutations(s)
# <itertools.permutations object at 0x10b9a5410>
print list(permutations(s))
# [('a', 'b', 'c'), ('a', 'c', 'b'), ('b', 'a', 'c'), ('b', 'c', 'a'), ('c', 'a', 'b'), ('c', 'b', 'a')]
print list(permutations(s, 2))
# [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]

print [''.join(x) for x in list(permutations(s))]
# ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
```

所以利用permutations实现的全排列的代码为：

