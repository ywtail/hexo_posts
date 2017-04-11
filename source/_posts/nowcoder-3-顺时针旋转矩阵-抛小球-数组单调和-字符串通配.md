---
title: 'nowcoder(3):顺时针旋转矩阵; 抛小球; 数组单调和; 字符串通配'
date: 2017-04-11 11:06:20
tags: [nowcoder,python]
categories: nowcoder
---

## 顺时针旋转矩阵

### 题目描述

[题目链接](https://www.nowcoder.com/practice/2e95333fbdd4451395066957e24909cc?tpId=49&tqId=29373&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

有一个NxN整数矩阵，请编写一个算法，将矩阵顺时针旋转90度。
给定一个NxN的矩阵，和矩阵的阶数N,请返回旋转后的NxN矩阵,保证N小于等于300。

**测试样例**

>[[1,2,3],[4,5,6],[7,8,9]],3
>返回：[[7,4,1],[8,5,2],[9,6,3]]

### 代码

- 常规思路
```python
# -*- coding:utf-8 -*-

class Rotate:
    def rotateMatrix(self, mat, n):
        rotate_mat = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                rotate_mat[i][j] = mat[n - j - 1][i]
        return rotate_mat

# 运行时间：400ms
# 占用内存：3148k
```

- 在评论中看到另一种：使用zip，只有一行代码

  参考：[Python中zip()函数用法实例教程](http://www.jb51.net/article/53051.htm)

  > zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list（列表）。若传入参数的长度不等，则返回list的长度和参数中长度最短的对象相同。利用*号操作符，可以将list unzip（解压）。
  ```python
  >>> a = [1,2,3]
  >>> b = [4,5,6]
  >>> c = [4,5,6,7,8]
  >>> zipped = zip(a,b)
  [(1, 4), (2, 5), (3, 6)]
  >>> zip(a,c)
  [(1, 4), (2, 5), (3, 6)]
  >>> zip(*zipped)
  [(1, 2, 3), (4, 5, 6)]
  ```

- 根据以上zip的用法，这一题可以这样写：
```python
# -*- coding:utf-8 -*-

class Rotate:
    def rotateMatrix(self, mat, n):
         return [x[::-1] for x in zip(*mat)]

# 运行时间：380ms
# 占用内存：3148k
```

## 抛小球

### 题目描述

[题目链接](https://www.nowcoder.com/practice/ae45a1d8bc1d43858c83762fe8c2802c?tpId=49&tqId=29306&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

小东和三个朋友一起在楼上抛小球，他们站在楼房的不同层，假设小东站的楼层距离地面N米，球从他手里自由落下，每次落地后反跳回上次下落高度的一半，并以此类推知道全部落到地面不跳，求4个小球一共经过了多少米？(数字都为整数)
给定四个整数A,B,C,D，请返回所求结果。

**测试样例**

>100,90,80,70
>返回：1020

### 代码

* 看了讨论才知道这题需要用到极限。
* 设楼层距离地面n米，则第一次落地共经过`n`米，第二次共经过`n+n*1/2*2`米，第三次共经过`n+n*1/2*2+n*1/4*2`米，第四次共经过`n+n*1/2*2+n*1/4*2+n*1/8*2`米，第m次共经过`n+n*1/2*2+...+n*(1/2)^(m-1)*2`米，计算得`n+2n(1/2+1/4+1/8+...+(1/2)^(m-1)=n+2n(1-(1/2)^(m-1))`，m趋于无穷（每次1/2一直不会为0），即`(1/2)^(m-1)=0`，所以m次共经过`n+2n(1-(1/2)^(m-1))=n+2n=3n`米。
* 所以答案为 3*(A+B+C+D) 
```python
# -*- coding:utf-8 -*-

class Balls:
    def calcDistance(self, A, B, C, D):
        return 3*(A+B+C+D)

# 运行时间：20ms
# 占用内存：3156k
```

## 数组单调和

### 题目描述

[题目链接](https://www.nowcoder.com/practice/8397609ba7054da382c4599d42e494f3?tpId=49&tqId=29364&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

现定义数组单调和为所有元素i的f(i)值之和。这里的f(i)函数定义为元素i左边(不包括其自身)小于等于它的数字之和。请设计一个高效算法，计算数组的单调和。
给定一个数组A同时给定数组的大小n，请返回数组的单调和。保证数组大小小于等于500，同时保证单调和不会超过int范围。

**测试样例**

>[1,3,5,2,4,6],6
>返回：27

### 代码

- 题目提示动态规划，暂时没有想到动态规划的解法，以下是常规求解
```python
# -*- coding:utf-8 -*-

class MonoSum:
    def calcMonoSum(self, A, n):
        ans = 0
        for i in range(1, n):
            for j in range(i):
                if A[j] <= A[i]:
                    ans += A[j]
        return ans

# 运行时间：100ms
# 占用内存：3156k
```

## 字符串通配

### 题目描述

[题目链接](https://www.nowcoder.com/practice/28acd1134e344040ad105b3786a79e7a?tpId=49&tqId=29355&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

对对于字符串A，其中绝对不含有字符’.’和’*’。再给定字符串B，其中可以含有’.’或’*’，’*’字符不能是B的首字符，并且任意两个’*’字符不相邻。exp中的’.’代表任何一个字符，B中的’*’表示’*’的前一个字符可以有0个或者多个。请写一个函数，判断A是否能被B匹配。
给定两个字符串A和B,同时给定两个串的长度lena和lenb，请返回一个bool值代表能否匹配。保证两串的长度均小于等于300。

**测试样例**

>"abcd",4,".*",2
>返回：true

### 代码

- 在这一题中，B仅由.和*组成，所以只需要考虑.和*的个数就能够ac
```python
# -*- coding:utf-8 -*-

class WildMatch:
    def chkWildMatch(self, A, lena, B, lenb):
        dotcount = B.count('.')
        starcount = B.count('*')
        if starcount == 0: # 如果没有*，则.的个数必须和lena相同才返回Ture，否则返回False
            if dotcount == lena:
                return True
            else:
                return False
        else:  # 如果有*，因为*可以匹配0或多个，所以只需要(dotcount - starcount) <= lena 就返回Ture，否则返回False
            if (dotcount - starcount) <= lena:
                return True
            else:
                return False
# 运行时间：20ms
# 占用内存：3156k
```
