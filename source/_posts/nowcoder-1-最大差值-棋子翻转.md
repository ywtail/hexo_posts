---
title: 'nowcoder(1):最大差值 棋子翻转'
date: 2017-03-19 22:17:46
tags: [nowcoder,python]
---

## 最大差值

[题目链接](https://www.nowcoder.com/practice/1f7675ae7a9e40e4bd04eb754b62fd00?tpId=49&tqId=29281&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

**题目描述**
有一个长为n的数组A，求满足0≤a≤b<n的A[b]-A[a]的最大值。
给定数组A及它的大小n，请返回最大差值。

**测试样例：**

> [10,5],2
>
> 返回：0

- 最简单的思路是遍历两遍，求最大差值。时间复杂度O(n^2)。

```python
# -*- coding:utf-8 -*-

class LongestDistance:
    def getDis(self, A, n):
        ma = 0
        for i in range(n):
            for j in range(i + 1, n):
                ma = max(A[j] - A[i], ma)
        return ma
```

> 运行时间：150ms
>
> 占用内存：3156k

- 另一种只遍历一遍，使用start来记录起始位置，star始终是遍历过的最小值。时间复杂度O(n)。

```python
# -*- coding:utf-8 -*-

class LongestDistance:
    def getDis(self, A, n):
        start = A[0]
        ma = 0
        for i in range(n):
            start = min(start, A[i])
            ma = max(ma, A[i] - start)
        return ma
```

> 运行时间：40ms
>
> 占用内存：3156k

## 棋子翻转

[题目链接](https://www.nowcoder.com/practice/0b5ab6cc51804dd59f9988ad70d8c4a0?tpId=49&tqId=29282&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

**题目描述**

在4x4的棋盘上摆满了黑白棋子，黑白两色的位置和数目随机其中左上角坐标为(1,1),右下角坐标为(4,4),现在依次有一些翻转操作，要对一些给定支点坐标为中心的上下左右四个棋子的颜色进行翻转，请计算出翻转后的棋盘颜色。

给定两个数组**A**和**f**,分别为初始棋盘和翻转位置。其中翻转位置共有3个。请返回翻转后的棋盘。

**测试样例：**

> [[0,0,1,1],[1,0,1,0],[0,1,1,0],[0,0,1,0]],[[2,2],[3,3],[4,4]]
> 返回：[[0,1,1,1],[0,0,1,0],[0,1,1,0],[0,0,1,0]]

- 思路很简单，但是写起来费事，非常容易出现幼稚的细节错误（尤其是像下面写的这么啰嗦）。

```python
# -*- coding:utf-8 -*-

class Flip:
    def flipChess(self, A, f):
        for (a, b) in f:
            if a == 1:
                A[a][b - 1] = (A[a][b - 1] + 1) % 2
            elif a == 4:
                A[a - 2][b - 1] = (A[a - 2][b - 1] + 1) % 2
            else:
                A[a][b - 1] = (A[a][b - 1] + 1) % 2
                A[a - 2][b - 1] = (A[a - 2][b - 1] + 1) % 2
            if b == 1:
                A[a - 1][b] = (A[a - 1][b] + 1) % 2
            elif b == 4:
                A[a - 1][b - 2] = (A[a - 1][b - 2] + 1) % 2
            else:
                A[a - 1][b] = (A[a - 1][b] + 1) % 2
                A[a - 1][b - 2] = (A[a - 1][b - 2] + 1) % 2
        return A
```

> 运行时间：40ms
>
> 占用内存：3156k

- 代码太长了，改短一点：

```python
# -*- coding:utf-8 -*-

class Flip:
    def flipChess(self, A, f):
        for (a, b) in f:
            if a < 4:
                A[a][b - 1] = (A[a][b - 1] + 1) % 2
            if a > 1:
                A[a - 2][b - 1] = (A[a - 2][b - 1] + 1) % 2
            if b < 4:
                A[a - 1][b] = (A[a - 1][b] + 1) % 2
            if b > 1:
                A[a - 1][b - 2] = (A[a - 1][b - 2] + 1) % 2
        return A
```

> 运行时间：30ms
>
> 占用内存：3160k 