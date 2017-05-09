---
title: 'nowcoder(5):最长公共子串; 股票交易日; 之字形打印矩阵'
date: 2017-05-09 21:44:58
tags: [nowcoder,python]
categories: nowcoder
---

## 最长公共子串

### 题目描述

[题目链接](https://www.nowcoder.com/practice/02e7cc263f8a49e8b1e1dc9c116f7602?tpId=49&tqId=29349&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

对于两个字符串，请设计一个时间复杂度为O(m*n)的算法(这里的m和n为两串的长度)，求出两串的最长公共子串的长度。
这里的最长公共子串的定义为两个序列U1,U2,..Un和V1,V2,...Vn，其中Ui + 1 == Ui+1,Vi + 1 == Vi+1，同时Ui == Vi。
给定两个字符串A和B，同时给定两串的长度n和m。

**测试样例**
>"1AB2345CD",9,"12345EF",7
返回：4

### 代码

- 动态规划。
```python
# -*- coding:utf-8 -*-

class LongestSubstring:
    def findLongest(self, A, n, B, m):
        table = [[0 for i in range(m + 1)] for j in range(n + 1)]  # 这样初始化不用考虑[i-1]和[j-1]是否越界的问题
        ans = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if A[i - 1] == B[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                    ans = max(ans, table[i][j])
        return ans

        # 运行时间：180ms
        # 占用内存：3156k

longestsubstring = LongestSubstring()
print longestsubstring.findLongest("1AB2345CD", 9, "12345EF", 7)
```

## 股票交易日

### 题目描述

[题目链接](https://www.nowcoder.com/practice/3e8c66829a7949d887334edaa5952c28?tpId=49&tqId=29317&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

在股市的交易日中，假设最多可进行两次买卖(即买和卖的次数均小于等于2)，规则是必须一笔成交后进行另一笔(即买-卖-买-卖的顺序进行)。
给出一天中的股票变化序列，请写一个程序计算一天可以获得的最大收益。请采用实践复杂度低的方法实现。
给定价格序列prices及它的长度n，请返回最大收益。保证长度小于等于500。

**测试样例**
>[10,22,5,75,65,80],6
返回：87

### 代码

- 求出以i点为分割点，左半段最大收益的数组leftprof，和右半段最大收益的数组rightprof。
然后遍历，找出最大的`leftprof[i]+rightprof[i]`组合。
```python
# -*- coding:utf-8 -*-

# coding=utf-8

class Stock:
    def maxProfit(self, prices, n):
        leftmin = prices[0]
        rightmax = prices[n - 1]
        leftprof = [0 for i in range(n)]
        rightprof = [0 for i in range(n)]
        sum = 0
        for i in range(1, n):
            leftmin = min(leftmin, prices[i]) # 从左向右找最小price记为leftmin
            leftprof[i] = max(leftprof[i - 1], prices[i] - leftmin)  
        for j in range(0, n - 1)[::-1]:
            rightmax = max(rightmax, prices[j]) # 从右向左找最大值记为rightmax
            rightprof[j] = max(rightprof[j + 1], rightmax - prices[j])
        for i in range(n): # 以i为分割点，左半段最大收益为leftprof[i]，右半段最大收益为rightprof[i]
            sum = max(sum, leftprof[i] + rightprof[i])
        return sum

        # 运行时间：50ms
        # 占用内存：3156k

stock = Stock()
print stock.maxProfit([10, 22, 5, 75, 65, 80], 6)

```

## 之字形打印矩阵

### 题目描述

[题目链接](https://www.nowcoder.com/practice/7df39c7556424eada267d8f793961a1e?tpId=49&tqId=29374&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

对于一个矩阵，请设计一个算法，将元素按“之”字形打印。具体见样例。
给定一个整数矩阵mat，以及他的维数nxm，请返回一个数组，其中元素依次为打印的数字。

**测试样例**

>[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],4,3
返回：[1,2,3,6,5,4,7,8,9,12,11,10]

### 代码

- 行是奇数时从左到右，行是偶数时从右到左。
```python
# -*- coding:utf-8 -*-

class Printer:
    def printMatrix(self, mat, n, m):
        ans = []
        for i in range(n):
            if i % 2 == 0:
                ans += mat[i]
            else:
                ans += mat[i][::-1]
        return ans
        
	# 运行时间：350ms
	# 占用内存：3148k

printer = Printer()
print printer.printMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 4, 3)
```