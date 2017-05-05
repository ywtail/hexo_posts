---
title: 'nowcoder(4):左右最值最大差; 年终奖; 最长公共子序列'
date: 2017-05-05 21:39:53
tags: [nowcoder,python]
categories: nowcoder
---
## 左右最值最大差

### 题目描述

[题目链接](https://www.nowcoder.com/practice/f5805cc389394cf69d89b29c0430ff27?tpId=49&tqId=29359&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

给定一个长度为N(N>1)的整型数组A，可以将A划分成左右两个部分，左部分A[0..K]，右部分A[K+1..N-1]，K可以取值的范围是[0,N-2]。求这么多划分方案中，左部分中的最大值减去右部分最大值的绝对值，最大是多少？
给定整数数组A和数组的大小n，请返回题目所求的答案。

**测试样例**

>[2,7,3,1,1],5
返回：6

### 代码

- 这一题思路很巧妙：先找最大值maxnum（这个最大值肯定是某一边的最值），再对比两个端点处(A[0]与A[n-1])的数值，数值大的与maxnum在一边，数值小的就是另一边的最值（假设A[n-1] < A[0]，那么A[n-1]就是右边的最值。因为继续往左扩充，如果A[n-2] < A[n-1]，那么右边的最大值依然是A[n-1]，如果A[n-2] > A[n-1]，那么显然右侧只包含A[n-1]一个元素时，左部分中的最大值减去右部分最大值的绝对值最大，最值依然是A[n-1]），求得的这两个值的差值就是答案。
```python
# -*- coding:utf-8 -*-

class MaxGap:
    def findMaxGap(self, A, n):
        maxnum = max(A)
        minnum = min(A[0], A[n - 1])
        return maxnum - minnum


maxgap = MaxGap()
print maxgap.findMaxGap([2, 7, 3, 1, 1], 5)

# 运行时间：40ms
# 占用内存：3156k
```

## 年终奖

### 题目描述

[题目链接](https://www.nowcoder.com/practice/72a99e28381a407991f2c96d8cb238ab?tpId=49&tqId=29305&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

小东所在公司要发年终奖，而小东恰好获得了最高福利，他要在公司年会上参与一个抽奖游戏，游戏在一个6 x 6的棋盘上进行，上面放着36个价值不等的礼物，每个小的棋盘上面放置着一个礼物，他需要从左上角开始游戏，每次只能向下或者向右移动一步，到达右下角停止，一路上的格子里的礼物小东都能拿到，请设计一个算法使小东拿到价值最高的礼物。
给定一个6 x 6的矩阵board，其中每个元素为对应格子的礼物价值,左上角为[0,0],请返回能获得的最大价值，保证每个礼物价值大于100小于1000。

### 代码

- 典型的动态规划算法。
```python
# -*- coding:utf-8 -*-

class Bonus:
    def getMost(self, board):
        for i in range(6):
            for j in range(6):
                board[i][j] += max(board[i][j - 1] if j > 0 else 0, board[i - 1][j] if i > 0 else 0)
        return board[5][5]


bonus = Bonus()
print bonus.getMost([[200, 700, 300, 100, 100, 400], [200, 700, 300, 100, 100, 400], [200, 700, 300, 100, 100, 400],
                     [200, 700, 300, 100, 100, 400], [200, 700, 300, 100, 100, 400], [200, 700, 300, 100, 100, 400]])


# 输出5300
# 运行时间：30ms
# 占用内存：3156k
```

## 最长公共子序列

### 题目描述

[题目链接](https://www.nowcoder.com/practice/c996bbb77dd447d681ec6907ccfb488a?tpId=49&tqId=29348&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

对于两个字符串，请设计一个高效算法，求他们的最长公共子序列的长度，这里的最长公共子序列定义为有两个序列U1,U2,U3...Un和V1,V2,V3...Vn,其中Ui&ltUi+1，Vi&ltVi+1。且A[Ui] == B[Vi]。
给定两个字符串A和B，同时给定两个串的长度n和m，请返回最长公共子序列的长度。保证两串长度均小于等于300。

**测试样例**

>"1A2C3D4B56",10,"B1D23CA45B6A",12
返回：6

### 代码

- 动态规划，将公共长度存到ans[N+1][M+1]数组中，求ans[i][j]。
```python
# -*- coding:utf-8 -*-

class LCS:
    def findLCS(self, A, n, B, m):
        ans = [[0 for i in range(m + 1)] for j in range(n + 1)] # 这样初始化不用考虑[i-1]和[j-1]是否越界的问题
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if A[i - 1] == B[j - 1]: 
                    ans[i][j] = ans[i - 1][j - 1] + 1
                else:
                    ans[i][j] = max(ans[i][j - 1], ans[i - 1][j])
        return ans[n][m]

        # 运行时间：240ms
        # 占用内存：3148k

lcs = LCS()
print lcs.findLCS("1A2C3D4B56", 10, "B1D23CA45B6A", 12)
```