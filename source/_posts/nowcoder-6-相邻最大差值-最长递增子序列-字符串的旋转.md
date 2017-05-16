---
title: 'nowcoder(6):相邻最大差值; 最长递增子序列; 字符串的旋转'
date: 2017-05-16 09:39:21
tags: [nowcoder,python]
categories: nowcoder
---

## 相邻最大差值

### 题目描述

[题目链接](https://www.nowcoder.com/practice/376ede61d9654bc09dd7d9fa9a4b0bcd?tpId=49&tqId=29366&tPage=2&rp=2&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

请设计一个复杂度为O(n)的算法，计算一个未排序数组中排序后相邻元素的最大差值。
给定一个整数数组A和数组的大小n，请返回最大差值。保证数组元素个数大于等于2小于等于500。

**测试样例**
>[9,3,1,10],4
返回：6

### 代码

- 由于题中数组元素>=2且<=500（数组长度并不大），并且要求复杂度为O(n)，可以考虑桶排序。
- 申请长度为`max(A) - min(A) + 1`的bucket数组作为桶，遍历A，令`bucket[A[i] - min(A)] = 1`。
- 最后遍历bucket，连续0的长度的最大值+1即为最大差值。
```python
# -*- coding:utf-8 -*-

class MaxDivision:
    def findMaxDivision(self, A, n):
        bucket = [0 for i in range(max(A) - min(A) + 1)]
        for i in range(n):
            bucket[A[i] - min(A)] = 1
        count = 1
        ans = 0
        for i in range(len(bucket)):
            if bucket[i] == 0:
                count += 1
            else:
                ans = max(ans, count)
                count = 1
        return ans

        # 运行时间：100ms
        # 占用内存：3156k

maxdivision = MaxDivision()
print maxdivision.findMaxDivision([208, 254, 473, 153, 389, 579, 398], 7) # 返回135
```

## 最长递增子序列

### 题目描述

[题目链接](https://www.nowcoder.com/practice/585d46a1447b4064b749f08c2ab9ce66?tpId=49&tqId=29347&tPage=2&rp=2&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

对于一个数字序列，请设计一个复杂度为O(nlogn)的算法，返回该序列的最长上升子序列的长度，
这里的子序列定义为这样一个序列U1，U2...，其中Ui < Ui+1，且A[Ui] < A[Ui+1]。
给定一个数字序列A及序列的长度n，请返回最长上升子序列的长度。

**测试样例**
>[2,1,4,3,1,5,6],7
返回：4

### 代码

- 复杂度为`O(n^2)`
使用length[]数组维护以当前元素结尾的递增子序列的长度：
如果`A[j] < A[i]`，则`length[i]=max(length[i],length[j]+1)`，
最后返回`max(length)`就是最长递增子序列长度。
```python
# -*- coding:utf-8 -*-

class AscentSequence:
    def findLongest(self, A, n):
        length = [1 for i in range(n)]
        ans = 1
        for i in range(1, n):
            for j in range(i):
                if A[j] < A[i] and length[j] + 1 > length[i]:
                    length[i] = length[j] + 1
            ans = max(ans, length[i])
        return ans

        #运行时间：350ms
        #占用内存：3156k

ascentsequence = AscentSequence()
print ascentsequence.findLongest([2, 1, 4, 3, 1, 5, 6], 7)
```

- 复杂度为`O(nlogn)`
使用数组B[i]来维护长度为i+1的递增子序列的最小末尾：
从左向右遍历数组A，如果A[i]大于B中最后一个元素，则将A[i]加入B中；
如果A[i]小于B中最后一个元素，则在B中找合适的位置j，用A[i]替换B[j]。
**这里注意在B中找合适位置使用二分法，以减少复杂度。**
```python
# -*- coding:utf-8 -*-

class AscentSequence:
    def findLongest(self, A, n):
        B = [0 for i in range(n)]
        b_endindex = 0  # 记录B最后一个元素的下标
        B[0] = A[0]
        for i in range(1, n):
            if A[i] > B[b_endindex]:
                b_endindex += 1
                B[b_endindex] = A[i]
            else:
                temp_index = self.findIndex(B, A[i], b_endindex)
                B[temp_index] = A[i]
        return b_endindex + 1

        # 运行时间：130ms
        # 占用内存：3156k

    # 二分法查找B中第一个比b大的元素的下标，作为替换位置
    def findIndex(self, B, b, b_endindex):
        left = 0
        right = b_endindex
        while (left < right):
            mid = (left + right) / 2
            if b == B[mid]:
                return mid
            elif b < B[mid]: #注意这里不是right=mid-1，因为要寻找的是第一个比b大的元素的下标，可能就是B[mid]，而B[mid-1]可能就小于b了。
                right = mid
            else:
                left = mid + 1
        return left


ascentsequence = AscentSequence()
print ascentsequence.findLongest([2, 1, 4, 3, 1, 5, 6], 7)
```
- 例如：
序列`A[0..8] = 2 1 5 3 6 4 8 9 7`，最终`B[0..4] = 1, 3, 4, 7, 9`，而不是`B[0..4] = 1, 3, 4, 8, 9`，这个1,3,4,7,9不是LIS字符串，7代表的意思是存储4位长度递增子序列的最小末尾是7。
当序列为`A[0..10]= 2 1 5 3 6 4 8 9 7 8 9`时，用B[i]来维护长度为i+1的递增子序列的最小末尾，最终`B[0..5]=1, 3, 4, 7, 8, 9`，得到正确答案6。

## 字符串的旋转

### 题目描述

[题目链接](https://www.nowcoder.com/practice/85062aa6016640d188a6a0daf9f5da0e?tpId=49&tqId=29375&tPage=2&rp=2&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

对于一个字符串，和字符串中的某一位置，请设计一个算法，将包括i位置在内的左侧部分移动到右边，将右侧部分移动到左边。
给定字符串A和它的长度n以及特定位置p，请返回旋转后的结果。

**测试样例**

>"ABCDEFGH",8,4
返回："FGHABCDE"

### 代码

- 找到截断点，用截断点右侧字符串连接左侧字符串。
```python
# -*- coding:utf-8 -*-

class StringRotation:
    def rotateString(self, A, n, p):
        return A[p + 1:] + A[:p + 1]

	# 运行时间：40ms
	# 占用内存：3156k

stringrotation = StringRotation()
print stringrotation.rotateString("ABCDEFGH", 8, 4)
```

- 两个A连接，从位置p+1开始，截取长度为n的串即为答案。
```python
# -*- coding:utf-8 -*-

class StringRotation:
    def rotateString(self, A, n, p):
        return (A + A)[p + 1:p + 1 + n]

	# 运行时间：30ms
	# 占用内存：3156k

stringrotation = StringRotation()
print stringrotation.rotateString("ABCDEFGH", 8, 4)
```