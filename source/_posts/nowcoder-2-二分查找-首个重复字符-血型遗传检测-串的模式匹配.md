---
title: 'nowcoder(2):二分查找; 首个重复字符; 血型遗传检测; 串的模式匹配'
date: 2017-04-08 21:29:59
tags: [nowcoder,python]
categories: nowcoder
---

## 二分查找

### 题目描述

[题目链接](https://www.nowcoder.com/practice/28d5a9b7fc0b4a078c9a6d59830fb9b9?tpId=49&tqId=29278&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

对于一个有序数组，我们通常采用二分查找的方式来定位某一元素，请编写二分查找的算法，在数组中查找指定元素。
给定一个整数数组A及它的大小n，同时给定要查找的元素val，请返回它在数组中的位置(从0开始)，若不存在该元素，返回-1。若该元素出现多次，请返回第一次出现的位置。

**测试样例**

>[1,3,5,7,9],5,3
>返回：1

### 代码

- 普通的二分，需要注意的是元素出现多次时返回第一次出现的位置。所以在找到元素后，继续向左查找第一次出现的位置。

```python
# -*- coding:utf-8 -*-

class BinarySearch:
    def getPos(self, A, n, val):
        if n == 0:
            return -1
        start = 0
        end = n - 1
        while (start <= end):
            mid = (start + end) / 2
            if val == A[mid]:
                for i in range(1, mid + 1):
                    if A[mid - i] < A[mid]:
                        return mid - i + 1
                return 0
            elif val < A[mid]:
                end = mid - 1
            else:
                start = mid + 1
        return -1
```

>运行时间：50ms
>占用内存：3148k

- 在评论中看到另一种：在元素出现多次时，不是继续向左查找，而是end=mid，最后返回start指向的元素。

```python
# -*- coding:utf-8 -*-

class BinarySearch:
    def getPos(self, A, n, val):
        if n == 0:
            return -1
        start = 0
        end = n - 1
        while (start < end):
            mid = (start + end) / 2
            if val == A[mid]:
                end=mid
            elif val < A[mid]:
                end = mid - 1
            else:
                start = mid + 1
        if A[start]==val:
            return start
        return -1
```

> 运行时间：50ms
> 占用内存：3148k

## 首个重复字符

### 题目描述

[题目链接](https://www.nowcoder.com/practice/dab59997905b4459a42587fece8a75f4?tpId=49&tqId=29279&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

对于一个字符串，请设计一个高效算法，找到第一次重复出现的字符。
给定一个字符串(不一定全为字母)A及它的长度n。请返回第一个重复出现的字符。保证字符串中有重复字符，字符串的长度小于等于500。

**测试样例**

>"qywyer23tdd",11
>返回：y

### 代码

- 用字典记录，如果出现过就返回。

```python
# -*- coding:utf-8 -*-

class FirstRepeat:
    def findFirstRepeat(self, A, n):
        re = {}
        for k in A:
            if k in re:
                return k
            else:
                re[k] = 0 #只是将k放入字典中，不一定=0（等于多少对答案无影响）
```

> 运行时间：20ms
> 占用内存：3156k

## 血型遗传检测

### 题目描述

[题目链接](https://www.nowcoder.com/practice/5541c433dee04c17ba7774c4a20430de?tpId=49&tqId=29303&tPage=3&rp=3&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

血型遗传对照表如下

| 父母血型  | 子女会出现的血型 | 子女不会出现的血型 |
| ----- | -------- | --------- |
| O与O   | O        | A,B,AB    |
| A与O   | A,O      | B,AB      |
| A与A   | A,O      | B,AB      |
| A与B   | A,B,AB,O | ——        |
| A与AB  | A,B,AB   | O         |
| B与O   | B,O      | A,AB      |
| B与B   | B,O      | A,AB      |
| B与AB  | A,B,AB   | O         |
| AB与O  | A,B      | O,AB      |
| AB与AB | A,B,AB   | O         |

请实现一个程序，输入父母血型，判断孩子可能的血型。
给定两个字符串father和mother，代表父母的血型,请返回一个字符串数组，代表孩子的可能血型(按照字典序排列)。

**测试样例**

>”A”,”A”
>返回：[”A”,“O”]

### 代码

- 总结了一下，一共就三种情况：（*注意题目中说了要按字典序排序*）
  - 含AB（AB与O是特例），则孩子可能血型是['A', 'AB', 'B']，特例（AB与O）则去掉'AB'这种可能
  - A和B，则孩子可能血型是['A','AB', 'B', 'O']
  - 普通，则孩子可能血型是{father, mother, 'O'}，这里注意去重

```python
# -*- coding:utf-8 -*-

class ChkBloodType:
    def chkBlood(self, father, mother):
        if 'AB' in [father, mother]:
            ans = ['A', 'AB', 'B']
            if 'O' in [father, mother]:
                ans.remove('AB')
        elif 'A' in [father, mother] and 'B' in [father, mother]:
            ans = ['A','AB', 'B', 'O']
        else:
            ans = list({father, mother, 'O'})
        return ans
```

> 运行时间：30ms
> 占用内存：3156k

## 串的模式匹配

### 题目描述

[题目链接](https://www.nowcoder.com/practice/084b6cb2ca934d7daad55355b4445f8a?tpId=49&tqId=29363&tPage=1&rp=1&ru=/ta/2016test&qru=/ta/2016test/question-ranking)

对于两个字符串A，B。请设计一个高效算法，找到B在A中第一次出现的起始位置。若B未在A中出现，则返回-1。
给定两个字符串A和B，及它们的长度lena和lenb，请返回题目所求的答案。

**测试样例**

>"acbc",4,"bc",2
>返回：2

### 代码

- find实现

```python
# -*- coding:utf-8 -*-

class StringPattern:
    def findAppearance(self, A, lena, B, lenb):
        return A.find(B)
```

> 运行时间：30ms
> 占用内存：3156k

- 老实写一遍：当B长度小于A时，返回-1；遍历A，如果A[i]与B[0]相等，则截取A[i:i + lenb]与B对比，如果相同就返回当前i

```python
# -*- coding:utf-8 -*-

class StringPattern:
    def findAppearance(self, A, lena, B, lenb):
        if lena < lenb:
            return -1
        for i in range(0, lena - lenb + 1):
            if A[i] == B[0]:
                if A[i:i + lenb] == B:
                    return i
        return -1
```

> 运行时间：10ms
> 占用内存：3156k