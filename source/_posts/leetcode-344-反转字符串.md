---
title: leetcode-344-反转字符串
date: 2019-02-27 21:49:24
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[反转字符串](https://leetcode-cn.com/problems/reverse-string/)

## 题目描述

编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 `char[]` 的形式给出。

不要给另外的数组分配额外的空间，你必须**原地修改输入数组**、使用 O(1) 的额外空间解决这一问题。

你可以假设数组中的所有字符都是 [ASCII](https://baike.baidu.com/item/ASCII) 码表中的可打印字符。

**示例 1：**

```
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]
```

**示例 2：**

```
输入：["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]
```

## 解答

### 方法一

从头尾两个方向遍历列表，交换元素

```python
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        n = len(s)
        if n <= 1:
            return
        start = 0
        end = n - 1
        while (start < end):
            s[start], s[end] = s[end], s[start]
            start += 1
            end -= 1
```