---
title: leetcode-557-反转字符串中的单词III
date: 2019-02-27 21:59:46
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[反转字符串中的单词 III](https://leetcode-cn.com/problems/reverse-words-in-a-string-iii/)

## 题目描述

给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

**示例 1:**

```
输入: "Let's take LeetCode contest"
输出: "s'teL ekat edoCteeL tsetnoc" 
```

**注意：**在字符串中，每个单词由单个空格分隔，并且字符串中不会有任何额外的空格。

## 解答

### 方法一

先split，然后每个单词反转，最后拼接

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        ans = []
        for w in s.split():
            ans.append(w[::-1])
        return " ".join(ans)
```