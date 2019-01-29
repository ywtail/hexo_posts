---
title: leetcode-5-最长回文子串
date: 2019-01-28 20:03:27
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

## 题目描述

给定一个字符串 `s`，找到 `s` 中最长的回文子串。你可以假设 `s` 的最大长度为 1000。

**示例 1：**

```
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
```

**示例 2：**

```
输入: "cbbd"
输出: "bb"
```

## 解答

### 方法一

遍历字符串s的所有子串，判断是否为回文串。找出长度最大的回文串。

```python
class Solution:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        r_s = s[::-1]
        n = len(s)
        max_len = 0
        ans = ""
        for x in range(1, n + 1):  # 子串长度
            for i in range(n + 1 - x):  # 偏移量
                if s[i:i + x] == s[i:i + x][::-1]:
                    if x > max_len:
                        max_len = x
                        ans = s[i:i + x]
        return ans
```

### 方法二

回文串是对称的，长度为奇数的回文串对称位置是中间的字符，偶数对称位置是中间2个字符间的空隙。可以遍历每个字符和中间位置，同时向左和右扩展，直到字符不同，或达到边界。为了不分奇偶情况讨论，在字符间填充符号`#`，遍历每个位置。

```python
class Solution:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        ss = '#' + '#'.join(s) + '#'
        n = len(ss)
        rem = [0] * n  # 记录各个位置最长回文串半径
        mid_index = 0
        max_r = 0
        for i in range(1, n):  # 当前位置
            for j in range(1, i + 1):  # 半径长度
                if (i + j) < n and ss[i - j] == ss[i + j]:
                    rem[i] += 1
                    if max_r < rem[i]:
                        max_r = rem[i]
                        mid_index = i
                else:
                    break
        return ss[(mid_index - max_r):(mid_index + max_r + 1)].replace("#","")
```

