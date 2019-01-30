---
title: leetcode-14-最长公共前缀
date: 2019-01-30 10:34:00
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

## 题目描述

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 `""`。

**示例 1:**

```
输入: ["flower","flow","flight"]
输出: "fl"
```

**示例 2:**

```
输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
```

**说明:**

所有输入只包含小写字母 `a-z` 。

## 解答

### 方法一

本题注意不要理解为公共子串。

```python
class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        ans = ""
        if len(strs) == 0:  # 输入[]需返回，否则min报错 min() arg is an empty sequence
            return ans
        min_len = min(map(len, strs))  # 列表中元素最短len，避免后续 string index out of range
        flag = 1
        for i in range(min_len):  # 列表中元素最短len
            for j in range(1, len(strs)):  # 列表中字符的个数
                if strs[0][i] == strs[j][i]:
                    continue
                else:
                    flag = 0
                    break
            if flag == 0:
                break
            ans += strs[0][i]
        return ans
```

一部分测试用例：

```
print(longestCommonPrefix([]))

print(longestCommonPrefix([""]))

print(longestCommonPrefix(["flower", "flow", "flight"]))
# fl

print(longestCommonPrefix(["dog", "racecar", "car"]))

print(longestCommonPrefix(["aa", "a"]))
# a
```

