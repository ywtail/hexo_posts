---
title: leetcode-20-有效的括号
date: 2019-02-27 20:05:47
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

## 题目描述

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。

注意空字符串可被认为是有效字符串。

**示例 1:**

```
输入: "()"
输出: true
```

**示例 2:**

```
输入: "()[]{}"
输出: true
```

**示例 3:**

```
输入: "(]"
输出: false
```

**示例 4:**

```
输入: "([)]"
输出: false
```

**示例 5:**

```
输入: "{[]}"
输出: true
```

## 解答

### 方法一

使用列表模拟栈，能匹配就pop，不能则返回false。最终栈空返回true，否则返回false。

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        re_dict = {"(": ")", "{": "}", "[": "]"}
        stack = []
        for t in s:
            if t not in re_dict:
                if len(stack) == 0:
                    return False
                elif stack[-1] != t:
                    return False
                else:
                    stack.pop()
            else:
                stack.append(re_dict[t])
        if (len(stack)):
            return False
        return True
```

