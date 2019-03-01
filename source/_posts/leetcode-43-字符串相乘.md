---
title: leetcode-43-字符串相乘
date: 2019-03-01 17:15:45
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)

## 题目描述

给定两个以字符串形式表示的非负整数 `num1` 和 `num2`，返回 `num1` 和 `num2` 的乘积，它们的乘积也表示为字符串形式。

**示例 1:**

```
输入: num1 = "2", num2 = "3"
输出: "6"
```

**示例 2:**

```
输入: num1 = "123", num2 = "456"
输出: "56088"
```

**说明：**

1. `num1` 和 `num2` 的长度小于110。
2. `num1` 和 `num2` 只包含数字 `0-9`。
3. `num1` 和 `num2` 均不以零开头，除非是数字 0 本身。
4. **不能使用任何标准库的大数类型（比如 BigInteger）**或**直接将输入转换为整数来处理**。

## 解答

### 方法一

直接转为整数计算，理论上不符合题意，但是目前leetcode无法check。大致有以下2种方式

```python
return str(int(num1) * int(num2))
return str(eval(num1)*eval(num2))
```

### 方法二

竖乘法，将手工运算方法转为代码。这种方式需要注意细节。

### 方法三

这种方法来自：[Easiest JAVA Solution with Graph Explanation](https://leetcode.com/problems/multiply-strings/discuss/17605/Easiest-JAVA-Solution-with-Graph-Explanation)

`a*b=c`，则`len(c)<=len(a)+len(b)`，如`999*999 < 999*1000`，所以`999*999`的值不会超过6位，可以设置长度为`len(a)+len(b)`的列表来存最后的结果。

使用竖乘法，num1的第i位和num2的第j位，结果存在`i+j`和`i+j+1`位

```python
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        ans = [0 for i in range(len(num1) + len(num2))]

        for i in range(len(num1))[::-1]:
            for j in range(len(num2))[::-1]:
                mul = eval(num1[i]) * eval(num2[j])
                ans[i + j + 1] += mul  # 先加到ans[i+j+1]再处理进位；而不是分2位加到ans[i+j]和ans[i+j+1]再处理进位。

                ans[i + j] += ans[i + j + 1] // 10
                ans[i + j + 1] %= 10

        # 处理高位的0
        ans_str = ""
        for i in range(len(ans)):
            if not (len(ans_str) == 0 and ans[i] == 0):  # 首位的0跳过
                ans_str += str(ans[i])

        if len(ans_str) == 0:
            return "0"
        return ans_str
```