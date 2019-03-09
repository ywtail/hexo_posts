---
title: leetcode-238-除自身以外数组的乘积
date: 2019-03-05 21:12:43
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

## 题目描述

给定长度为 *n* 的整数数组 `nums`，其中 *n* > 1，返回输出数组 `output` ，其中 `output[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

**示例:**

```
输入: [1,2,3,4]
输出: [24,12,8,6]
```

**说明:** 请**不要使用除法，**且在 O(*n*) 时间复杂度内完成此题。

**进阶：**
你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组**不被视为**额外空间。）

## 解答

### 方法一

使用除法，注意有0的情况

```python
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n = len(nums)
        mul = 1
        zero_cnt = 0
        for i in range(n):
            if nums[i] == 0:
                zero_cnt += 1
                if zero_cnt > 1:  # 2个0的情况
                    return [0] * n
                continue
            mul *= nums[i]
        ans = []
        if zero_cnt > 0:  # 1个0的情况
            for i in range(n):
                if nums[i] == 0:
                    ans.append(mul)
                else:
                    ans.append(0)
            return ans
        for i in range(n):  # 没有0的情况
            ans.append(mul // nums[i])
        return ans
```

