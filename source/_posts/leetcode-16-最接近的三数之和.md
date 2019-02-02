---
title: leetcode-16-最接近的三数之和
date: 2019-02-02 16:13:53
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)

## 题目描述

给定一个包括 *n* 个整数的数组 `nums` 和 一个目标值 `target`。找出 `nums` 中的三个整数，使得它们的和与 `target` 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

```
例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.

与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
```

## 解答

### 方法一

注意这里只需要返回三个数的和，而不是返回这三个数。这里使用枚举法

```python
class Solution:
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        n = len(nums)
        ans = 0  # 存3数之和
        sub_min = float('inf')  # 存abs(target - s)的最小值
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    s = nums[i] + nums[j] + nums[k]
                    if s == target:  # 如果s==target直接返回
                        return s
                    if abs(target - s) < sub_min:
                        sub_min = abs(target - s)
                        ans = s
        return ans
```

一部分测试用例：

```
print(threeSumClosest([-1, 2, 1, -4], 1))
# 2
```

