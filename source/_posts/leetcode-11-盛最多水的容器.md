---
title: leetcode-11-盛最多水的容器
date: 2019-02-28 21:58:49
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

## 题目描述

给定 *n* 个非负整数 *a*1，*a*2，...，*a*n，每个数代表坐标中的一个点 (*i*, *ai*) 。在坐标内画 *n* 条垂直线，垂直线 *i* 的两个端点分别为 (*i*, *ai*) 和 (*i*, 0)。找出其中的两条线，使得它们与 *x* 轴共同构成的容器可以容纳最多的水。

**说明：**你不能倾斜容器，且 *n* 的值至少为 2。

![img](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

**示例:**

```
输入: [1,8,6,2,5,4,8,3,7]
输出: 49
```

## 解答

### 方法一

枚举法，提交超时，时间复杂度为O(n^2)

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        ans = 0
        for i in range(n):
            for j in range(i + 1, n):
                ans = max(ans, (j - i) * min(height[i], height[j]))
        return ans
```

### 方法二

双指针法。使用2个指针指向列表的首尾，向中间移动，使用ans记录最大面积，更新指向值更小的指针。这种方式只需遍历一遍，时间复杂度为O(n)

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        l = 0
        r = len(height) - 1
        ans = 0
        while l < r:
            ans = max(ans, (r - l) * min(height[l], height[r]))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return ans
```



