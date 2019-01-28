---
title: leetcode-4-寻找两个有序数组的中位数
date: 2019-01-28 16:35:36
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[寻找两个有序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

## 题目描述

给定两个大小为 m 和 n 的有序数组 `nums1` 和 `nums2`。

请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

你可以假设 `nums1` 和 `nums2` 不会同时为空。

**示例 1:**

```
nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0
```

**示例 2:**

```
nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5
```

## 解答

思路是对两个数组排序后求中位数。

### 方法一

使用`sorted`排序，取中位数

```python
def twoSum(nums, target):
	n=len(nums)
	for i in range(n):class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums = sorted(nums1 + nums2)
        l = len(nums)
        if l % 2 == 0:
            return (nums[l // 2 - 1] + nums[l // 2]) / 2
        else:
            return nums[l // 2]
```

### 方法二

自己实现排序，然后取中位数。

```python
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        m = len(nums1)
        n = len(nums2)
        sort_nums = []
        i = 0
        j = 0
        l = m + n
        now_index = 0
        while i < m and j < n:
            if nums1[i] < nums2[j]:
                sort_nums.append(nums1[i])
                i += 1
                now_index += 1
            else:
                sort_nums.append(nums2[j])
                j += 1
                now_index += 1
        if i < m:
            sort_nums.extend(nums1[i:])
        if j < n:
            sort_nums.extend(nums2[j:])

        if l % 2 == 0:
            print((sort_nums[l // 2 - 1] + sort_nums[l // 2]) / 2)
        else:
            print(sort_nums[l // 2])
```



