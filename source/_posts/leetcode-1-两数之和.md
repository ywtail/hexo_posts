---
title: leetcode-1-两数之和
date: 2019-01-26 23:39:22
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[两数之和](https://leetcode-cn.com/problems/two-sum/)

## 题目描述

给定一个整数数组 `nums` 和一个目标值 `target`，请你在该数组中找出和为目标值的那 **两个** 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

**示例:**

```
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```

## 解答

### 方法一

遍历两次，如果两个不同元素之和为target，返回下标。

```python
def twoSum(nums, target):
	n=len(nums)
	for i in range(n):
	    for j in range(i+1,n):
	        if nums[i]+nums[j]==target:
	            return [i,j]
```

### 方法二

遍历一次，使用字典m_dict记录当前元素下标和匹配的值，

1. 设当前元素为p，下标为i，那么应该寻找的元素q=target-p，使用字典m_dict记录{q:i}
2. 遍历数组时，如果当前元素x在字典m_dict中，则返回m_dict[x]和当前下标

```python
def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        m_dict={}
        for i in range(len(nums)):
            if nums[i] in m_dict:
                return [m_dict[nums[i]],i]
            m_dict[target-nums[i]]=i
```

## 代码

https://github.com/ywtail/leetcode/blob/master/1_1.py

