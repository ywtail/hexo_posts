---
title: leetcode-15-三数之和
date: 2019-01-30 22:22:19
tags: [leetcode,python]
categories: leetcode
---

## 题目链接

[三数之和](https://leetcode-cn.com/problems/3sum/)

## 题目描述

给定一个包含 *n* 个整数的数组 `nums`，判断 `nums` 中是否存在三个元素 *a，b，c ，*使得 *a + b + c =* 0 ？找出所有满足条件且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

```
例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

## 解答

### 方法一

1. 对列表中元素排序
2. 遍历，对当前元素 `nums[i]`，相当于求 `nums[i+1:]` 的两数之和为`target = 0-nums[i]`。在位置 `i` 之前的元素不会再参与到计算中，因为假设在 `nums[j] (j<i)` 和某元素`nums[x]`之和为`target`，那么在遍历位置`j`时已经将` [nums[j],nums[i], nums[x]] `加入结果列表
3. 使用二分法求排序列表中两元素之和为 `target`

```python
class Solution:
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums = sorted(nums)
        n = len(nums)
        ans = []
        for i in range(n):
            if i > 0 and i < n and nums[i] == nums[i - 1]:
                continue
            target = 0 - nums[i]
            start = i + 1
            end = n - 1
            while start < end:
                if nums[start] + nums[end] < target:
                    start += 1
                elif nums[start] + nums[end] > target:
                    end -= 1
                else:
                    # 避免ans中有重复元素，这里应该能优化
                    if [nums[i], nums[start], nums[end]] not in ans:
                        ans.append([nums[i], nums[start], nums[end]])
                    start += 1
                    end -= 1
        return ans
```

一部分测试用例：

```
print(threeSum([1, -1, -1, 4, 2, 1, 0]))
# [[-1, -1, 2], [-1, 0, 1]]
print(threeSum([-2,0,0,2,2]))
# [[-2, 0, 2]]
```

