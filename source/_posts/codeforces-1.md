---
title: Codeforces（1）：723A 716A 714A
date: 2016-10-22 14:43:07
tags: [Codeforces,python]
categories: codeforces
---

## 723A. The New Year: Meeting Friends

- **题目地址**
>http://codeforces.com/problemset/problem/723/A

- **题目大意**
>三个朋友住在一条线上，给出这三个点位置。他们要在某一个点庆祝新年，求总距离的最小值。

- **输入输出**
>input
7 1 4
output
6
input
30 20 10
output
20

- **分析**
>三个数排序，最小距离=最大值-最小值

- **相关代码**
```python
xi=sorted(map(int,raw_input().split()))
print xi[2]-xi[0]
```


## 716A. Crazy Computer

- **题目地址**
>http://codeforces.com/problemset/problem/716/A

- **题目大意**
>有n个词，如果两个词输入时间间超过c，那么屏幕清空。给出n,c以及每个词输入的时刻，求屏幕上最后有多少个词。

- **输入输出**
>input
6 5
1 3 8 14 19 20
output
3
input
6 1
1 3 5 7 9 10
output
2

- **分析**
>遍历一遍，如果两个时刻相差大于c，那么计数器置为1，否则计数器+1。

- **相关代码**
```python
n,c=map(int,raw_input().split())
ti=map(int,raw_input().split())
num=1

for i in range(1,n):
    if ti[i]-ti[i-1]>c:
        num=1
    else:
        num+=1
print num
```


## 714A. Meeting of Old Friends

- **题目地址**
>http://codeforces.com/problemset/problem/714/A

- **题目大意**
>给出5个数：l1, r1, l2, r2, k，A将在l1到r1时段清醒，B将在l2到r2时段去拜访A，时刻k他俩是不能一起的。求他们能在一起玩多久。

- **输入输出**
>input
1 10 9 20 1
output
2
input
1 100 50 200 75
output
50

- **分析**
>两个集合求交集的问题，分情况讨论。k如果在交集之中，最后结果需要-1。

- **相关代码**
```python
l1,r1,l2,r2,k=map(int,raw_input().split())

if l2>r1 or r2<l1:
    print 0
else:
    if r2<=r1:
        temp=max(l1,l2)
        ans=r2-temp+1
        if k>=temp and k<=r2:
            ans-=1
    else:
        temp=max(l1,l2)
        ans=r1-temp+1
        if k>=temp and k<=r1:
            ans-=1
    print ans
```