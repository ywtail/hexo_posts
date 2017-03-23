---
title: codeforces（2）：732A 731A 712A
date: 2016-10-22 16:04:39
tags: [Codeforces,python]
categories: codeforces
---

## 732A. Buy a Shovel

- **题目地址**
>http://codeforces.com/problemset/problem/732/A

- **题目大意**
>某人去买大铁锹，口袋有无数个10元硬币，以及1个r元硬币 (1 ≤ r ≤ 9)。已知大铁锹的单价k和r，求至少买几把锹能不找钱。

- **输入输出**
>input
117 3
output
9
input
237 7
output
1
input
15 2
output
2

- **分析**
>最后一位（“个”位）对上就行。例如单价117只要7*i的个位数是3或者0，剩余的钱用10元硬币支付就可以不找钱。最少买1把锹（题目要求），最多买10把（全用10元硬币支付）。

- **相关代码**
```python
k,r=map(int,raw_input().split())
t=k%10
for i in range(1,11):
    if t*i%10==0 or t*i%10==r:
        print i
        break
```


## 731A. Night at the Museum

- **题目地址**
>http://codeforces.com/problemset/problem/731/A

- **题目大意**
>一个大轮子上顺时针写着a-z，可以顺时针或逆时针转，初始位置是a。输入一个字符串，问最少转多少格。

- **输入输出**
>input
zeus
output
18
input
map
output
35
input
ares
output
34

- **分析**
>把字符'a'加到输入字符串最前面，遍历这个字符串，如果相邻俩字符的差temp大于13，则逆时针转，走的格数是26-ttemp。

- **相关代码**
```python
s='a'+raw_input()
ans=0
for i in range(1,len(s)):
    temp=abs(ord(s[i])-ord(s[i-1]))
    if temp>13:
        temp=26-temp
    ans+=temp
print ans
```


## 712A. Memory and Crow

- **题目地址**
>http://codeforces.com/problemset/problem/712/A

- **题目大意**
>给出n个数：a1,a2,...,an,已知ai = bi - b(i + 1) + b(i + 2) - b(i + 3)....，求bi。例如在输入输出第一个例子中，6 = 2 - 4 + 6 - 1 + 3，- 4 = 4 - 6 + 1 - 3。

- **输入输出**
>input
5
6 -4 8 -2 3
output
2 4 6 1 3 
input
5
3 -2 -1 5 6
output
1 -3 4 11 6 

- **分析**
>找规律发现，a1+a2=b1，a2+a3=b2，... ，an=bn

- **相关代码**
```python
n=int(raw_input())
ai=map(int,raw_input().split())
for i in range(n-1):
    print ai[i]+ai[i+1],
print ai[n-1]
```