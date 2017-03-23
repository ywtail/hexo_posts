---
title: codeforces（4）：659A 656A 634A
date: 2016-10-23 10:13:27
tags: [Codeforces,python]
categories: codeforces
---

## 659A. Round House

- **题目地址**
>http://codeforces.com/problemset/problem/659/A

- **题目大意**
>一个圆房子，有n个入口，1到n顺时针排列。小明现在在a入口，问经过b个入口小明在哪个入口。b为正表示顺时针走，b为负逆时针。

- **输入输出**
>input
6 2 -5
output
3
input
5 1 3
output
4
input
3 2 7
output
3

- **分析**
>逆时针经过b个入口（-b），表示顺时针经过(n-b)%n个入口。所以此刻在a+(n+b)%n入口，注意这个值有可能>n，所以需要模n。

- **相关代码**
```python
n,a,b=map(int,raw_input().split())
ans=a+(n+b)%n
if ans>n:
    ans=ans%n
print ans
```


## 656A. Da Vinci Powers

- **题目地址**
>http://codeforces.com/problemset/problem/656/A

- **题目大意**
>输入a(0 ≤ a ≤ 35)，输出一个整数。

- **输入输出**
>input
3
output
8
input
10
output
1024
input
35
output
33940307968

- **分析**
>并不是求2的阶乘
Da Vinci在计算2的13次方时，使用4096*2=8192,少进了一位，变成8092，后面的直接乘以2就都是错的

- **相关代码**
```python
n=int(raw_input())
x=8092
if n<13:
    print 2**n
else:
    print 2**(n-13)*x
```


## 634A. Island Puzzle

- **题目地址**
>http://codeforces.com/problemset/problem/634/A

- **题目大意**
>有一个环形岛屿链，其中岛屿编号从1到n。给出这n个岛上雕像的个数，以及期望的雕像个数，问能不能通过移动雕像达到期望。
在这n个岛中，只有一个岛上没有雕像，移动规则：每次只能把雕像移到雕像数为0的岛上。

- **输入输出**
>input
3
1 0 2
2 0 1
output
YES
input
2
1 0
0 1
output
YES
input
4
1 2 3 0
0 3 2 1
output
NO

- **分析**
>只能往没有雕像的岛屿移动，所以只有0在不停移动，这些岛屿中雕像数的相对位置是不变的。
ai存当前雕像序列，bi存期望雕像序列，将0从序列中删除。
`temp=ai.index(bi[0])`定位bi[0]在ai中的位置，bi序列只可能是`ai[temp:]+ai[:temp]`。

- **相关代码**
```python
def func():
    n=int(raw_input())
    ai=map(int,raw_input().split())
    bi=map(int,raw_input().split())
    del ai[ai.index(0)]
    del bi[bi.index(0)]
    temp=ai.index(bi[0])
    ai=ai[temp:]+ai[:temp]
    for i in range(n-1):
        if ai[i]!=bi[i]:
            print "NO"
            return
    print "YES"
    return

func()
```