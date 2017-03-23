---
title: codeforces（3）：727A 724A 722A
date: 2016-10-22 20:37:49
tags: [Codeforces,python]
categories: codeforces
---

## 727A. Transformation: from A to B

- **题目地址**
>http://codeforces.com/problemset/problem/727/A

- **题目大意**
>给2个数a,b，判断能不能通过以下两个步骤从a变换到b：
(1) replace the number x by 2·x; 
(2) replace the number x by 10·x + 1
如果不能输出"NO"，如果能，输出3行："YES"；变换序列的长度；变换序列

- **输入输出**
>input
2 162
output
YES
5
2 4 8 81 162 
input
4 42
output
NO
input
100 40021
output
YES
5
100 200 2001 4002 40021 

- **分析**
>从b往a推：如果b是偶数，必然是通过变换(1)得到的；如果b是奇数，则只能通过(2)得到。
如果b是奇数，且b-1不能被10整除，则直接输出"NO"。

- **相关代码**
```python
a,b=map(int,raw_input().split())
outputs=[b]
num=1
while b>a:
    if b%2==0:
        b=b/2
        outputs.append(b)
        num+=1
    else:
        if (b-1)%10==0:
            b=(b-1)/10
            outputs.append(b)
            num+=1
        else:
            break
if a==b:
    print "YES"
    print num
    for x in outputs[::-1]:
        print x,
else:
    print "NO"
```


## 724A. Checking the Calendar

- **题目地址**
>http://codeforces.com/problemset/problem/724/A

- **题目大意**
>给出一周中的两天（星期几），判断能否分别出现在非闰年的某一个月的第一天，以及下个月的第一天，这两个月必须在同一年。非闰年每个月的天数分别为：31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31。

- **输入输出**
>input
monday
tuesday
output
NO
input
sunday
sunday
output
YES
input
saturday
tuesday
output
YES

- **分析**
>根据天数来判断。一年的最后一个月是31天，很多个月都是31天，所以“这两个月必须在同一年”这个条件对解题没有限制。
天数只有28,30,31三种情况：经过28天，星期几不变，即a==b是可以满足条件的，经过30天，星期几往后数2天，同理，31往后数3天。
为了方便计算，构建列表，用index将星期几映射为整数。

- **相关代码**
```python
letters=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
a=letters.index(raw_input())
b=letters.index(raw_input())
if a==b or (a+2)%7==b or (a+3)%7==b:
    print "YES"
else:
    print "NO"
```


## 722A. Broken Clock

- **题目地址**
>http://codeforces.com/problemset/problem/722/A

- **题目大意**
>校正时钟：指明是12小时制还是24小时制，给出一个时间，修改尽量少的位数来使时间显示达到要求。有多种答案时给出一个答案。
校正标准：In 12-hours format hours change from 1 to 12, while in 24-hours it changes from 0 to 23. In both formats minutes change from 0 to 59.

- **输入输出**
>input
24
17:30
output
17:30
input
12
17:30
output
07:30
input
24
99:99
output
09:09

- **分析**
>“时”部分分情况讨论；“分”部分如果>=60，则将第一位改为'0'

- **相关代码**
```python
n=int(raw_input())
time=raw_input().split(':')
h=int(time[0])
m=int(time[1])

if m>=60:
    time[1]='0'+time[1][1]

if n==24:
    if h>23:
        time[0]='0'+time[0][1]
else:
    if h==0:
        time[0]='10'
    elif h>12:
        if time[0][1]=='0':
            time[0]='10'
        else:
            time[0]='0'+time[0][1]

print time[0]+':'+time[1]
```