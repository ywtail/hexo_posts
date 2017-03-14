---
title: 刷题使用过的python小技巧
date: 2017-03-14 08:23:22
tags: [python,Codeforces]
---

在刷题的过程中，使用了一些python常用的技巧，汇总如下（出处或更详细的解释会在小标题下方标明）。

## 字符串

### replace

- `str.replace(old, new[, max])`

> old -- 将被替换的子字符串。
> new -- 新字符串，用于替换old子字符串。
> max -- 可选字符串, 替换不超过 max 次

### strip去空格或字符

- 去掉字符串前后空格，可以使用strip，lstrip，rstrip方法
```python
>>> a="abc".center (30)  
>>> a  
'             abc              '  
>>> b=a.lstrip ()  
>>> b  
'abc              '  
>>> c=a.rstrip ()  
>>> c  
'             abc'  
>>> d=a.strip ()  
>>> d  
'abc'
```

- 这三个方法默认是去掉空格，也可以通过参数去掉其他字符，等价与replace
```python
>>> a="abc"  
>>> b=a.strip ('a')  
>>> b  
'bc'  
>>> c=a.replace ('a','')  
>>> c  
'bc' 
```

- 去除中间空格先split()，再join
```python
s=' hello   world hello'
print ''.join(s.split())
#helloworldhello
```

- **注意：**join只能连接字符串列表！不能连接整数列表。对整数列表先转换为字符串列表。
```python
x=[1,2,3]
#print ''.join(x)
#TypeError: sequence item 0: expected string, int found
print ''.join(map(str,x))
#123
```

### 判断字符串是否为整数或字母isalnum

- s.isdigit()判断是否为数字，s.isalpha()判断是否为字母，s.isalnum()判断是否为数字或字母。注意:这里有括号。
- 与filter结合能够筛选出字符串中的整数、字母、整数字母构成的串。注意，这里无括号。
```python
print '123'.isdigit()
#True
print '123a'.isdigit()
#False
print '123a'.isdigit
#<built-in method isdigit of str object at 0x017394C0>

print 'abc1'.isalpha()
#False
print 'abc1'.isalnum()
#True
print 'abc#1'.isalpha()
#False

s='dade142.;!0142f[.,]ad'
print filter(str.isdigit,s)
#1420142
print filter(str.isalpha,s)
#dadefad
print filter(str.isalnum,s)
#dade1420142fad
```

### 全排列permutations

参考[PYTHON-进阶-ITERTOOLS模块小结](http://wklken.me/posts/2013/08/20/python-extra-itertools.html#itertoolspermutationsiterable-r "")
- `itertools.permutations(iterable[, r])`
创建一个迭代器，返回iterable中所有长度为r的项目序列，如果省略了r，那么序列的长度与iterable中的项目数量相同：返回p中任意取r个元素做排列的元组的迭代器
```python
import itertools
print itertools.permutations('123',3)
#<itertools.permutations object at 0x016CAE70>
print list(itertools.permutations('123',3))
#[('1', '2', '3'), ('1', '3', '2'), ('2', '1', '3'), ('2', '3', '1'), ('3', '1','2'), ('3', '2', '1')]
for x in list(itertools.permutations('123',3)):
	print ''.join(x),
#123 132 213 231 312 321
```


## 列表及字典
### 去除列表中重复元素
参考：[Python去除列表中重复元素的方法](http://www.jb51.net/article/62587.htm "")
- set
```python
l1 = [1,4,5,6,2,3,1,3,5,3]
print set(l1)
#set([1, 2, 3, 4, 5, 6])
l2 = list(set(l1))
print l2
#[1, 2, 3, 4, 5, 6]
```

- 还有一种据说速度更快的，没测试过两者的速度差别
```python
l1 = [1,4,5,6,2,3,1,3,5,3]
print {}.fromkeys(l1).keys()
#[1, 2, 3, 4, 5, 6]
```

- 这两种都有个缺点，祛除重复元素后排序变了。如果想要保持他们原来的排序，使用list类的sort方法
```python
l1 = [1,4,5,6,2,3,1,3,5,3]
l2 = list(set(l1))
l2.sort(key=l1.index)
print l2
#[1, 4, 5, 6, 2, 3]
```

- 也可以按照如下的方式写
```python
l1 = [1,4,5,6,2,3,1,3,5,3]
l2 = sorted(set(l1),key=l1.index)
print l2
#[1, 4, 5, 6, 2, 3]
```

- 还可以使用最普通的遍历
```python
l1 = [1,4,5,6,2,3,1,3,5,3]
l2 = []
for x in l1:
	if x not in l2:
		l2.append(x)
print l2
#[1, 4, 5, 6, 2, 3]
```

- 遍历也可以按照如下的方式写
```python
l1 = [1,4,5,6,2,3,1,3,5,3]
l2=[]
[l2.append(x) for x in l1 if x not in l2]
print l2
#[1, 4, 5, 6, 2, 3]
```

### 统计元素出现次数

- 最简单的方法是从collections引入Counter包，出现频度最高的元素会默认在前面。可用dict()操作符将其转换为一个普通的dict来进行额外处理(转成dict并不是按出现频度排序的)。
```python
from collections import Counter
l1 = [1,4,5,6,2,3,1,3,5,3]
print Counter(l1)
#Counter({3: 3, 1: 2, 5: 2, 2: 1, 4: 1, 6: 1})
print dict(Counter(l1))
#{1: 2, 2: 1, 3: 3, 4: 1, 5: 2, 6: 1}
```

### 列表排序

参考[python sort、sorted高级排序技巧](http://www.jb51.net/article/57678.htm "")
- 先对第一个数排序，再对第二个数排序。
```python
from operator import itemgetter
data=[[1,3],[1,2],[2,3],[1,1],[3,1],[2,2]]
print sorted(data,key=itemgetter(0))
#[[1, 3], [1, 2], [1, 1], [2, 3], [2, 2], [3, 1]]
print sorted(data,key=itemgetter(1))
#[[1, 1], [3, 1], [1, 2], [2, 2], [1, 3], [2, 3]]
print sorted(data,key=itemgetter(0,1))
#[[1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 1]]
```

### 字典排序

参考[深入Python(1): 字典排序 关于sort()、reversed()、sorted()](http://www.cnblogs.com/BeginMan/p/3193081.html "")
- 函数原型：sorted(dic,value,reverse)

> 解释：dic为比较函数，value 为排序的对象（这里指键或键值），
reverse：注明升序还是降序，True--降序，False--升序（默认）

```python
dic = {'a':31, 'bc':5, 'c':3, 'asd':4, '33':56, 'd':0}
print sorted(dic.iteritems(), key=lambda d:d[1], reverse = False )  
#[('d', 0), ('c', 3), ('asd', 4), ('bc', 5), ('a', 31), ('33', 56)]
```
> dic.iteritems()，返回字典键值对的元祖集合
> key=lambda d:d[1] 是将键值(value)作为排序对象。

## 其他

### import math

- math.ceil(1.2)=2.0
- math.floor(1.6)=1.0
- round(1.2)=1.0
- round(1.6)=2.0
- math.pi=3.141592653589793
- math.sqrt(4)=2.0
- 下取整不一定`import math` ,`math.floor(3.5)`
可以直接`int(3.5)`

### python2整数相除division
在python3中不需要考虑这个问题。
在python2中进行两个整数相除的时候，在默认情况下都是只能够得到整数的结果，而在需要进行对除所得的结果进行精确地求值时，想在运算后即得到浮点值，需要在进行除法运算前导入一个实除法的模块`from __future__ import division`，即可在两个整数进行相除的时候得到浮点的结果。


### 字符和ASCII转换
参考：[Python字符和字符值(ASCII或Unicode码值)转换方法](http://www.jb51.net/article/66457.htm "")

ASCII码（0~255范围）
- ord('A')：将字符转换为对应的ASCII数值，即‘A’-->65
- chr(65)：将数值转换为对应的ASCII字符，即65-->'A'

### 按位与或非

在python中，按位操作不需要转化成二进制，可直接使用整数操作。
- 按位与   ( bitwise and of x and y )
  &  举例： 5&3 = 1  解释： 101  11 相同位仅为个位1 ，故结果为 1
- 按位或   ( bitwise or of x and y ) 
  |  举例： 5|3 = 7  解释： 101  11 出现1的位是 1 1 1，故结果为 111
- 按位异或 ( bitwise exclusive or of x and y )
  ^  举例： 5^3 = 6  解释： 101  11 对位相加(不进位)是 1 1 0，故结果为 110
- 按位反转 (the bits of x inverted )
  ~  举例： ~5 = -6  解释： 将二进制数+1之后乘以-1，即~x = -(x+1)，-(101 + 1) = -110
```python
>>> 3&5
1
>>> 3|5
7
>>> 3^5
6
>>> ~5
-6
```

### 把if else写到一行

- 常规写法（只是举例，并不是要求最大值，如果要求最大值c=max(a,b)就ok了）
```python
a,b,c=1,2,3
if a>b:
	c=a
else:
	c=b
print c
#2
```
- 表达式写法
```python
a,b,c=1,2,3
c=a if a>b else b
print c
#2
```

- 二维列表写法
```python
a,b,c=1,2,3
c=[b,a][a>b]
print c
#2
```

### print格式
参考[跟老齐学Python之print详解](http://www.jb51.net/article/55768.htm "")

- `%2d`表示占2位，右对齐。注意逗号位置
```python
a=[[1,2,3],[4,5,6]]
for i in range(2):
	for j in range(3):
		print "%2d" %a[i][j],
	print ""
# 1  2  3
# 4  5  6
```

### Tips

python的有些写法与其它不同，偶尔容易记混淆，以下是记错过的。

- 不支持i++(需要写成i+=1)，不支持a>b?a:b
- 次方：2**3=8（`^`表示异或）
- `mylist.insert(0,x)`从索引0开始插入，再也不用[::-1]了
- 交换两个值：a,b=b,a
- `ntimes=list1.count("RL")`数字符串中有多少个RL
- `list1.find("RL",start)`，从start开始找RL，返回索引
- `sorted(times)`排序
- `sum([1,2,3])`列表求和
- 字符串中的字符不能直接赋值，可以使用切片来改变值。
- 字符串不能直接交换(a,b=b,a)，需要首先list(s)，再交换，最后join
- 多个变量赋值可以写一行`a,b,c=1,2,3`
- 二维列表：`book=[[0 for i in range(4)] for j in range(5)]` 5行4列
- 97%3=1
- -97%3=2
- 在使用sorted过程中，`sorted['6','534']=['534','6']`。要注意转换
- 报错`TypeError: expected a character buffer object`，在向文件中写入数据时，要求数据格式为str类型，比如：`outputFileHandler.write(0)`参数为Int类型，则会引发类似错误。解决方法：将非str数据，强制转换为str类型，如上改为:`outputFileHandler.write(str(0))`
- `list.remove(ele)`只能删除第一个匹配的
- log(8)/log(2)=log2(8)=3，位数多了在要求精确计算的情况下就得换方法，例如：0.9999999999999999999直接等于1。在刷题要求精度时需考虑这一点，适当选择计算方法。