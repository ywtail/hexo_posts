---
title: Learn Python the Hard Way读书笔记
date: 2017-03-14 22:49:11
tags: python
categories: python
---

- import test不用加.py
- 调用.py中的函数使用filename.functionname()
- pop(0)弹出list第一个，弹出后list中就没有了。再pop(0)就是原本第二个了
- pop(-1)弹出列表最后一个。
- sorted(words)对列表中元素进行排序
- if,elif,else只要检查到一个True就可以停下来了。
- 列表list=[1,'a',2]，可以赋值给一个变量。
- for number in list:这里number在循环开始时就被定义了，每次循环被重新定义一次。
- 列表和数组是不一样的吗？取决于语言和实现方式。在python里都叫列表。
- for i in range(1,3)其实是i=1,i=2。不包括3。
- for i in range(1,3)：list.append()在列表尾部追加元素。(list.insert(0,a)在index=0位置添加)
- `print(10*' ')`打印10个空格。必须是一个字符串*整数
- 添加元素到列表尾：append。
- del list[5] 删除该元素
- list[2:4] 元素2 3
- 元组fibs=(1,2,3);print(fibs[1])打印出2.元组与列表区别：元组一旦创建就不能修改了。
- 词语间的空格可以不打印出来。%把参数传过去，如果有多个，使用括号和逗号

```python
x=40
y=50
print "I have",x,y,"days"
print "I have%ddays" %x
print "I have %d %d days" %(x,y)

#I have 40 50 days
#I have40days
#I have 40 50 days
```

- 防止使用非ASCII字符遇到编码错误，在最顶端加上：

```python
# coding=utf-8
或者
# -*- coding: utf-8 -*-
```

- python pep:https://www.python.org/dev/peps/pep-0263/

```
 To define a source code encoding, a magic comment must
    be placed into the source files either as first or second
    line in the file, such as:

          # coding=<encoding name>

    or (using formats recognized by popular editors)

          #!/usr/bin/python
          # -*- coding: <encoding name> -*-

    or

          #!/usr/bin/python
          # vim: set fileencoding=<encoding name> :

 More precisely, the first or second line must match the regular
    expression "^[ \t\v]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)".
```

- 四舍五入

```python
print round(1.73)
print round(1.49)

#2.0
#1.0
```

- %r用来做调试，会显示原始数据（raw data），而%s和其他符号则是用来向用户显示输出的。如下多了引号。
- 打印`'''`和`"""`都行，风格问题

```python
x="Hello World"
print "It is %s" %x
print "It is %r" %x
#It is Hello World
#It is 'Hello World'
#用双引号打印单引号因为python会用最有效的方式打印出字符串，而不是完全按照写的方式。%r是用来调试和排错的，没有必要打印出好看的格式。

x="Hello\nWorld"
print "It is %s" %x
print "It is %r" %x
print """
Hello
Hello world
Hello Hello
"""

#输出 注意前后各多一个空行
It is Hello
World
It is 'Hello\nWorld'

Hello
Hello world
Hello Hello
```

- 创建字符串单双引号都可以，不过一般单引号用来创建简短的字符串。
- raw_input()输入提示

```python
x=raw_input("Age? ")
print x

#Age? 18
#18
```

- input()函数会把输入的东西当作python代码进行处理，会有安全问题，应该避开这个函数。使用raw_input()
- 可以接受参数的脚本。运行时

```python
# coding:utf-8
from sys import argv # 导入sys模块module中的argv

filename,a,b=argv # 将argv解包(unpack)，与其将所有参数放到一个变量下，不如将每个参数赋值给一个变量。
print filename
print a+"~"+b

# argv即参数变量（argument variable）,这个变量保存着运行python脚本时传递给python脚本的参数。

# input&output：
# $ python tempp.py hello world
# tempp.py
# hello~world
```

- 将-作为用户提示放入变量，不需要重复写raw_input()中的提示。

```python
# coding=utf-8
from sys import argv
script,username=argv
prompt='>'
print "Hi,"+username
print "How old"
age=raw_input(prompt)
print "Do u like me"
likes=raw_input(prompt)
print """You have been %s years old!
Game over!
I'm very OK""" %age


# $ python tempp.py Lcc
# Hi,Lcc
# How old
# >18
# Do u like me
# >No
# You have been 18 years old!
# Game over!
# I'm very OK
```

- 读取文件，先open，再read。把文件名写死不是一个好主意，所以使用argv或者raw_input()

```python
# coding=utf-8
from sys import argv
script,filename=argv

f1=open(filename) #返回的不是文件内容，是"file object"，read之后才返回内容
print f1
print f1.read()
f1.close() #处理完后将其关闭

file_again=raw_input("Please input filename:\n>")
f2=open(file_again)
print f2.readline()
f2.close()

# $ cat a.txt
# Hello!
# word!

# $ python tempp.py a.txt
# <open file 'a.txt', mode 'r' at 0x012BD230>
# Hello!
# word!
# Please input filename:
# >a.txt
# Hello!
```

- 文件相关命令

> close:关闭文件。跟编辑器的"保存"是一个意思
> read:读取文件内容。可以把结果赋值给一个变量
> readline:读取文本文件的一行
> truncate:清空文件，慎用
> write(stuff):将stuff写入文件

- 如果写了`open(filename).read()`，就不用再close了，因为read（）一旦运行，文件就被python关掉了。
- 写文件`open(filename,"w").write(stuff)`，“w”是必须的，因为open对文件的写入态度是安全第一，所以只有特别指定后才会进行写入操作。
- exists

```python
# coding=utf-8
from os.path import exists
print "Does a.txt exists?"
print exists("a.txt") #将文件名作为参数，如果存在返回Ture，否则返回False

# $ python tempp.py a.txt
# Does a.txt exists?
# True
```

- 运行run函数、调用call函数和使用use函数时一个意思
- 尽量避免变量的名称相同：全局变量和函数变量
- 每次运行.seek(0)就回到文件开始。seek函数的处理对象是字节而非行，seek(0)会转到文件的0byte(也就是第一个字节)的位置。`open("test.txt").seek(0)`
- 运行readline会读取一行，让后将磁头移动到`\n`后面 


- 为什么python中for循环可以使用未定义的变量？

> 循环开始时变量就被定义了，每次循环都会被重新定义一次。

- 判断字符串s中是否有字母x，可以用if "x" in s:
- `form sys import exit`。exit(0)可以中断某个程序，正常退出。exit(1)表示发生错误。
- `for x,y in mydict.items():`
- Python dict.get()方法

> get()方法返回给定键的值。如果键不可用，则返回默认值None。
> 语法：
> 以下是get()方法的语法：
> `dict.get(key, default=None)`
> 参数：
> key -- 这是要搜索在字典中的键。
> default -- 这是要返回键不存在的的情况下默认值。
> 返回值：
> 该方法返回一个给定键的值。如果键不可用，则返回默认值为None。
> 例子：

```python
dict = {'Name': 'Zara', 'Age': 27}

print "Value : %s" %  dict.get('Age')
print "Value : %s" %  dict.get('Sex', "Never")

#Value : 27
#Value : Never
```

- 类，为什创建__init__或者别的类函数时要多加一个self变量？

> 如果不加self，那么lyrics="Happy"这样的代码意义就不明确了，它指的可能是实例lyrics属性，也可能是一个叫做lyrics的局部变量。有了self.lyrics，就清楚地知道这指的是实例的属性self.lyrics

```python
# coding=utf-8

class Song(object):
  """docstring for Song"""
  def __init__(self, lyrics):
    self.lyrics = lyrics

  def sing_me_a_song(self):
    for line in self.lyrics:
      print line

happy_baby=Song(["Happy boy","Happy girl","Over"])

bulls_on_parade=Song(["They rally around the family","With pockets full of shells"])

happy_baby.sing_me_a_song()

bulls_on_parade.sing_me_a_song()

# $ python tempp.py
# Happy boy
# Happy girl
# Over
# They rally around the family
# With pockets full of shells
```

