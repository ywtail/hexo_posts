---
title: '绘图可视化 (1): matplotlib绘图基础'
date: 2017-06-04 21:30:53
tags: [python,matplotlib,可视化]
categories: python
top: 2
---

本文主要参考 [十分钟入门Matplotlib](http://codingpy.com/article/a-quick-intro-to-matplotlib/)

文中绘制的图形在 [这个链接](https://ywtail.github.io/python/%E5%8F%AF%E8%A7%86%E5%8C%96/%E7%BB%98%E5%9B%BE%E5%8F%AF%E8%A7%86%E5%8C%96-1-matplotlib%E7%BB%98%E5%9B%BE%E5%9F%BA%E7%A1%80.html) 中显示。

>Matplotlib 是 Python 的一个绘图库。它包含了大量的工具，你可以使用这些工具创建各种图形，包括简单的散点图，正弦曲线，甚至是三维图形。Python 科学计算社区经常使用它完成数据可视化的工作。

绘图过程中常用样式如下
>颜色： 蓝色 - `b` 绿色 - `g` 红色 - `r` 青色 - `c` 品红 - `m` 黄色 - `y` 黑色 - `k`（`b`代表蓝色，所以这里用黑色的最后一个字母） 白色 - `w` 
>线： 直线 - `-` 虚线 - `--` 点线 - `:` 点划线 - `-.` 常用点标记 点 - `.` 像素 - `,` 圆 - `o` 方形 - `s` 三角形 - `^`

- 本文主要是在 `jupyter notebook` 中绘图，首先需要 `import matplotlib`，通常的引入约定如下
```python
import matplotlib.pyplot as plt
%matplotlib inline # 加这行不需要再写plt.show()
``` 

- 画折线，红色
```python
plt.plot([1,2,4,8,16],'r')
```

- 画 `y=sin(x)`
```python
import numpy as np

x=np.linspace(0,2*np.pi) 
plt.plot(x,np.sin(x))
```

- 在一张图中绘制多个数据集
```python
x=np.linspace(0,2*np.pi)
plt.plot(x,np.sin(x),x,np.sin(2*x))
```

- 自定义图形的外观
代码展示了两种不同的曲线样式：`r-o` 和 `g--`。字母 `r` 和 `g` 代表线条的颜色，后面的符号代表线和点标记的类型。例如 `-o` 代表包含实心点标记的实线，`--` 代表虚线。
```python
x=np.linspace(0,2*np.pi)
plt.plot(x,np.sin(x),'r-o',
         x,np.cos(x),'g--')
```

- 使用子图在一个窗口绘制多张图
```python
x=np.linspace(0,2*np.pi)
plt.subplot(2,1,1) #行，列，活跃区
plt.plot(x,np.sin(x),'r') # 红色
plt.subplot(2,1,2)
plt.plot(x,np.cos(x),'g') # 绿色
```

- 绘制简单的散点图
```python
x=np.linspace(0,2*np.pi)
plt.plot(x,np.sin(x),'go') # 绿色圆点
```

- 彩色映射散点图
同前面一样我们用到了 `scatter()` 函数，但是这次我们传入了另外的两个参数，分别为所绘点的大小和颜色。通过这种方式使得图上点的大小和颜色根据数据的大小产生变化。
然后我们用 `colorbar()` 函数添加了一个颜色栏。
```python
x=np.random.rand(100)
y=np.random.rand(100)
size=np.random.rand(100)*50
colour=np.random.rand(100)
plt.scatter(x,y,size,colour)
plt.colorbar()
```

- 直方图
```python
x=np.random.randn(100)*10
plt.hist(x,50)
```

- 添加标题，坐标轴标记和图例
为了给图形添加图例，我们需要在 `plot()` 函数中添加命名参数 `label` 并赋予该参数相应的标签。然后调用 `legend()` 函数就会在我们的图形中添加图例。
接下来我们只需要调用函数 `title()`，`xlabel()` 和 `ylabel()` 就可以为图形添加标题和标签。
```python
x=np.linspace(0,2*np.pi)
plt.plot(x,np.sin(x),'r-x',label='Sin(x)')
plt.plot(x,np.cos(x),'g-^',label='Cos(x)')
plt.legend() #展示图例，必须调用这个上面的label才会显示
plt.xlabel('Reds') #给x轴添加标签
plt.ylabel('Amplitude') #给y轴添加标签
plt.title('Sin and Cos Waves') #添加图形标题
```