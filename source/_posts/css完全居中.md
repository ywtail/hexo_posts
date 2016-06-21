---
title: css完全居中
date: 2016-06-21 15:53:27
tags: css
---
列出两种方法备忘一下。
主要参考[如何只用CSS做到完全居中](http://blog.jobbole.com/46574/ "")，英文原文在[Absolute Centering in CSS](http://codepen.io/shshaw/full/gEiDt "")。
这是一篇很好的文章。

首先画一个矩形框来进行演示，为了截图效果，我们给body一个背景颜色。
HTML相关内容如下：
```html
<body>
	<div class="container">
	</div>
</body>
```
CSS相关内容如下：
```css
body {
	background-color: #999;
}

.container {
	background-color: #fff;
	width: 300px;
	height: 200px;
}
```

效果如图：
![初始效果](http://7q5c08.com1.z0.glb.clouddn.com/css/%E6%AD%A3%E5%B8%B81.png "")

## 水平居中
在css文件的`.container`中加上说明`margin: auto`。

```css
.container {
	background-color: #999;
	width: 300px;
	height: 200px;
	margin: auto;
}
```

效果如图：
![水平居中](http://7q5c08.com1.z0.glb.clouddn.com/css/%E6%B0%B4%E5%B9%B3%E5%B1%85%E4%B8%AD1.png "")

## 完全居中

* 方法一
设置`position`为`absolute`，具体如下。
```css
.container {
	background-color: #fff;
	width: 300px;
	height: 200px;
	position: absolute;
	left: 50%;
	top: 50%;
	transform: translate(-50%,-50%);
}
```

* 方法二
将`position`设置为`absolute`，将`top,bottom,left,right`都设置为0。
```css
.container {
	background-color: #fff;
	width: 300px;
	height: 200px;
	margin: auto;
	position: absolute;
	top: 0;
	bottom: 0;
	left: 0;
	right: 0;
}
```

效果如图：
![完全居中](http://7q5c08.com1.z0.glb.clouddn.com/css/%E5%AE%8C%E5%85%A8%E5%B1%85%E4%B8%AD1.png "")

**注意：**
在容器内完全居中将父元素的`position`设置为 `relative`。（经实测，父元素`position: absolute`也可以）


## 向左偏移

令`left:0; right:auto`。

```css
.container {
	background-color: #fff;
	width: 300px;
	height: 200px;
	margin: auto;
	position: absolute;
	top: 0;
	bottom: 0;
	left: 0;
	right: auto;
}
```

效果如图：
![向左偏移](http://7q5c08.com1.z0.glb.clouddn.com/css/%E5%90%91%E5%B7%A6%E5%81%8F%E7%A7%BB.png "")