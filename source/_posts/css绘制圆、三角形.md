---
title: css绘制圆、三角形
date: 2016-06-22 16:29:53
tags: css
---

# border-radius与圆（弧）

> `border-radius` 用来设置边框圆角。当使用一个半径时确定一个圆形；当使用两个半径时确定一个椭圆，这个(椭)圆与边框的交集形成圆角效果。
![图片来自 https://developer.mozilla.org/zh-CN/docs/Web/CSS/border-radius](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE%E5%9B%BE%E7%89%87.png "")

## border-radius基础

> 基本语法：`border-radius ： none | <length>{1,4} [/ <length>{1,4} ]`

> `<length>`取值范围：由浮点数字和单位标识符组成的长度值。不可为负值。

> 简单说明：`border-radius` 是一种缩写方法。如果“/”前后的值都存在，那么“/”前面的值设置其水平半径，“/”后面值设置其垂直半径。如果没有“/”，则水平和垂直半径相等。另外其四个值是按照top-left、top-right、bottom-right、bottom-left的顺序来设置的其主要会有下面几种情形出现：
1. 只有一个值，那么 top-left、top-right、bottom-right、bottom-left 四个值相等。
2. 有两个值，那么 top-left 等于 bottom-right，并且取第一个值；top-right 等于 bottom-left，并且取第二个值。
3. 有三个值，其中第一个值是设置top-left;而第二个值是 top-right 和 bottom-left 并且他们会相等,第三个值是设置 bottom-right。
4. 有四个值，其中第一个值是设置 top-left 而第二个值是 top-right 第三个值 bottom-right 第四个值是设置 bottom-left。


举几个实例：
- `border-radius： 2em` 即
```css
border-top-left-radius:2em;
border-top-right-radius:2em;
border-bottom-right-radius:2em;
border-bottom-left-radius:2em;
```

- `border-radius: 2em 1em 4em / 0.5em 3em` 即
```css
border-top-left-radius: 2em 0.5em;
border-top-right-radius: 1em 3em;
border-bottom-right-radius: 4em 0.5em;
border-bottom-left-radius: 1em 3em;
```

- `border-radius: 10px 15px 20px 30px / 20px 30px 10px 15px` 即
```css
  border-top-left-radius: 10px 20px;
  border-top-right-radius: 15px 30px;
  border-bottom-right-radius: 20px 10px;
  border-bottom-left-radius: 30px 15px;
```

## border-radius画圆

未进行处理时，div的边框是一个矩形。为了演示效果，我们画一个黄色的100px × 100px 的矩形。
HTML相关代码如下：
```html
<div class="circle"></div>
```

CSS相关代码如下：
```css
.circle {
	width: 100px;
	height: 100px;
	background-color: yellow;
}
```

效果如图：
![矩形（100px × 100px）](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E7%9F%A9%E5%BD%A2.png "")

>### 圆

分析：对100px × 100px 的矩形来说，圆形即将每个角的水平半径和垂直半径都设置为50px，所以应在css文件中加入`border-radius: 50px`，即
```css
.circle {
	width: 100px;
	height: 100px;
	background-color: yellow;
	border-radius: 50px;
}
```

效果如图：
![圆](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E5%9C%86.png "")

>### 1/4圆

分析：分析：对100px × 100px 的矩形来说，1/4圆即将**某一个角**的水平半径和垂直半径都设置为100px，这里我们设置右下角，在css文件中加入`border-radius: 0 0 100px 0`，即
```css
.circle {
	width: 100px;
	height: 100px;
	background-color: yellow;
	border-radius: 0 0 100px 0;
}
```

效果如图：
![1/4圆](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/1-4%E5%9C%86.png "")

>### 其他圆（弧）

找到水平半径和垂直半径就可以，假设我们需要画地平线上刚升起的太阳。水平半径设为50px，垂直半径设为30px。
```css
.circle {
	width: 100px;
	height: 100px;
	background-color: yellow;
	border-radius: 50px 50px 0 0/30px 30px 0 0;
}
```

效果如图:
![面包？](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E9%9D%A2%E5%8C%85.png "")
可以看到这并不是我们要的效果，这里需要注意将高度修改一下。因为这里我们设置的垂直半径为30px，所以将高度设置为30px以达到需要的效果。（画半圆的效果同理）
```css
.circle {
	width: 100px;
	height: 30px;
	background-color: yellow;
	border-radius: 50px 50px 0 0/30px 30px 0 0;
}
```

效果如图：
![日出](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E5%88%9D%E5%8D%87%E7%9A%84%E5%A4%AA%E9%98%B3.png "")


# border与三角形

怎么用 `border` 绘制三角形呢？首先我们梳理一下基础知识，知道 `border` 其实是由四部分组成的，通过将某些部分设置为透明来绘制三角形。

## border基础

>### border

初始：首先我们设div的宽和高都为100px，为了演示效果，我们将border的宽度设的大一些，并给四边涂上不同的颜色。
HTML相关代码如下：
```html
<div class="div1"></div>
```

css相关代码如下：
```css
.div1 {
	width: 100px;
	height: 100px;
	border: 50px solid;
	border-color: #00BCD4 #FFEB3B #E91E63 #9E9E9E;
}
```

效果如下图。我们可以看到边框是由4个梯形组成的。
![带边框矩形](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E5%88%9D%E5%A7%8B.png "")

三角形怎么来？上图的梯形是由于div遮挡了三角形吗？如果我们设置div的宽和高都为0，能不能出现4个三角形呢？
```css
.div1 {
	width: 0;
	height: 0;
	border: 50px solid;
	border-color: #00BCD4 #FFEB3B #E91E63 #9E9E9E;
}
```

效果如下图。
![设置块宽和高为0](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E5%9D%97%E5%AE%BD%E9%AB%98%E4%B8%BA0.png "")

>### border-width

`border-width` 属性如果单独使用的话是不会起作用的。需要先使用 `border-style` 属性（取值：solid等）来设置边框。
`border-width` 简写属性为元素的所有边框设置宽度（默认值是medium），或者单独地为各边边框设置宽度。

>* `border-width:thin;`
所有 4 个边框都是细边框

>* `border-width:thin medium;`
上边框和下边框是细边框
右边框和左边框是中等边框

>* `border-width:10px medium thick;`
上边框是 10px
右边框和左边框是中等边框
下边框是粗边框

>* `border-width:thin medium thick 10px;`
上边框是细边框
右边框是中等边框
下边框是粗边框
左边框是 10px 宽的边框

举个例子:设`border-width: 50px 20px`，即
```css
.div1 {
	width: 0;
	height: 0;
	border-style: solid;
	border-width: 50px 20px;
	border-color: #00BCD4 #FFEB3B #E91E63 #9E9E9E;
```
效果如下图。
![ ](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E6%94%B9%E5%8F%98%E5%AE%BD%E5%BA%A6.png "")


## border画三角形

由上图可以看出，要想得到三角形，将其他的3个三角形设置为透明（transparent）的就ok了。
```css
.div1 {
	width: 0;
	height: 0;
	border: 50px solid;
	border-color: transparent transparent #E91E63 transparent;
}
```

效果如下图。
![三角形](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E4%B8%89%E8%A7%92%E5%BD%A2.png "")

可以看到上图是一个等腰直角三角形，如果需要锐角或钝角三角形，则调整border-width就可以。
```css
.div1 {
	width: 0;
	height: 0;
	border-style: solid;
	border-width: 50px 20px;
	border-color: transparent transparent #E91E63 transparent;
}
```

效果如下图。
![锐角三角形](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E9%94%90%E8%A7%92%E4%B8%89%E8%A7%92%E5%BD%A2.png "")

# 其他图形

* 月牙
最后来画个包大人的胎记吧。画两个圆，然后使用position调整。
相关代码和效果图如下。
```css
.circle1 {
	width: 100px;
	height: 100px;
	background-color: yellow;
	border-radius: 50px;
}

.circle2 {
	width: 100px;
	height: 100px;
	background-color: #fff;
	border-radius: 50px;
	position: relative;
	left: 25px;
	top: -110px;
}
```

![小月牙](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/%E5%B0%8F%E6%9C%88%E7%89%99.png "")

* Pac-Man
css相关代码和效果图如下。
```css
.div1 {
	width: 0;
	height: 0;
	border: 50px solid;
	border-radius: 50px;
	border-color: yellow transparent yellow yellow;
}
```

![Pac-Man](http://7q5c08.com1.z0.glb.clouddn.com/css/%E7%94%BB%E5%9B%BE/QQ%E6%88%AA%E5%9B%BE20160622154756.png "")

更多图形见[【转】纯CSS画的基本图形（矩形、圆形、三角形、多边形、爱心、八卦等）](http://www.cnblogs.com/jscode/archive/2012/10/19/2730905.html "")

**参考**

- [border-radius](https://developer.mozilla.org/zh-CN/docs/Web/CSS/border-radius "")
- [CSS3的圆角Border-radius](http://www.w3cplus.com/css3/border-radius "")
- [CSS border-width 属性](http://www.w3school.com.cn/cssref/pr_border-width.asp "")
- [用 CSS 实现三角形与平行四边形](http://jerryzou.com/posts/use-css-to-paint-triangle-and-parallelogram/ "")
