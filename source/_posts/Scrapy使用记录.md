---
title: Scrapy使用记录
date: 2018-03-22 21:25:44
tags: [python,Scrapy,爬虫] 
categories: [python]
---

### 安装

```bash
pip install Scrapy
```

### 创建项目

```bash
scrapy startproject tutorial
```

该命令将会创建包含下列内容的 `tutorial` 目录:

```
tutorial/
    scrapy.cfg
    tutorial/
        __init__.py
        items.py
        pipelines.py
        settings.py
        spiders/
            __init__.py
            ...

```

这些文件分别是:

- `scrapy.cfg`: 项目的配置文件
- `tutorial/`: 该项目的python模块。之后您将在此加入代码。
- `tutorial/items.py`: 项目中的item文件.
- `tutorial/pipelines.py`: 项目中的pipelines文件.
- `tutorial/settings.py`: 项目的设置文件.
- `tutorial/spiders/`: 放置spider代码的目录.


定义Item，Item用来保存爬去的数据，与python中的dict类似。

编写爬虫，保存在 `tutorial/spiders` 目录下的 `dmoz_spider.py` 文件中：

```python
import scrapy

class DmozSpider(scrapy.Spider):
    name = "dmoz"
    allowed_domains = ["dmoz.org"]
    start_urls = [
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Books/",
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/"
    ]

    def parse(self, response):
        filename = response.url.split("/")[-2]
        with open(filename, 'wb') as f:
            f.write(response.body)
```

### 运行

进入项目的根目录，执行下列命令启动spider:

```
scrapy crawl dmoz
```

其中，`dmoz` 为`tutorial/spiders` 目录下的 `dmoz_spider.py` 文件中为爬虫设置的`name`。就像 parse 方法指定的那样，有两个包含url所对应的内容的文件被创建了: Book , Resources 。

### 提取Item

#### Selectors选择器简介

从网页中提取数据有很多方法。Scrapy使用了一种基于 [XPath](http://www.w3.org/TR/xpath) 和 [CSS](http://www.w3.org/TR/selectors) 表达式机制: [Scrapy Selectors](http://scrapy-chs.readthedocs.io/zh_CN/0.24/topics/selectors.html#topics-selectors)。 关于selector和其他提取机制的信息请参考 [Selector文档](http://scrapy-chs.readthedocs.io/zh_CN/0.24/topics/selectors.html#topics-selectors) 。

这里给出XPath表达式的例子及对应的含义:

- `/html/head/title`: 选择HTML文档中 `<head>` 标签内的 `<title>` 元素
- `/html/head/title/text()`: 选择上面提到的 `<title>` 元素的文字
- `//td`: 选择所有的 `<td>` 元素
- `//div[@class="mine"]`: 选择所有具有 `class="mine"` 属性的 `div` 元素

上边仅仅是几个简单的XPath例子，XPath实际上要比这远远强大的多。 如果您想了解的更多，我们推荐 [这篇XPath教程](http://www.w3schools.com/XPath/default.asp) 。

#### 在Shell中尝试Selector选择器

为了介绍Selector的使用方法，接下来我们将要使用内置的 [Scrapy shell](http://scrapy-chs.readthedocs.io/zh_CN/0.24/topics/shell.html#topics-shell) 。Scrapy Shell需要您预装好IPython(一个扩展的Python终端)。

您需要进入项目的根目录，执行下列命令来启动shell:

```
scrapy shell "http://www.dmoz.org/Computers/Programming/Languages/Python/Books/"
```

### 保存爬取到的数据

最简单存储爬取的数据的方式是使用 [Feed exports](http://scrapy-chs.readthedocs.io/zh_CN/0.24/topics/feed-exports.html#topics-feed-exports):

```
scrapy crawl dmoz -o items.json
```

该命令将采用 [JSON](http://en.wikipedia.org/wiki/JSON) 格式对爬取的数据进行序列化，生成 `items.json` 文件。

### 其他

- 以为是机器人不允许爬取，更改 `settings.py`，将 `ROBOTSTXT_OBEY` 置为 `False`

- 获取的中文是Unicode，不方便阅读，可以使用BeautifulSoup，不仅可以提取网页结构中的文字，还可以显示为中文。
```python
@staticmethod
def get_plain_text(html):  # 不仅可以解析html，还可以将unicode编码输出为汉字
    return BeautifulSoup(html, "lxml").text
```

- 使用xpath定位兄弟结点的方法：`//div[@id='D']/preceding-sibling::div[1]`和`//div[@id='D']/following-sibling::div[1]`
参考：[Python selenium —— 父子、兄弟、相邻节点定位方式详解](http://blog.csdn.net/huilan_same/article/details/52541680)
`preceding-sibling`，其能够获取当前节点的所有同级哥哥节点，注意括号里的标号，**1 代表着离当前节点最近的一个哥哥节点，数字越大表示离当前节点越远**，当然，`xpath轴：preceding`也可以，但是使用起来比较复杂，它获取到的是该节点之前的所有非祖先节点。`following-sibling`，跟`preceding-sibling`类似，它的作用是获取当前节点的所有同级弟弟节点，同样，**1 代表离当前节点最近的一个弟弟节点，数字越大表示离当前节点越远**

- node()[not(self::div)]用法参考：https://stackoverflow.com/questions/4455684/xpath-get-only-node-content-without-other-elements
表示不要它自己的孩子div结点。例如如下代码，只要“   This is some text”，而不要h1和div标签内的内容，可以使用`/div/node()[not(self::h1|self::div)]`
```html
<div>
   This is some text
   <h1>This is a title</h1>
   <div>Some other content</div>
</div>
```

- 对于.py中有需要读写文件的部分，注意文件的路径！如果使用相对路径，程序会从执行`scrapy scrawl`所在目录的相对路径找。如果每次执行crawl命令所在的目录不同，那么就会报错`IOError: [Errno 2] No such file or directory: 't.txt'`。所以，最好使用绝对路径，或者每次执行的命令的目录都相同（例如每次都在根目录执行）。

### 参考
- [Scrapy 0.24.6 文档](http://scrapy-chs.readthedocs.io/zh_CN/0.24/index.html)
- [Python selenium —— 父子、兄弟、相邻节点定位方式详解](http://blog.csdn.net/huilan_same/article/details/52541680)
- https://stackoverflow.com/questions/4455684/xpath-get-only-node-content-without-other-elements