---
title: python-docx使用记录
date: 2017-06-30 21:31:41
tags: [python,docx]
categories: [python]
top: 2
---

因为要处理中文，所以在这里使用 python3（相对 python2 编码问题较少）。

安装 docx：使用 `pip3 install python-docx` 

`python-docx`包含的对象集合如下

```
doc.paragraphs    #段落集合
doc.tables        #表格集合
doc.sections      #节  集合
doc.styles        #样式集合
doc.inline_shapes #内置图形 等等...
```

### 获取表格内容

打开doc文档

```python
import docx
doc=docx.Document('filename.docx') #打开文档
```

`doc.tables`返回的是文档中的表格，为了便于理解，举例如下

```python
for table in doc.tables: #列举文档中的表格
    for row in table.rows: #表格中的每一行
        t1=row.cells[1].text #每一行中第2列（从0开始计数）的内容
        t2=row.cells[2].text #每一行中第3列的内容
```

获取表格中的数据后用 DataFrame 存，最后保存为csv文件。如果有中文乱码问题，最后加上`encoding='gb2312'`

```python
df.to_csv('filename.csv',index=False,encoding='gb2312')
```



### 参考

- [Python读取word文档——python-docx](http://www.itwendao.com/article/detail/172784.html)
- [Python读写docx文件](http://yshblog.com/blog/40)