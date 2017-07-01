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

```python
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

`doc.tables`返回的是文档中的表格，`rows`，`columns`和 `cell` 对象在遍历表格的时候很有用。

Table 对象有两个属性 `rows` 和 `columns`，等同于 Row 的列表以及 Column 的列表。因此迭代、求长度等对list的操作同样适用于 Rows 和 Columns。

`cell` 也是表格中常用的对象，可以利用以下五种方法得到Cell对象：

- 使用 Table 对象的 `cell(row,col)` 方法。左上角的坐标为0,0
- 使用 Table 对象的 `row_cells(row_index)` 方法得到一个 list，它包含了某一行的按列排序的所有 Cell
- 得到一个 Row 对象后，使用 `Row.cells` 属性得到该 Row 的按列排序的所有  Cell
- 使用 Table 对象的 `column_cells(column_index)` 方法得到一个 list，它包含了某一列的按行排序的所有 Cell
- 得到一个 Column 对象后，使用 `Column.cells` 属性得到该 Column 的按行排序的所有 Cell

如果想遍历所有 Cell，可以先遍历所有行（`table.rows`），再遍历每一行所有的 Cell； 也可以先遍历所有列（`table.columns`），再遍历每一列所有的 Cell。

一个Cell对象最常用的属性是 `text`。设置这个属性可以设定单元格的内容，读取这个属性可以获取单元格的内容。

为了便于理解，举例如下

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
- [使用表格—— 使用Python读写Office文档之三](http://www.ctolib.com/topics-57923.html)