---
title: python-docx使用记录
date: 2017-06-30 21:31:41
tags: [python,docx]
categories: [python]
---

因为要处理中文，所以在这里使用 python3（相对 python2 编码问题较少）。

安装 docx：使用 `pip3 install python-docx` 
如果安装失败可以尝试：`pip3 easy-install python-docx`

docx文档结构分为3层：
- `Document`对象表示整个文档
- `Document`包含了`Paragraph`对象的列表，`Paragraph`对象用来表示段落
- 一个`Paragraph`对象包含了`Run`对象的列表，`Run`：
word里不只有字符串，还有字号、颜色、字体等属性，都包含在`style`中。一个`Run`对象就是`style`相同的一段文本。
新建一个`Run`就有新的`style`。

### 基本操作

参考：http://python-docx.readthedocs.io/en/latest/

基本操作包括打开文档、在文档中写入内容、存储文档，简洁示例如下。
```python
from docx import Document
doc=Document() #不填文件名默认新建空白文档。填文件名（必须是已存在的doc文件）将打开这一文档进行操作
doc.add_heading('Hello') #添加标题
doc.add_paragraph('word') #添加段落
doc.save('test.docx') #保存，必须有1个参数
```

`python-docx`包含的对象集合如下

```python
doc.paragraphs    #段落集合
doc.tables        #表格集合
doc.sections      #节  集合
doc.styles        #样式集合
doc.inline_shapes #内置图形 等等...
```

http://python-docx.readthedocs.io/en/latest/ 中的示例如下：
> 
```python
from docx import Document
from docx.shared import Inches

document = Document()

document.add_heading('Document Title', 0)

p = document.add_paragraph('A plain paragraph having some ')
p.add_run('bold').bold = True
p.add_run(' and some ')
p.add_run('italic.').italic = True

document.add_heading('Heading, level 1', level=1)
document.add_paragraph('Intense quote', style='IntenseQuote')

document.add_paragraph(
    'first item in unordered list', style='ListBullet'
)
document.add_paragraph(
    'first item in ordered list', style='ListNumber'
)

document.add_picture('monty-truth.png', width=Inches(1.25))

table = document.add_table(rows=1, cols=3)
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Qty'
hdr_cells[1].text = 'Id'
hdr_cells[2].text = 'Desc'
for item in recordset:
    row_cells = table.add_row().cells
    row_cells[0].text = str(item.qty)
    row_cells[1].text = str(item.id)
    row_cells[2].text = item.desc

document.add_page_break()

document.save('demo.docx')
```

### 读写标题

背景：需要将某个文档中的标题拷贝到另一个文档中，但是标题太过分散，手动拷贝太费劲，所以考虑使用 docx 来处理。

打开 doc 文档，获取所有的 paragraphs（里面包含了Heading），查看这些 paragraphs 的 style（查看需要获取的标题是几级的）

```python
import docx
doc=docx.Document('filename.docx') #打开文档

ps=doc.paragraphs
for p in ps:
    print(p.style)
```

通过上面执行结果知道在这个文档（`filename.docx`）中，标题的 style 包括 `Heading 1`、`Heading 2`、`Heading 3`（其他文档的标题也许不是这些 style），我们通过 `p.style.name`来匹配这些标题，将标题及其 level 存到 re 中备用。

```python
re=[]
for p in ps:
    if p.style.name=='Heading 1':
        re.append((p.text,1))
    if p.style.name=='Heading 2':
        re.append((p.text,2))
    if p.style.name=='Heading 3':
        re.append((p.text,3))   
```

现在已经获取了标题内容以及标题的 level，将 re 列表“解压”：`titles,titledes=zip(*re)`，标题存在 titles 列表中，level 存在 titledes 列表中，接下来将标题写到新文档中

```python
newdoc=docx.Document()
for i in range(len(titles)):
    newdoc.add_heading(titles[i],level=titledes[i])
newdoc.save('newfile.docx')
```

### 获取表格内容

背景：需要获取某个文档中所有表格的第二列和第三列内容。

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

### 创建表格

 `Document.add_table` 的前两个参数设置表格行数和列数，第三个参数设定表格样式，也可以用 table 的 style 属性获取和设置样式。如果设置样式，可以直接用样式的英文名称，例如『Table Grid』；如果对样式进行了读取，那么会得到一个 Style对象。这个对象是可以跨文档使用的。除此之外，也可以使用 `Style.name` 方法得到它的名称。

下面创建一个 6 行 2 列的表格，可以通过 `table.cell(i,j).text` 来对表格进行填充。

```python
doc=docx.Document()
tabel=doc.add_table(rows=6,cols=2,style = 'Table Grid') #实线
tabel.cell(0,0).text='编号'
tabel.cell(1,0).text='位置'
```

上面创建的表格每一列等宽，可以设置表格的列宽使其更美观。

```python
from docx.shared import Inches
for t in doc.tables:
    for row in t.rows:
        row.cells[0].width=Inches(1)
        row.cells[1].width=Inches(5)
```

### 参考

- http://python-docx.readthedocs.io/en/latest/
- [Python读取word文档——python-docx](http://www.itwendao.com/article/detail/172784.html)
- [Python读写docx文件](http://yshblog.com/blog/40)
- [使用表格—— 使用Python读写Office文档之三](http://www.ctolib.com/topics-57923.html)