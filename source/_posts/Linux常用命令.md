---
title: Linux常用命令
date: 2019-04-09 19:54:00
tags: [Linux,shell]
categories: Linux
---

## 文件、目录

- 绝对路径：『一定由根目录 / 写起』；相对路径：『不是由 / 写起』
- 比较特殊的目录
```
.         代表此层目录
..        代表上一层目录
-         代表前一个工作目录
~         代表『目前使用者身份』所在的家目录
~account  代表 account 这个使用者的家目录(account是个帐号名称)
```
- 常见的处理目录的命令
    - cd：Change Directory 变换目录
    - pwd：Print Working Directory 显示目前的目录
    - mkdir：make directory 创建一个新的目录
    - rmdir：删除一个空的目录。仅能删除空目录，要删除非空目录需使用`rm -r`命令
- 使用者能使用的命令是依据 PATH 变量所规定的目录去搜寻的；
- 不同的身份(root 与一般用户)系统默认的 PATH 并不相同。差异较大的地方在於 /sbin, /usr/sbin ；
- ls 可以检视文件的属性，尤其 -d, -a, -l 等选项特别重要！
- 文件的复制、删除、移动可以分别使用：cp, rm , mv等命令来操作；
- 检查文件的内容(读档)可使用的命令包括有：cat, tac, nl, more, less, head, tail, od 等
- cat -n 与 nl 均可显示行号，但默认的情况下，空白行会不会编号并不相同；
- touch 的目的在修改文件的时间参数，但亦可用来创建空文件；
- 一个文件记录的时间参数有三种，分别是 access time(atime), status time (ctime), modification time(mtime)，ls 默认显示的是 mtime。
- 新建文件/目录时，新文件的默认权限使用 umask 来规范。默认目录完全权限为drwxrwxrwx， 文件则为-rw-rw-rw-。
- 观察文件的类型可以使用 file 命令来观察；
- 搜寻命令的完整档名可用 which 或 type ，这两个命令都是透过 PATH 变量来搜寻档名；
- 搜寻文件的完整档名可以使用 whereis 或 locate 到数据库文件去搜寻，而不实际搜寻文件系统；
- 利用 find 可以加入许多选项来直接查询文件系统，以获得自己想要知道的档名。

### chmod 文件权限
- `chmod 选项 文件名` 变更文件或目录权限
  - r 读取权限，4
  - w 写入， 2
  - x 执行，1
- `chmod 750 filename/*.sh` 批量修改为可执行文件

### cd
> `cd [相对路径或绝对路径]` 最重要的就是目录的绝对路径与相对路径，还有一些特殊目录的符号

- `cd -` 返回上一个目录（非上一层），上一层为`cd ..`
- `cd ~` 或 `cd` 回到当前账号home目录，亦即是 /root 这个目录
- `cd ~vbird` 到 vbird 这个使用者的家目录，亦即 /home/vbird

### pwd
>`pwd [-P]`：
`-P`  ：显示出确实的路径，而非使用链接 (link) 路径。

- `pwd` 显示出目前的工作目录
- `pwd -P` 显示出实际的工作目录，而非链接路径，P要大写

### mkdir

> `mkdir [-mp] 目录名称`:
> `-m` ：配置文件的权限。直接配置，不需要看默认权限 (umask)
> `-p` ：帮助你直接将所需要的目录(包含上一级目录)递回创建起来！

- `mkdir -m 711 test2` 创建权限为rwx--x--x的目录
- `mkdir -p` 建立嵌套目录。-p, --parents  可以是一个路径名称。此时若路径中的某些目录尚不存在,加上此选项后,系统将自动建立好那些尚不存在的目录,即一次可以建立多级目录

### rmdir
> `rmdir [-p] 目录名称`：`-p` ：连同上一级『空的』目录也一起删除
> 注意：只能删除空目录

假设有目录结构`test1/test2/test3/test4`，都是空目录
- `rmdir test1/test2/test3/test4` 只会删除目录`test4`，现目录结构变为`test1/test2/test3`
- `rmdir -p test1/test2/test3/test4` 效果等同于`rm -r test1`。`rmdir -p test1/test2/test3` 会报错：目录非空

虽然使用 rmdir 比较不危险，但是局限大，只能删空目录，日常大多用`rm -r`

### ls
> `ls [-aAdfFhilnrRSt] 目录名称`
> -a  ：全部的文件，连同隐藏档( 开头为 . 的文件) 一起列出来(常用)
> -A  ：全部的文件，连同隐藏档，但不包括 . 与 .. 这两个目录
> -d  ：仅列出目录本身，而不是列出目录内的文件数据(常用)
> -f  ：直接列出结果，而不进行排序 (ls 默认会以档名排序！)
> -F  ：根据文件、目录等资讯，给予附加数据结构，例如：`*`:代表可运行档； `/`:代表目录； `=`:代表 socket 文件； `|`:代表 FIFO 文件；
> -h  ：将文件容量以人类较易读的方式(例如 GB, KB 等等)列出来；
> -i  ：列出 inode 号码，inode 的意义下一章将会介绍；
> -l  ：长数据串列出，包含文件的属性与权限等等数据；(常用)
> -n  ：列出 UID 与 GID 而非使用者与群组的名称 (UID与GID会在帐号管理提到！)
> -r  ：将排序结果反向输出，例如：原本档名由小到大，反向则为由大到小；
> -R  ：连同子目录内容一起列出来，等於该目录下的所有文件都会显示出来；
> -S  ：以文件容量大小排序，而不是用档名排序；
> -t  ：依时间排序，而不是用档名。

- `ls -t` 以文件修改时间排序，修改时间越接近now排序越靠前，即修改顺序的逆序
- `ls -r`, –reverse 依相反次序排列，可以理解为逆序。`ls -tr` 修改时间的顺序

## 命令及文件查找

- 命令文档名的查找：`which`或 `type`。这两个命令都是透过 `$PATH` 变量来搜寻档名；
- 文件文档名的查找：`whereis`, `locate`, `find`。`whereis` 或 `locate` 到数据库文件去搜寻，而不实际搜寻文件系统；利用 `find` 可以加入许多选项来直接查询文件系统，以获得自己想要知道的档名。

### which 查找命令文档

> which [-a] command
> -a ：将所有由 `$PATH`目录中可以找到的命令均列出，而不止第一个被找到的命令名称
> 这个命令是根据`$PATH`这个环境变量所规范的路径，去搜寻命令的文档名。which 后面接的是要搜索的命令，若加上 -a 选项，则可以列出所有的可以找到的同名文档

在终端机模式当中，连续输入两次`tab`按键就能够知道使用者有多少命令可以下达。 这些命令的完整文档路径可以透过 `which` 或 `type` 来搜索

- `which cd` 常用命令`cd` 无法通过`which`找到，因为`cd`是bash内建的命令，而`which`默认是找`$PATH`内所规范的目录。`cd`需要通过`type cd`来查找

### whereis 查找特定文件
> whereis [-bmsu] 文件或目录名
> -b    :只找 binary 格式的文件
> -m    :只找在说明档 manual 路径下的文件
> -s    :只找 source 来源文件
> -u    :搜寻不在上述三个项目当中的其他特殊文件

通常先使用 `whereis` 或者是 `locate` 来查找文件，如果真的找不到了，才以 `find` 来查找(`find` 是直接搜寻硬盘)。因为 `whereis` 与 `locate` 是利用数据库来搜寻数据，所以相当的快速，而且并没有实际的搜寻硬盘， 比较省时间

为什么`whereis` 比 `find` 快？
这是因为 Linux 系统会将系统内的所有文件都记录在一个数据库文件里面， 而当使用 `whereis` 或者是底下要说的 `locate` 时，都会以此数据库文件的内容为准， 因此，有的时后你还会发现使用这两个运行档时，会找到已经被杀掉的文件！ 而且也找不到最新的刚刚创建的文件呢！这就是因为这两个命令是由数据库当中的结果去搜寻文件的所在啊

### locate

> locate [-ir] keyword
> -i  ：忽略大小写的差异；
> -r  ：后面可接正规表示法的显示方式

`locate passwd` ，那么在完整档名 (包含路径名称) 当中，只要有 `passwd` 在其中， 就会被显示出来

**限制**：
使用 locate 来寻找数据的时候特别的快， 这是因为 locate 寻找的数据是由『已创建的数据库 /var/lib/mlocate/』 里面的数据所搜寻到的，所以不用直接在去硬盘当中存取数据

就是因为是经由数据库来搜寻的，而数据库的创建默认是在每天运行一次 (每个 distribution 都不同，CentOS 5.x 是每天升级数据库一次！)，所以当你新创建起来的文件， 却还在数据库升级之前搜寻该文件，那么 `locate` 会告诉你『找不到！』呵呵！因为必须要升级数据库呀！

那能否手动升级数据库哪？当然可以啊！升级 locate 数据库的方法非常简单，直接输入『 updatedb 』就可以了！ updatedb 命令会去读取 /etc/updatedb.conf 这个配置档的配置，然后再去硬盘里面进行搜寻档名的动作， 最后就升级整个数据库文件罗！因为 updatedb 会去搜寻硬盘，所以当你运行 updatedb 时，可能会等待数分钟的时间喔！

- `updatedb`：根据 /etc/updatedb.conf 的配置去搜寻系统硬盘内的档名，并升级 /var/lib/mlocate 内的数据库文件；
- `locate`：依据 /var/lib/mlocate 内的数据库记载，找出使用者输入的关键字档名。

### find

> find [PATH] [option] [action]
> 1. 与时间有关的选项：共有 -atime, -ctime 与 -mtime ，以 -mtime 说明
>    -mtime  n ：n 为数字，意义为在 n 天之前的『一天之内』被更动过内容的文件；
>    -mtime +n ：列出在 n 天之前(不含 n 天本身)被更动过内容的文件档名；
>    -mtime -n ：列出在 n 天之内(含 n 天本身)被更动过内容的文件档名。
>    -newer file ：file 为一个存在的文件，列出比 file 还要新的文件档名
> 2. 与使用者或群组名称有关的参数：
>    -uid n ：n 为数字，这个数字是使用者的帐号 ID，亦即 UID ，这个 UID 是记录在
>             /etc/passwd 里面与帐号名称对应的数字。这方面我们会在第四篇介绍。
>    -gid n ：n 为数字，这个数字是群组名称的 ID，亦即 GID，这个 GID 记录在
>             /etc/group，相关的介绍我们会第四篇说明～
>    -user name ：name 为使用者帐号名称喔！例如 dmtsai 
>    -group name：name 为群组名称喔，例如 users ；
>    -nouser    ：寻找文件的拥有者不存在 /etc/passwd 的人！
>    -nogroup   ：寻找文件的拥有群组不存在於 /etc/group 的文件！
>                 当你自行安装软件时，很可能该软件的属性当中并没有文件拥有者，
>                 这是可能的！在这个时候，就可以使用 -nouser 与 -nogroup 搜寻。
> 3. 与文件权限及名称有关的参数：
>    -name filename：搜寻文件名称为 filename 的文件；
>    -size [+-]SIZE：搜寻比 SIZE 还要大(+)或小(-)的文件。这个 SIZE 的规格有：
>                    c: 代表 byte， k: 代表 1024bytes。所以，要找比 50KB
>                    还要大的文件，就是『 -size +50k 』
>    -type TYPE    ：搜寻文件的类型为 TYPE 的，类型主要有：一般正规文件 (f),
>                    装置文件 (b, c), 目录 (d), 连结档 (l), socket (s), 
>                    及 FIFO (p) 等属性。
>    -perm mode  ：搜寻文件权限『刚好等於』 mode 的文件，这个 mode 为类似 chmod
>                  的属性值，举例来说， -rwsr-xr-x 的属性为 4755 ！
>    -perm -mode ：搜寻文件权限『必须要全部囊括 mode 的权限』的文件，举例来说，
>                  我们要搜寻 -rwxr--r-- ，亦即 0744 的文件，使用 -perm -0744，
>                  当一个文件的权限为 -rwsr-xr-x ，亦即 4755 时，也会被列出来，
>                  因为 -rwsr-xr-x 的属性已经囊括了 -rwxr--r-- 的属性了。
>    -perm +mode ：搜寻文件权限『包含任一 mode 的权限』的文件，举例来说，我们搜寻
>                  -rwxr-xr-x ，亦即 -perm +755 时，但一个文件属性为 -rw-------
>                  也会被列出来，因为他有 -rw.... 的属性存在！
> 4. 额外可进行的动作：
>    -exec command ：command 为其他命令，-exec 后面可再接额外的命令来处理搜寻到
>                    的结果。
>    -print        ：将结果列印到萤幕上，这个动作是默认动作！

- `find -name abc.sh` 按文件名查找文件abc.sh，会在硬盘查找，尽量少用`find`，先使用`whereis`和`locate`


## tar 打包压缩及解压

- `tar -zcvf one.tar.gz 1/` 将文件夹1压缩打包为one.tar.gz

- `tar -cvf one.tar.gz 1/` 不压缩打包，更快，但是文件大小不减小

- `tar -xzvf` 解压。未压缩可使用`tar -xvf` 解压


## vim
### 删除文件中所有内容
方法1:    按`ggdG`
方法2:    `:%d`

### 区块选择

| 区块选择的按键 |                                        |
| -------------- | -------------------------------------- |
| v              | 字符选择，会将光标经过的地方反白选择！ |
| V              | 行选择，会将光标经过的行反白选择！     |
| [Ctrl]+v       | 区块选择，可以用长方形的方式选择资料   |
| y              | 将反白的地方复制起来，p粘贴            |
| d              | 将反白的地方删除掉                     |


## crontab

> linux下用来周期性执行任务或等待处理某些事情的一个守护进程。用户所建立的crontab文件中，每一行都代表一项任务，每个的每个字段代表一项设置：共6个字段，前5个是时间设定，第6个是要执行的命令，格式如下：
> `minute   hour   day   month   week   command `
> minute： 表示分钟，可以是从0到59之间的任何整数。
> hour：表示小时，可以是从0到23之间的任何整数。
> day：表示日期，可以是从1到31之间的任何整数。
> month：表示月份，可以是从1到12之间的任何整数。
> week：表示星期几，可以是从0到7之间的任何整数，这里的0或7代表星期日。
> command：要执行的命令，可以是系统命令，也可以是自己编写的脚本文件。

- `crontab -l` 列出
- `crontab FILENAME` 将文件FILENAME新增为定时任务，会覆盖之前crontab任务，慎用
- `crontab -e` 修改

## alias

- `alias 新命令='原命令 -选项/参数'` 设置命令别名
- `alias -p` 查看系统已设置的别名
- 令别名永久生效：(如果没有.bashrc文件就创建)
  `vim ~/.bashrc`
  在文件最后加上别名设置，如：
  `alias 90='ssh admin@111.111.111.90'`
  完成后：
  `source ~/.bashrc`
- 如果再某台机器上设置的alias干扰正常用法，可以使用`\rm filename`或者使用`/bin/rm filename`来删除，使用这两种方式使用的就不是alias中设置的rm了。

## shell

### 统计和排序

- `wc -l` 统计行数
  ```shell
  bash-3.2$ wc -l filename.txt
         3 filename.txt
  bash-3.2$ cat filename.txt | wc -l
         3
  ```
- `uniq` 的一个特性：检查重复行的时候只会检查相邻的行。所以去重前需要先保证有序：`sort | uniq`。或者直接使用`sort -u`
- `sort` 默认以字符升序排列

### awk
- `awk -F` 指定输入文本分隔符拆分。`awk -F"\t" '{print $5}'` 表示以"\t"为分隔符切分文本，并打印第5列（从1开始计数）。`{print $0}` 表示打印整行。
- `awk -v var=value` 赋值一个用户定义变量，将外部变量传递给awk

### grep
- `grep -n` 显示行号
- `grep -A 3 -B 3` 显示前后3行

### 数组 array

#### shell脚本中的list(array)
[does linux shell support list data structure?](https://stackoverflow.com/questions/12316167/does-linux-shell-support-list-data-structure) 

```bash
array=("item 1" "item 2" "item 3") 

# 这么创建list，中间能够包含空格。如果中间不包含空格，可以不加引号，如：
array=(red orange white "light gray")

# 创建list及访问元素（推荐），元素本身可以包含空格
array=("item 1" "item 2" "item 3")
for i in "${array[@]}"; do   # The quotes are necessary here
    echo "$i"
done

#item 1
#item 2
#item 3

# 如果${array[@]}不加引号，则元素被空格分开
for i in ${array[@]}; do
    echo $i
done
#item
#1
#item
#2
#item
#3

# 使用如下方式
list='"item 1" "item 2" "item 3"'
for i in $list; do
    echo $i
done
#"item
#1"
#"item
#2"
#"item
#3"

# 将$list看做一个字符串，并非list（array）
for i in "$list"; do
    echo $i
done
#"item 1" "item 2" "item 3"
```

#### 使用array存储命令

```
cmd_str_list=("sh run.sh a"
"sh run.sh b"
)

for ((i = 0; i < ${#cmd_str_list[@]}; i++))
do
    cmd_str="${cmd_str_list[$i]}"
    echo ${cmd_str}
    eval ${cmd_str}
done
```
参考: [Bash array with spaces in elements](https://stackoverflow.com/questions/9084257/bash-array-with-spaces-in-elements)

#### ls结果分配给数组
- `array=($(ls file))`在Linux Bash中，将ls结果分配给数组。可用下标访问

#### 数组拷贝

参考：[How to copy an array in Bash?](https://stackoverflow.com/questions/19417015/how-to-copy-an-array-in-bash)

```bash
a=(foo bar "foo 1" "bar two")  #create an array
b=("${a[@]}")                  #copy the array in another one 

for value in "${b[@]}" ; do    #print the new array 
echo "$value" 
done   
```

### echo

`echo -e` 打印特殊字符，如tab
```bash
echo "a\tb"
#a\tb

echo -e "a\tb"
#a   b
```

### expr数学计算

`1+2`字符串，`1 + 2`数学计算（注意空格）

```bash
bash-3.2 d$ expr 1+2
1+2
bash-3.2$ expr 1 + 2
3
```

### sed
- `sed '1d'` 删除第一行，更多介绍：<https://www.cnblogs.com/ggjucheng/archive/2013/01/13/2856901.html>
- `sed 's/要替换的字符串/新的字符串/g'`

### 变量替换
除了使用`sed 's/要替换的字符串/新的字符串/g'`，还可以使用：
`${变量/旧字符串/新字符串}` 若变量内容符合“旧字符串”，则第一个旧字符串会被新字符串替换。
`${变量//旧字符串/新字符串}` 若变量内容符合“旧字符串”，则全部的旧字符串会被新字符串替换。
也可以在旧字符前加\，结果一致。如果需要替换转义符，需要加\\
例如：
```bash
bash-3.2$ header="theme_res_id\tdid\tc3_dist"
bash-3.2$ echo ${header//id/,} # 将id替换为,
theme_res_,\td,\tc3_dist
bash-3.2$ echo ${header//\t/,} # 将t替换为,
,heme_res_id\,did\,c3_dis,
bash-3.2$ echo ${header//\\t/,} # 将转义符\t替换为,
theme_res_id,did,c3_dist
```
参考： Bash变量的删除、取代与替换https://andyyoung01.github.io/2017/02/21/Bash%E5%8F%98%E9%87%8F%E7%9A%84%E5%88%A0%E9%99%A4%E3%80%81%E5%8F%96%E4%BB%A3%E4%B8%8E%E6%9B%BF%E6%8D%A2/

### string split
- shell脚本中string split功能：<https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash?page=1&tab=votes#tab-top> 。使用cut比较简洁：

```bash
$ echo "bla@some.com;john@home.com" | cut -d ";" -f 1
bla@some.com
$ echo "bla@some.com;john@home.com" | cut -d ";" -f 2
john@home.com
```

### 函数返回值
中文参考：[shell 函数返回值接收问题](https://blog.csdn.net/mdx20072419/article/details/9381339)
[Linux Shell函数返回值](https://blog.csdn.net/ithomer/article/details/7954577)

函数可以有返回值，return，但是需要返回值是数字，而不能是字符串，字符串会提示：`/data0/home/rec/chengsujun/test_ly/search_index_builder/DataHope/scripts/../utils.sh: line 33: return: index_data/sku_precise/2019-01-16: numeric argument required`

虽然有这个提示，但是执行似乎成功了

[How to return a string value from a Bash function](https://stackoverflow.com/questions/3236871/how-to-return-a-string-value-from-a-bash-function)

点赞最高：没有什么方法

```bash
function func_mk_hadoop_dir(){
    local date=${1}
    local index_type=${2}
    local HADOOP_DATA_DIR=${HADOOP_DIR}/${index_type}/${date}
    return "${HADOOP_DATA_DIR}"
}

#调用加``符号
hadoop_data_dir=`func_mk_hadoop_dir ${date} ${index_type}`
#这么执行报错：return: index_data/sku_precise/2019-01-16: numeric argument required
```

使用echo方法：
```
function func_mk_hadoop_dir(){
    local date=${1}
    local index_type=${2}
    local HADOOP_DATA_DIR=${HADOOP_DIR}/${index_type}/${date}
    echo "${HADOOP_DATA_DIR}"
}
```

### 报错：“integer expression expected”（需要的是整数)

将字符串比较写为了整数比较的格式，就会出现这个报错。这是在比较两个日期时报的错，解决方案： <https://unix.stackexchange.com/questions/84381/how-to-compare-two-dates-in-a-shell>

```bash
todate=2013-07-18
cond=2013-07-15

if [ $todate -ge $cond ];
then
   echo "break"
fi    

# 将变量用引号引起来，-ge修改为>，改写为：（if中[[]]也可替换为[]）
date_a=2013-07-18
date_b=2013-07-15
if [[ "$date_a" > "$date_b" ]] ;
then
  echo "break"
fi

# 备注：也有其他方案例如先将日期转为整数再比较：
todate=$(date -d 2013-07-18 +%s)
cond=$(date -d 2014-08-19 +%s)

# 或者
todate=$(date -d 2013-07-18 +"%Y%m%d")  # = 20130718
cond=$(date -d 2013-07-15 +"%Y%m%d")    # = 20130715

# 或者删除-
$ echo "2013-07-15" | tr -d "-"
20130715

```

### 报错：bad substitution
`for i in ${awk -F ":" '{print $1}' /etc/passwd | grep stu}`报错`${awk -F ":" '{print $1}' /etc/passwd | grep stu}: bad substitution`

变量引用是`$()`，而不是`${}`，所以只需要把这行代码改为：`for i in $(awk -F ":" '{print $1}' /etc/passwd | grep stu)`

### 输出到文件需要加echo

例如：
```bash
a=5
$a>t.txt
```
会报错：`-bash: 5: command not found`，文件t.txt中没有内容

正确方式是：

```bash
a=5
echo $a>t.txt

```

### shell 脚本中set命令

#### set -e
`-e` 选项作用范围 :<https://blog.csdn.net/fc34235/article/details/76598448>

> set -e就是当命令以非零状态退出时，则退出shell。主要作用是，当脚本执行出现意料之外的情况时，立即退出，避免错误被忽略，导致最终结果不正确。
>
> set -e 命令用法总结如下：
>
> 1.当命令的返回值为非零状态时，则立即退出脚本的执行。
>
> 2.作用范围只限于脚本执行的当前进行，不作用于其创建的子进程。
>
> 3.另外，当想根据命令执行的返回值，输出对应的log时，最好不要采用set -e选项，而是通过配合exit 命令来达到输出log并退出执行的目的。

#### set -u
参考：(<http://man.linuxde.net/set>)

`set -u`：当执行时使用到未定义过的变量，则显示错误信息。 

不要在公共机器上`set -u`，会导致自动补全功能失效，并报错：

```bash
cd sear-bash: !ref: unbound variable
-bash: !ref: unbound variable
-bash: words[i]: unbound variable
```

关闭： `set +u`

`set -x`同理，关闭:`set +x`


## 查看物理cpu与逻辑cpu概述 
https://blog.csdn.net/BeautifulGrils/article/details/79799634

1. 物理cpu数：主板上实际插入的cpu数量，可以数不重复的 physical id 有几个（physical id）
2. cpu核数：单块CPU上面能处理数据的芯片组的数量，如双核、四核等 （cpu cores）
3. 逻辑cpu数：一般情况下，逻辑cpu=物理CPU个数×每颗核数，如果不相等的话，则表示服务器的CPU支持超线程技术（HT：简单来说，它可使处理器中的1 颗内核如2 颗内核那样在操作系统中发挥作用。这样一来，操作系统可使用的执行资源扩大了一倍，大幅提高了系统的整体性能，此时逻辑cpu=物理CPU个数×每颗核数x2）
备注一下：Linux下top查看的CPU也是逻辑CPU个数
```bash
#查看物理cpu个数：
cat /proc/cpuinfo |grep "physical id"| sort -u | wc -l
2
#查看逻辑cpu个数：（核数？=逻辑cpu个数）
cat /proc/cpuinfo |grep processor| wc -l
32
#查看cpu核数：（单个cpu的核数?所有的核数需要*物理cpu个数，由于超线程技术，因此再*2）
cat /proc/cpuinfo |grep cores|uniq
8
#查看CPU型号：cpu型号是 E5-2640
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c
32  Intel(R) Xeon(R) CPU E5-2640 v2 @ 2.00GHz
```

http://www.uuboku.com/137.html

## 查看内存

可使用`free`，默认单位是kB，free -m单位是M，free -g单位是G）
```bash
free -m
              total        used        free      shared  buff/cache   available
Mem:          31963       11688        1143         368       19131       19476
Swap:         16383        1419       14964
#总的物理内存为 32G，虚拟内存为16G（由于进制的关系，上面显示的是 31963  和 16383）。

cat /proc/meminfo
MemTotal:       32730616 kB
MemFree:         1168892 kB
MemAvailable:   19942200 kB
Buffers:               0 kB
Cached:         18844272 kB
#看第一行 32730616 kB，总的物理内存为32G
```

## 其他
- `ssh` 默认从`~/.ssh/`中匹配，所以，如果这个目录下.ssh不可用（由于文件权限等原因），在个人目录中新生成`.ssh`文件，在执行时应该使用`-i`显示指定`.ssh/id_rsa`文件地址
- `hostname -i` 获取本机ip
- `select_dt=$(date -d ''${input_dt}' day' +%Y-%m-%d)` 注意这里的参数${input_dt}必须用单引号引起来，否则会报错。shell：引号里面用需要加引号
- `if [ $# -ge 2 ] ; then` 如果参数的数量>2，$# 表示设置的参数个数
- `${where_condition:=1=1}`
  参考：http://www.cnblogs.com/fhefh/archive/2011/04/22/2024750.html
  : ${VAR:=”some default”}
  这些代码开始的冒号是一个正确执行非活动任务的shell命令。在这个句法中，它仅仅扩展了行中紧随其后的所有参数。本例中，只要是要在花括号内扩展参数值。
  本行ongoing冒号中的参数是最有趣的部分；它是用花括号起来的一些逻辑的参数扩展。:=句法表示VAR变量将会和“some defalut”字符串进行比较。
  在这个表达式中，如果变量VAR还没有被设置，那么“:=”之后表达式的值将被赋给它，这个值可能是一个数字，一个字符串，或者是另外一个变量。
- `sudo netstat -anp | grep TIME_WAIT | grep :22` 查看22端口
- `kill -9` 进程即刻结束
- 取出字符串中变量 `${!c}`

参考：
- [鸟哥的Linux私房菜](http://cn.linux.vbird.org/)