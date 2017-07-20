---
title: mysql使用记录
date: 2017-07-19 19:47:15
tags: [tools,sql]
categories: tools
top: 2
---

在mac中使用mysql的记录。

## 安装
使用homebrew安装mysql
```bash
brew install mysql
```

下载 `sequel pro`，使用 sequel-pro 来对数据库进行管理。以下介绍引用自 [segmentfalt: Mac 上的 MySQL 管理工具 -- Sequel Pro](https://segmentfault.com/a/1190000006255923)

> Sequel Pro 是 Mac 用户常用的 MySQL 管理软件，属于开源项目 CocoaMySQL 的一个分支。它支持包括索引在内的所有表管理功能，支持MySQL视图，可以同时使用多个窗口来操作多个数据库/表。完全可以媲美大家熟悉的 phpMyadmin。
>
> Sequel Pro 的部分特性如下：
>
> 1. 操作快速，简单。通过简单的几个参数设定即可连接本地或远程MySQL。
> 2. 支持多窗口操作。在不同的个窗口中，对多数据库实施操作。
> 3. SQL语句的语法彩色、加亮显示。
> 4. SQL语句的关键字、表名、字段名的自动完成。
> 5. 支持30多种不同的字符编码。
> 6. 快速导入/恢复、导出/备份SQL及CSV格式的数据。
> 7. 兼容MySQL3、4、5。
> 8. 支持在MAMP/XAMP架构上连接数据库，支持SSH连接模式；
> 9. 免费使用，当然，如果你觉得不错，可以 Donate 支持一下作者。

## 使用

### 命令行

启动 MySQL 服务，运行 `mysql.server`，在命令行输入：

```mysql
( ⌁ ) mysql.server start
```

登录 MySQL，运行:

```mysql
( ⌁ ) mysql -uroot
```

**Note**: 默认情况下，MySQL 用户 `root` 没有密码，这对本地开发没有关系，但如果你希望修改密码，你可以运行:

```mysql
$ mysqladmin -u root password 'new-password'
```

关闭 MySQL，运行：（注意运行位置）

```mysql
( ⌁ ) mysql.server stop                                                
Shutting down MySQL
.... SUCCESS!
```

你可以了解更多 `mysql.server` 的命令，运行：

```mysql
( ⌁ ) mysql.server --help                                                                                       
Usage: mysql.server  {start|stop|restart|reload|force-reload|status}  [ MySQL server options ]
```

### sequel pro

在运行（`mysql.server start`）后可以在图形界面使用mysql：

- 打开之前安装的sequel-pro，点击左下方`＋`新建一个 `FAVORITES`，在右侧进行配置。
- Name： 随便写
- Host：因为是在本地，所以填`127.0.0.1`
- Username：与自己的用户名相同，如果按照上面的写法，用户是`root`
- Password：如果设置了就写，没设置就空着
- Database：要连接的数据库名
- Port：端口号默认 3306，可以不写

设置好后点`connect`，就可以利用图形界面来操作了。

更多的操作方法可以参考 [segmentfalt: Mac 上的 MySQL 管理工具 -- Sequel Pro](https://segmentfault.com/a/1190000006255923)

## 常用 sql

使用sql语句来操作数据库，需要注意的是：

- SQL 对大小写不敏感
- 某些数据库系统要求在每条 SQL 命令的末端使用分号。分号是在数据库系统中分隔每条 SQL 语句的标准方法，这样就可以在对服务器的相同请求中执行一条以上的语句。

### 对数据库的操作

` show databases;` 查看现在有哪些数据库（注意分号）

```mysql
mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
| test               |
+--------------------+
5 rows in set (0.00 sec)
```

`create database 数据库名;` 新建一个数据库

```mysql
mysql> create database test;
Query OK, 1 row affected (0.00 sec)
```

`use 数据库名;` 选择要操作的 Mysql 数据库，使用该命令后所有 Mysql 命令都只针对该数据库

```mysql
mysql> use test;
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed
```

`drop database 数据库名;` 删除数据库

```mysql
mysql> drop database chou;

Query OK, 1 row affected (0.01 sec)
```

### 对表的操作

`show tables;` 显示指定数据库的所有表，使用该命令前需要使用 use 命令来选择要操作的数据库。

```mysql
mysql> show tables;
+----------------+
| Tables_in_test |
+----------------+
| test_table1    |
+----------------+
1 row in set (0.00 sec)
```

`show columns from 数据表` 显示数据表的属性，属性类型，主键信息 ，是否为 NULL，默认值等其他信息。

```mysql
mysql> show columns from t;
+-------------+------------------+------+-----+---------+----------------+
| Field       | Type             | Null | Key | Default | Extra          |
+-------------+------------------+------+-----+---------+----------------+
| id          | int(11) unsigned | NO   | PRI | NULL    | auto_increment |
| word        | varchar(30)      | YES  |     | NULL    |                |
| explanation | varchar(5000)    | YES  |     | NULL    |                |
| tag         | varchar(100)     | YES  |     | NULL    |                |
+-------------+------------------+------+-----+---------+----------------+
4 rows in set (0.00 sec)
```

`delete from 数据表` 删除表中所有内容，删除后可以查询表中还有多少条记录

```mysql
mysql> delete from test;
Query OK, 6 rows affected (0.00 sec)

mysql> select count(*) from test;
+----------+
| count(*) |
+----------+
|        0 |
+----------+
1 row in set (0.00 sec)
```

`insert into 表名 (列名1,列名2) values ('值1','值2');` 将数据插入表。

## Tips

### source

通过 Mysql Source 命令能够将 SQL 文件导入 Mysql 数据库中。

在日常工作中，可以通过脚本生成 sql 语句，存入 `.sql` 文件中，然后使用 `source filename.sql` 就可以执行这些 sql 语句。

### char、varchar和text

此部分引用 [MySQL之char、varchar和text的设计- billy鹏- 博客园](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwinoqKn2ZbVAhVN72MKHZqZDXMQFggnMAA&url=http%3A%2F%2Fwww.cnblogs.com%2Fbillyxp%2Fp%2F3548540.html&usg=AFQjCNG2t0wISJxMON3NftjccYHa-2p58Q)

> 1、char（n）和varchar（n）中括号中n代表字符的个数，并不代表字节个数，所以当使用了中文的时候(UTF8)意味着可以插入m个中文，但是实际会占用m*3个字节。
>
> 2、同时char和varchar最大的区别就在于char不管实际value都会占用n个字符的空间，而varchar只会占用实际字符应该占用的空间+1，并且实际空间+1<=n。
>
> 3、超过char和varchar的n设置后，字符串会被截断。
>
> 4、char的上限为255字节，varchar的上限65535字节，text的上限为65535。
>
> 5、char在存储的时候会截断尾部的空格，varchar和text不会。
>
> 6、varchar会使用1-3个字节来存储长度，text不会。

**总体来说：**

> 1、char，存定长，速度快，存在空间浪费的可能，会处理尾部空格，上限255。
>
> 2、varchar，存变长，速度慢，不存在空间浪费，不处理尾部空格，上限65535，但是有存储长度实际65532最大可用。
>
> 3、text，存变长大数据，速度慢，不存在空间浪费，不处理尾部空格，上限65535，会用额外空间存放数据长度，顾可以全部使用65535。

从官方文档中我们可以得知当varchar大于某些数值的时候，其会自动转换为text，大概规则如下：

- 大于varchar（255）变为 tinytext

- 大于varchar（500）变为 text
- 大于varchar（20000）变为 mediumtext

所以对于过大的内容使用varchar和text没有太多区别。

**所以我们认为当超过255的长度之后，使用varchar和text没有本质区别，只需要考虑一下两个类型的特性即可。（主要考虑text没有默认值的问题）**

从上面的简单测试看，基本上是没有什么区别的，但是个人推荐使用varchar（10000），毕竟这个还有截断，可以保证字段的最大值可控，如果使用text那么如果code有漏洞很有可能就写入数据库一个很大的内容，会造成风险。所以，本着short is better原则，还是使用varchar根据需求来限制最大上限最好。

## 问题及解决方法

### ERROR! The server quit without updating PID file

启动时报错：`ERROR! The server quit without updating PID file (/usr/local/var/mysql/bogon.pid).`

解决：首先查看有没有运行的mysql实例：`ps -ef | grep mysql`，如果有，就根据PID杀掉进程：`kill -9 [PID]`，这里将PID替换为查询到的PID（数字），再`mysql.server start`就成功了。

参考：[stackoverflow:MySql server startup error 'The server quit without updating PID file '](https://stackoverflow.com/questions/4963171/mysql-server-startup-error-the-server-quit-without-updating-pid-file)

## 参考

-  [segmentfalt: Mac 上的 MySQL 管理工具 -- Sequel Pro](https://segmentfault.com/a/1190000006255923)
- [MySql | Mac 开发配置手册 - GitBook](https://aaaaaashu.gitbooks.io/mac-dev-setup/content/MySql/index.html)
- [MySQL 教程| 菜鸟教程](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwj7hs38vJXVAhXEjFQKHafoCWIQFggmMAA&url=http%3A%2F%2Fwww.runoob.com%2Fmysql%2Fmysql-tutorial.html&usg=AFQjCNFPI6wVf6Vtm5rBmJ3RFYE6bQR8ZQ)
- [stackoverflow:MySql server startup error 'The server quit without updating PID file '](https://stackoverflow.com/questions/4963171/mysql-server-startup-error-the-server-quit-without-updating-pid-file)