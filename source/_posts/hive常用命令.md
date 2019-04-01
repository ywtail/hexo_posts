---
title: hive常用命令
date: 2019-03-31 11:42:34
tags: hive
categories: hive
---



## 增

### 建表

- 重要的数据建表最好为externel ，在drop表时，不会drop对应hadoop文件，在重新create table后能够快速恢复数据。
- 可以显示指定format为orc，注意最后3行顺序
- location可以不写，但是建完后需要检测存储路径是否和预期一致

```sql
create [external]
    table if not exists app.test
    (
        sku string,
        csku string
    )
    partitioned by
    (
        dt string
    )
    stored as orc
    location 'xxxx/app.db/test'
    tblproperties ('orc.compress'='SNAPPY');
```

### 增加字段(列)

- 添加字段(在所有存在的列后面，但是在分区列之前添加一个字段)
  - 不添加说明：
    `alter table 表名 add columns (字段名 字段数据类型, 字段名 字段数据类型)`
  - 添加说明：
    `alter table 表名 add columns (字段名 字段数据类型 comment '字段说明', 字段名 字段数据类型 comment '字段说明')`

### 增加表记录

- 添加一条记录：`insert into 表名(字段名[，字段名]) values('值'[,'值']);`
- 或使用select
  ```sql
  insert into 表名(字段名[，字段名]) 
  select ‘...’
  ```
  使用insert into向表中追加数据，可能追加的数据与表中已有的数据相同，不会覆盖，因此会出现相同的两条记录。
- 添加记录（同记录覆盖）
  使用insert overwrite
  注意 `insert overwrite table 表名 select…..`
  table不能省略。
  会将之前的记录全部删除，即时与追加的记录不同

### 增加分区

```sql
ALTER TABLE table_name ADD PARTITION (partCol = 'value1') location 'loc1'; 

-- 一次添加一个分区
ALTER TABLE table_name ADD IF NOT EXISTS PARTITION (dt='20130101') LOCATION '/user/hadoop/warehouse/table_name/dt=20130101'; 

-- 一次添加多个分区
ALTER TABLE page_view ADD PARTITION (dt='2008-08-08', country='us') location '/path/to/us/part080808' PARTITION (dt='2008-08-09', country='us') location '/path/to/us/part080809';  
```

## 删

### 删除表格

- 删除表格：`drop table 表名;`
- 删除所有记录（行），不删除表结构
  `truncate table 表名;`

### 删除分区

```bash
ALTER TABLE login DROP IF EXISTS PARTITION (dt='2008-08-08');

ALTER TABLE page_view DROP IF EXISTS PARTITION (dt='2008-08-08', country='us');
```

### 删外部表数据

外部表，直接`rm -r hdfs`数据，对表`show partitions $table_name` 分区依然存在只是没有数据。

使用`alter table app.app_discovery_content_info_da_tag drop partition (dt='${drop_dt}')`，`show partitions $table_name`分区不存在，但是`hadoop fs -ls` 路径，改分区数据依然存在。

方案一：

将表改为内部表`external =false`，删分区（hdfs数据会删掉），再改回为外部表`external=True`。

方案二：

先删分区，再用rm -r删数据

```
ALTER TABLE some.table DROP PARTITION (part="some")
hdfs dfs -rm -R /path/to/table/basedir
```

方案二存在问题：原本打算将-rm操作放在index_builder中，但是现builder是在rec账号下，没有权限删除路径在recpro下的数据。

所以，最终选方案一，更安全，简洁。

## 改

### 修改表记录

- `update` 修改表记录
  - `UPDATE 表名称 SET 列名称 = 新值 WHERE 列名称 = 某值`
  - 例如 `update tt set name='Joe', salary=70000, managerid=3 where id=1;`

### 修改字段名(列名)

```sql
CREATE TABLE test_change (a int, b int, c int);

-- 修改列名a为a1
ALTER TABLE test_change CHANGE a a1 INT; 

-- 修改列名a为a1，将a1的数据类型置为string，并将a1放在b列后
-- 表的新结构： b int, a1 string, c int
ALTER TABLE test_change CHANGE a a1 STRING AFTER b; 

-- 修改列名b为b1，并将其置为第一列
-- 表的新结构： b1 int, a string, c int
 ALTER TABLE test_change CHANGE b b1 INT FIRST; 
```

### 修改表属性

```sql
-- 内部表转外部表 
alter table table_name set TBLPROPERTIES ('EXTERNAL'='TRUE'); 

-- 外部表转内部表
alter table table_name set TBLPROPERTIES ('EXTERNAL'='FALSE'); 
```

### 表的重命名

```sql
ALTER TABLE table_name RENAME TO new_table_name
```

### 修改分区

```bash
ALTER TABLE table_name PARTITION (dt='2008-08-08') SET LOCATION "new location";

ALTER TABLE table_name PARTITION (dt='2008-08-08') RENAME TO PARTITION (dt='20080808');

-- 将19分区修改到18(外部表的location可能不会变，待验证)
alter table app.xxxx partition(dt='2019-01-19') rename to partition(dt='2019-01-18');
```

### 修改location

一般location在create table if not exist后，不会被触发，而如果删除表则会删除location，因此如果新建表的location错误指向了已经存在的location，需要手动修改新表location
- 命令行新建location目录： `hadoop fs -mkdir 'hdfs://ns9/user/recsys/recpro/app.db/index_qrqm_sku_info_basic_new'`
- `hive`
- `alter table 表名 set locaion 'hdfs://ns9/user/recsys/recpro/app.db/index_qrqm_sku_info_basic_new'`
- 验证：`show create table 表名`查看LOCATION是否更新

## 查

### 常用查询语句

- 查看表字段介绍： `desc 表名`
- select列名:https://stackoverflow.com/questions/26181454/just-get-column-names-from-hive-table
  `show columns in 表名`
- 查某一列：`select 列名 from 表名`
- `order by` 排序(string类型字典序，数值类型按大小排)
  - `order by colname` 默认升序
  - `order by colname desc` 降序
- `limit` 返回对应行
  - `SELECT * FROM table LIMIT 3;` 返回前3行
  - `SELECT * FROM table LIMIT 0,3;` 返回前3行
  - `SELECT * FROM table LIMIT 5,10;` // 从第6行开始，最多返回10行（可能后面的数据不到10行）。即检索记录行 6-15
  - `SELECT * FROM table LIMIT 95,-1;` // 检索记录行 96-last
- 空值数计算：`select sum(if(a is null,1,0)) from ...`
- 空值率计算：`select avg(if(a is null,1,0)) from ...`
- 查询某个字段按字符分割后长度：`size(split(sim_recall,','))`
- 直接查看表信息
  - 通过执行log：`Table 表名 stats: [numFiles=4, numRows=623232, totalSize=234324, rawDataSize=32243]`
  - 通过explain (发现有时通过这种方式查看numRows和count(*)结果不一致)
- sql的模糊匹配 
  - %：表示任意0个或多个字符。可匹配任意类型和长度的字符
    `select distinct * from app.xxxxxx where tb_name like 'app.xxx%';`中`%`不能少(匹配_a等)_
  - `_`： 表示任意单个字符。匹配单个任意字符

### 查询重复的项

- 定位重复行可先group by，记录下count，再对count排序，选count>1的

```sql
select
sku
from
(
select
sku, count(*) as cnt
from
app.xxxx
group by sku
)a
where a.cnt > 1 order by cnt desc limit 100;
```

- 数据中有两列A和B，查看是否存在以下情况：
  A B
  1 2
  2 1
  可以先将AB按大小拼起来，再查询

```sql
select
    sku,
    csku,
    round(score, 10) score,
    if(sku>csku,concat_ws(',',csku,sku),concat_ws(',',sku,csku)) sku_csku_pair
from
    app.xxx
where
    dt = '2018-12-25'
    and tab_name = 'B'
    and sku_sr_type = '0'
```

## tips

- 使用严格模式优点
  - 禁止不指定分区查询
  - 避免join产生笛卡尔积
- 通过时间可以定位hadoop中文件是否被修改，在同一个集市中，如果有其他表指定的location错误，则很可能被修改。
- 如果location指定错误，例如新表指定到旧表路径中，那么会影响旧表。如果直接drop新表，而新表不是external表，则会删除location中文件，无法恢复。所以，需要修改新表的location，在修改之前不要对新表做任何操作。
- `concat_ws` 需要数据类型为 `string`
- `null`
  - `where col=NULL` 不会返回结果
  - `where col is null` 会返回为空的列
- `join`
  - `left join on` 保留左边表关键字信息（即使右表中没有匹配，也从左表返回所有的行 ），可以使用`where 右表.col is not null`实现join效果
  - `right join on` 保留右边关键字信息(即使左表中没有匹配，也从右表返回所有的行)
  - `inner join` 两边都有才保留，也可写为`join`
  - `full join` 只要其中一个表中存在匹配，就返回行
- 这么写中间表`(${subSql}) t`只查了一遍

```sql
from(${subSql}) t
    insert into table ${num_rows_tb} partition
        ( tb_name = '${tb_name}'
        )
    select ct, '${select_dt}', sysdate()
    insert overwrite table ${null_rate_tb} partition
        ( tb_name = '${tb_name}'
        )
    select
        split(kv, ':') [0] as k,
        cast(split(kv, ':') [1] as double) as v,
        '${select_dt}' as select_dt lateral view explode(split(kvs, ',')) kvs as kv;"
```

- 注意 lateral view explode要写在from的后面，上述如果不将from写在最上方，应该写为如下，而不能将from放在最后

```sql
 select
        split(kv, ':') [0] as k,
        cast(split(kv, ':') [1] as double) as v,
        '${select_dt}' as select_dt
from(${subSql}) t
 lateral view explode(split(kvs, ',')) kvs as kv;
```

- map数过大不一定运行快，过大的map数会被pending。
- 在join时，如果on的是double数据类型，需要注意小数的精度不同可能造成=判定失败。可以用round(score, 2) score 只保留2位小数后再对比。

 

## hdfs文件操作命令

### 常用

- `hadoop fs -mkdir path`创建文件夹
- `hadoop fs -put local_path hdfs_path` 上传本机的HDFS文件。local_path指非hdfs路径，例如/home/name
- `hadoop fs -get hdfs_path local_path` 把HDFS的文件下载到本机
- `hadoop fs -cat file_name` 读取HDFS文件
  1. hadoop fs 和hdfs dfs 作用一样。都可以在本机上查看HDFS文件。
  2. HDFS下的文件可以压缩存储，这样能够减少表查询时对Hadoop集群的IO。

### 从文件导数据到表

- 使用put将test.txt文件放到hdfs中：`hadoop fs -put '/home/test/test.txt' /usr/recsys/rec/index_data/test`
- 使用load hdfs地址将数据导入表中：`load data inpath '/user/recsys/rec/index_data/test/test.txt' into table app.test;`
- 导入成功，可以在select limit表检查确认一下。

### `hadoop fs -ls [path]`查看HDFS文件名

后面不加目录参数的话，默认当前用户的目录。ls会显示文件详细信息，如果只想ls出文件名，可以这么写：（参考 <https://stackoverflow.com/questions/21569172/how-to-list-only-the-file-names-in-hdfs> ）

```
hadoop fs -ls <HDFS_DIR>|cut -d ' ' -f17
```

或者先`sed '1d'` 删除第一行（第一行是总述信息），将多个空格替换为一个空格(`sed 's/要被取代的字串/新的字串/g'`)再cut：

获取文件名：`hadoop fs -ls | sed '1d;s/ */ /g' | cut -d\ -f8`

获取文件名最后一列（这里是dt=2018-10-27）：`hadoop fs -ls | sed '1d;s/ */ /g' | cut -d\ -f8 | xargs -n 1 basename`

更优解：`hadoop fs -ls /tmp | sed 1d | perl -wlne'print +(split " ",$_,8)[7]'`

最终采用：`hadoop fs -ls | sed '1d;s/ */ /g' | cut -d\ -f8` 获取文件名列表

```bash
# 遍历文件。
hdfs dfs -ls ${path} | sed '1d;s/  */ /g' | cut -d\  -f8| while read line
    do
        echo ${line}
done 
```

对每个文件，`$(echo ${line}|cut -d "=" -f 2)` 取分区数据。以`=`分割取第2个元素

补充：sed介绍： <https://www.cnblogs.com/ggjucheng/archive/2013/01/13/2856901.html>

### 常用设置

- 集市限制小文件个数，可在脚本中加以下参数合并小文件：

```sql
set hive.merge.mapfiles = true
set hive.merge.mapredfiles = true
set hive.merge.size.per.task = 256000000
set hive.merge.smallfiles.avgsize = 104857600
```
