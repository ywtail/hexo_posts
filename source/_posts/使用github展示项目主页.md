---
title: 使用github展示项目主页
date: 2016-06-03 22:15:15
tags: git
---
本文主要流程如下：

* 简单介绍git相关命令
* 使用github展示项目主页：
    1. 创建名为`gh-pages`的分支；
    2. 将要展示的内容放到`gh-pages`分支下（必须如此）；
    3. 访问 http://Github用户名.github.io/项目名。
* 多pc使用：
    1. 从github上克隆项目；
    2. 克隆分支；
    3. 将内容推送到`gh-pages`分支。

## git相关命令

```bash
查看分支：git branch

创建分支：git branch <name>

切换分支：git checkout <name>

创建+切换分支：git checkout -b <name>

合并某分支到当前分支：git merge <name>

删除分支：git branch -d <name>

查看远程分支 git branch -a
```

在学习使用分支的过程中发现，在本地删除了分支，但是github上依然存在，找了两个删除远程分支的方法。

```bash
删除远程分支：git push origin :<name>

删除远程分支：git push origin --delete <name>
```

## 使用github展示项目主页

### 创建`gh-pages`分支

* 使用如下命令会给项目创建一个 `gh-pages` 分支并切换到该分支。其中，`--orphan` 表示该分支是全新的，不继承原分支的提交历史（默认 git branch gh-pages创建的分支会继承 master 分支的提交历史，所以就不纯净了）。
```
git checkout --orphan gh-pages
```

* 接下来把新分支中的文件删掉（这一步可以不执行。这个命令会删除本地当前文件夹下所有内容，如果不想删就不执行这个。）
```
git rm -rf .
```

>**注意：**
这里 `git branch` 是显示不出 `gh-pages` 分支的（需要做一次提交才行），不要着急，一直进行到push完毕才会显示的。

### 将要展示的内容放到`gh-pages`分支

有以下两种方式：
* 直接把需要的拷贝过来(如果没有执行`git rm -rf .`可以直接提交，文件都在当前目录呢)，然后开始提交
```
git add .
git commit -a -m "test"
git push origin gh-pages
```

* merge别的分支(例如merge master分支)，然后提交.
```bash
git merge master #merge别的分支
git push origin gh-pages #提交
```

>**注意：** 
1. 必须将要展示的内容放到 gh-pages 分支下。
2. 之前看的教程习惯使用`git commit -m "<Explanation>"`，但是如果在本地删除了文件，在github上依然存在。使用`git commit -a -m "<Explanation>"`只是多了`-a`，就能把删除行为加上，使github上显示和本地就完全相同。
3. 第一次提交使用`git push origin <name>`这么长，以后直接`git push`就ok。

### 访问

 访问 `http://<Github用户名>.github.io/<项目名>`就可以查看了。
 
## 多pc使用

### 从github上clone

知道仓库的地址，然后使用`git clone`命令克隆（Git支持多种协议，包括https，但通过ssh支持的原生git协议速度最快）。clone地址直接从github复制，格式大约如下。
```
git clone git@github.com:<用户名>/<项目名>.git
```

### 克隆分支

git clone默认会把远程仓库整个给clone下来，但只会在本地默认创建一个`master`分支。所以首先查看有哪些分支，然后clone自己需要的分支（使用-t参数，它默认会在本地建立一个和远程分支名字一样的分支）。

```
git branch -a #查看有哪些分支
git checkout -t origin/<分支名> #clone分支
```

### 将内容推送到`gh-pages`分支

本地更改后，将内容推送到`gh-pages`分支，就能够使用github展示项目主页。（最后可以将内容merge到master）

**参考**

- [深入 Github 主页](https://www.awesomes.cn/source/10 "")
- [Git 分支 - 远程分支](https://git-scm.com/book/zh/v1/Git-%E5%88%86%E6%94%AF-%E8%BF%9C%E7%A8%8B%E5%88%86%E6%94%AF "")
- [廖雪峰：分支管理](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/001375840038939c291467cc7c747b1810aab2fb8863508000 "")
- [Git clone远程分支](http://www.tuicool.com/articles/6fmQRnq "")