---
title: hexo 多pc同步（二）
date: 2016-05-15 09:00:39
tags: hexo
---
因为有两台电脑，所以搜索了下多pc同步问题。这样，不论是重装系统，还是换机器，都能方便管理自己的博客。

<!--more-->

首先介绍一下[hexo和博客源文件之间的关系](http://ywtail.github.io/2016/05/14/hexo-1/ "")。
> 1.hexo帮助把博客发送到github，同时把md文件转换成网页文件。

> 2.hexo的安装编辑等都和github上显示的内容不是同一东西，也就是，用hexo生成发布了github博客后，hexo的那些文件是没有传到github上的，还是在本地，传到github上的只是由hexo生成的用来显示的网页文件。如果想两台电脑同时使用hexo进行同一博客维护，那么这两台电脑都得安装有hexo的，如果两台电脑只是单纯的把github上最后显示出来的文件clone下来是没有用的，而是要把第一台维护博客是的hexo的源文件clone下来（得先提交到github），再在同一份的hexo源文件中进行维护，然后再生成博客的网页文件。

> 举个栗子：hexo源文件就好比母本(本地的md文件)，而hexo deploy传到github上的只是母本的一个镜像(其实是网页文件)，我们要操作的时候在第二台电脑上要拿到母本，然后操作母本，那么第二台电脑再执行hexo deploy的时候就和第一台电脑一样了，如果第二台电脑拿到的是镜像，那怎么可能和第一台电脑一样进行维护呢？

> 具体做法是：把hexo的文件上传到git托管云如github上，然后在第二台电脑要把这些hexo的源文件clone下来，因为这些源文件内还有了生成博客需要的md原始文件，所以只要有了源文件就可以再次生成博客展示的文件


所以在本文中，首先，将本台机器A中的内容推到github。

1. 在github添加仓库，例如hexo。
2. 将本台机器A中的内容推到github远程仓库中。

然后，在机器B将相关内容clone下来，完成后将最新内容推到github。在机器B操作流程如下：

1. 安装git；添加ssh（参见[hexo 简单搭建（一）](http://ywtail.github.io/2016/05/14/hexo-1/ "")。
2. 在机器B上`npm install -g hexo`安装hexo。
3. 在B上新建文件夹，从github中clone相关内容。
4. `npm install`安装依赖包。
5. 写文章，做修改后将最新内容推到github远程仓库中。

具体如下。

---

## **在github添加仓库**

完成[hexo 简单搭建（一）](http://ywtail.github.io/2016/05/14/hexo-1/ "")，则已经安装好了git，也添加好了ssh。所以在本机A中，直接在github中添加远程仓库hexo。具体方法参见[添加远程库](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/0013752340242354807e192f02a44359908df8a5643103a000 "")。

## **将hexo推送到github**

介绍：在`H:\hexo`中有文件`.gitignore`，这个文件是hexo初始化带来的，作用是声明不被git记录的文件，`.gitignore`包含以下内容。
```
node_modules/
public/
.deploy*/
```

使用github备份hexo，方法如下。

1.在`H:\hexo`中，右键选择Git Bash，输入如下，完成后会生成一个.git文件。
```
git init
```

2.输入如下。`<server>`是指在线仓库的地址（例如我的就是git@github.com:ywtail/hexo.git）。origin是本地分支,remote add操作会将本地仓库映射到云端。
```
git remote add origin <server>
```

3.输入如下，将内容推送到github。有疑问参见[Git教程](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/0013743256916071d599b3aed534aaab22a0db6c4e07fd0000 "")。
```
git add .  #添加blog目录下所有文件，注意有个`.`（`.gitignore`声明过的文件不包含在内)
git commit -m "first commit" #添加更新说明
git push -u origin master #推送更新到云端服务器
```

完成后就能在github的对应仓库中看到推送的hexo相关内容了。

## **从github克隆**

换电脑后，重新安装git、node.js、添加ssh，任意目录下`npm install -g hexo`安装hexo。然后，新建一个文件夹hexo，在hexo下运行
```
git init
git remote add origin <server> #将本地文件和云端仓库映射起来。这步不可以跳过
git fetch --all
git reset --hard origin/master
```
其中，fetch是将云端所有内容拉取下来。reset则是不做任何合并处理，强制将本地内容指向刚刚同步下来的云端内容。

## **更新同步**

clone下来后使用`npm install`安装依赖包。
完成后生成`node_modules/`等文件。

然后就可以自由地写博客了，写完后重新同步到github，方法如上。

**参考**

- [hexo和博客源文件之间的关系捋清](http://ywtail.github.io/2016/05/14/hexo-1/ "")
- [廖雪峰的Git教程](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000 "")
- [利用git解决hexo博客多PC间同步问题](http://chitanda.me/2015/06/18/hexo-sync-in-multiple-pc/ "")
