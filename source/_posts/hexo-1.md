---
title: hexo 简单搭建（一）
date: 2016-05-14 22:45:55
tags: hexo
---
简单记录下搭建hexo的过程。

## **安装Git**

[下载](http://git-scm.com/download/ "")并安装。

## **安装Node.js**

[下载](https://nodejs.org/ "")，安装并配置环境变量。
**注意：** 在[官网](https://nodejs.org/download/ "")下载的exe文件不能安装，会导致在安装hexo的过程中各种提示command not found，例如提示“sh: npm: command not found”。
因此，直接点击“install”，保存.msi文件，再安装node.js。

## **安装hexo**

在任意位置点击鼠标右键，选择Git bash。输入以下命令：
```bash
npm install -g hexo
```

## **创建hexo文件夹并安装依赖包**

在H:\hexo内右键，选Git bash。输入以下命令
```bash
hexo init
npm install
```
## **本地查看**

输入以下命令后，在浏览器查看。端口为4000。（127.0.0.1:4000）
```bash
hexo generate
hexo server
```
generate也可简写为g，同理，server为s。

**注意：**

1.hexo s无效并出现如下提示信息时
```bash
$ hexo s
Usage: hexo <command>
```
使用以下命令安装server
```bash
npm install hexo-server --save
```

2.打开localhost:4000，显示Cannot GET /
重新npm install，就好了。

现在只能在本地查看，下面部署到github。

## **创建repository**

创建方法为+New repository --> 输入Repository name --> Create repository。
![][1]
**注意：** repository名字必须与github上名字一样！！即格式必须为`yourname/yourname@github.com`或`yourname/yourname@github.io`。

## **添加SSH key**

参照[GitHub官网][2]给出的方法。
下面简单写一下。

1.检查电脑是否已经有SSH key
在用户主目录下，看看有没有.ssh目录，如果有，再看看这个目录下有没有id_rsa和id_rsa.pub这两个文件，如果已经有了，可直接跳到步骤3。
或者用以下方式查看。
```bash
$ ls -al ~/.ssh
# Lists the files in your .ssh directory, if they exist
```

2.如果没有SSH key，则创建新的SSH key（（Windows下打开Git Bash）。
```bash
$ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# Creates a new ssh key, using the provided email as a label
```
觉得不需要设置密码就一路回车。
完成成功后可以在用户主目录里找到.ssh目录，里面有id_rsa和id_rsa.pub两个文件，这两个就是SSH Key的秘钥对，id_rsa是私钥，不能泄露，id_rsa.pub是公钥，可以公开。

3.将.ssh文件中id_rsa.pub中的内容拷贝，添加到github。具体：在GitHub右上方点击头像，选择”Settings”，在右边的”Personal settings”侧边栏选择”SSH Keys”。接着粘贴key，点击”Add key”按钮。

## **部署**

编辑H:\hexo\_config.yml。
```bash
deploy:
  type: git
  repository: https://github.com/yourname/yourname.github.io.git
  branch: master
```

执行以下命令完成部署。

```bash
hexo g
hexo d
```
其中，d为deploy。

休息一会，喝几口冷水，再在浏览器访问yourname.github.io就可以看到自己的博客了。
**注意：** 

1.如果在执行hexo d时，报错如下（找不到git部署）

```bash
ERROR Deployer not found: git
```
解决方法如下
```bash
npm install hexo-deployer-git --save
```
再重新执行g，d即可访问。

2.如果执行hexo d时，提示如下
```bash
*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.
```
按照上面的提示输入email和name。
```bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

3.不论在创建repository时是`yourname/yourname@github.com`还是`yourname/yourname@github.io`，都必须访问yourname.github.io才能访问。yourname.github.com是不能够访问的。

**参考**

- [hexo系列教程：（二）搭建hexo博客][3]
- [hexo你的博客][4]
- [使用hexo和Github上创建自己的博客][5]
- [廖雪峰的官方网站][6]


  [1]: http://7q5c08.com1.z0.glb.clouddn.com/20150827hexo_1.png
  [2]: https://help.github.com/articles/generating-ssh-keys/
  [3]: http://zipperary.com/2013/05/28/hexo-guide-2/
  [4]: http://ibruce.info/2013/11/22/hexo-your-blog/
  [5]: http://www.itnose.net/detail/6231502.html
  [6]: http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/001374385852170d9c7adf13c30429b9660d0eb689dd43a000