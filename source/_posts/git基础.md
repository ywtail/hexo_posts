---
title: git基础
date: 2016-07-16 14:58:32
tags: git
categories: git
top: 1
---

### 将内容添加到远程仓库

1. 在github上Create a new repository
2. 在本地新建文件夹test，里面存放需要放到github上的内容。
3. 在test文件夹下：

```bash
git init
git remote add origin git@github.com:ywtail/repositoryName.git
git add .
git commit -a -m 'test'
git push -u origin master
```

**注意：**test目录中必须有文件，需要先add,commit再`push -u origin master`。否则会报错：`error: src refspec master does not match any.`

以后对文件夹中内容修改后提交：
```bash
git add .
git commit -a -m 'description'
git push
```

### 其它问题

如果在`push -u origin master`后报错如下
```bash
ssh: connect to host github.com port 22: Bad file number
fatal: Could not read from remote repository.
```
则可以通过以下方案解决：
>该方案来自：http://stackoverflow.com/questions/7144811/git-ssh-error-connect-to-host-bad-file-number
中文版：http://qa.helplib.com/221029

在`.ssh`文件夹中新建文件`config`，即创建文件`~ / . ssh / config `
在config中贴入以下代码并保存：
```bash
Host github.com
User git
Hostname ssh.github.com
PreferredAuthentications publickey
IdentityFile ~/.ssh/id_rsa
Port 443
```
完成后再`push -u origin master`就成功了。