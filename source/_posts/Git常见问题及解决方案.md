---
title: Git常见问题及解决方案
date: 2017-06-07 11:09:21
tags: git
categories: git
top: 2
---

### git 重复上次失败的提交

在`git push`时失败了，提示有单个文件超过 100M，所以不能 push。
在本地删除了这个文件，再 `git add`、`git commit -a -m`、`git push` 依然失败，因为依然会进行上一次失败的 push，超过 100M 的文件依然尝试上传，即使此时在本地已经删除了。

此时`git commit` 或 `git status` 提示如下
```bash
Your branch is ahead of 'origin/master' by 3 commits.
```

解决方案参考：https://stackoverflow.com/questions/16288176/your-branch-is-ahead-of-origin-master-by-3-commits

因为不想再提交这个超过 100M 的文件，所以使用 `git reset --hard origin/master` 将本地重置为远程 git。再提交就成功了。

### 删除本地 git init 过的文件 override r--r--r--
使用 `rm -r filename` 删除本地`git init`过的文件时，提示如下
```bash
override r--r--r-- XXXXXXX? 
```
解决方案参考：https://unix.stackexchange.com/questions/104330/how-do-i-override-rw-r-r-root-wheel-for-a-all-files-in-a-directory

最终使用 `sudo rm -r filename` 删除成功。
