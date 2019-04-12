# 目前用到的Git命令

## 配置Git
- `git config --global user.name "username"` - 配置姓名

- `git config --global user.email "useremail"` - 配置邮箱

## 创建项目
- `git init` - 可以创建一个空的代码仓库
- `git clone git@github.com:smallFF/Reservoir.git` - 可以直接克隆为本地代码库

- 我是通过GitHub上创建代码库，然后再直接克隆到本地的，后面对于 `push` 操作比较方便

## 检查状态
- `git status` - 查看项目状态

## 把文件加入仓库
- `git add .` 或者 `git add -A` - 添加所有更改的文件

- `git add filename` - 添加单个文件

- 这里的加入到仓库只是加入到暂存区，不执行提交不会计入版本

## 执行提交
- `git commit -m "Descriptions"` - 一般用于 `git add` 相关命令后，加入到仓库了就执行提交操作，以便记录版本

- **提交之前一定要用`git add`命令添加文件！！！**

## 查看提交历史
- `git log` - 以详细信息显示所有提交历史

- `git log --pretty=oneline` - 对于每次提交历史都以一行简要描述

## 撤销修改
- `git checkout .` - 撤销最后一次提交后所做的所有改变

- `git checkout` - 此命令加上相关参数能够恢复到以前的任何提交，但是我没怎么用过，之后有需求再细查

## 检出以前的提交
- `git checkout id` - 这里的 `id` 是前面通过 `git log` 得到特定版本的id，通常对于此命令，我们只取id的前6个字符作为 `git checkout id` 里面的id参数。**进行这个操作了就离开了主分支，如果没有必要，不要在这种状态下对项目进行修改！！！**

- `git checkout master` - 回到主分支

## 删除仓库
- `rmdir /s .git` - 只需要删除.git文件夹就可以删除这个Git仓库了

- **这个操作只是删除了之前对项目操作的历史记录，不会删除我们的代码文件**

## 项目同步
- `git push origin master` - 把本地代码推送到GitHub的代码仓库中

- `git pull origin master` - 把GitHub代码仓库中的代码拉取到本地

- 我一般是两台电脑轮流用，通过 `push` 和 `pull` 这两个操作，每次在一台电脑上工作完了就 `push` 到云端，等到另一台电脑上要继续工作时就 `pull` 一下，然后工作完了再`push` 到云端。就这样来回 `push` 和 `pull` 操作，代码完全同步，不需要用U盘来回拷代码，非常爽！！！


最后说一下，因为目前只是我自己工作，所以我用的最多的是[**项目同步**](##项目同步)的功能，后续根据需要可以学习更高级的Git功能！