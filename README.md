# 20250521-Anaconda环境配置  

## Pytorch 环境配置  

管理员模式启动CMD  

> conda create -n pytorch python=3.9
> % 创建一个名称为"pytorch"的虚拟环境  
> conda init  
> % 启动Anaconda  
> conda activate pytorch  
> % 启动该虚拟环境"pytorch"  
> conda install pytorch  
> % 向虚拟环境中安装pytorch包  

## Git仓库配置

### 仓库创建及推送

如果你的本地项目还没有初始化为 Git 仓库，需要先在项目根目录下运行以下命令：  
> git init

在本地仓库中，需要将本地仓库与远程仓库关联起来(或者SSH):  
> git remote add origin <远程仓库的URL>  

推代码至远程仓库:  
> git push origin master

文件加入待推送位置:
> git add README.md  

附带推送信息:
> git commit -m "test success"

推送上传:
> git push origin main  

### Git忽略推送(大文件)

新建.gitignore文件并编辑无需追踪的部分文件夹或者文件

### 仓库文件拉取

由新电脑拉取文件时：
> cd ~/.ssh

git bash 中输入：
> ssh-keygen -t rsa -C "XXXX@XX.com"

按路径进入 .ssh，里面存储的是两个 ssh key 的秘钥，id_rsa.pub 文件里面存储的是公钥，id_rsa 文件里存储的是私钥，不能告诉别人。打开 id_rsa.pub 文件，复制里面的内容。  
接下需要登录到自己的 GitHub 上边添加这个密匙。  

### Git分支

同时创建：仓库+分支：
> git init -b <分支名>

已有仓库，再创建分支：  
> git branch <分支名>

创建并切换到分支:
> git checkout -b <分支名>

分支重命名：
> git branch -m <分支名> <分支名>

分支删除：
> git branch -d <分支名>