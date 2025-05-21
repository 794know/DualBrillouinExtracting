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
