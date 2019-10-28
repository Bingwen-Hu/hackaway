# Docker 命令精要

这一篇只介绍docker命令，涉及dockerfile的内容在[另一篇文章](dockerfile.md)。

### 安装
```sh
curl -fsSL https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
"deb [arch=amd64] https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu \
$(lsb_release -cs) \
stable"
sudo apt-get update
sudo apt-get install docker-ce
```

### 启动Docker服务
docker作为一项系统服务，需要先启动其守护进程。
```sh
sudo systemctl enable docker
sudo systemctl start docker
```

### 添加Docker用户组
将当前用户添加入docker用户组，这样就可以使用docker了。
```sh
sudo groupadd docker
sudo usermod -aG docker $USER
```

### Docker的Hello-world
```sh
docker run hello-world
```
如果这一步成功了，说明你前面的工作没白费:-)。

### 拉取镜像
```sh
docker pull blvc/caffe
docker pull ubuntu:18.04
```
blvc是用户名，caffe是仓库名，没有版本号，所以用`lastest` 。ubuntu是仓库名，18.04是版本号，没有用户名，所以是官方默认`library`

### 查看镜像
```sh
docker images
docker image ls -a
docker image ls ubuntu
```
第一行只查看上层镜像，第二层查看了所有镜像，包括那些为上层镜像服务的默默无闻的中间层镜像。第三行只列出那些ubuntu仓库下的镜像。


### 删除本地镜像
找到想删除的镜像的ID，用下列的命令来删除
```bash
docker image rm MAEG_ID
```
由于镜像跟容器的依赖关系，只有当所有基于此镜像的容器或者上层镜像都被删除之后，镜像才能被删除。

### 创建容器
镜像跟容器的关系，就像类和实例的关系，也就像汽车设计图跟汽车的关系，像面包模具跟面包的关系。
```bash
docker run -it ubuntu:18.04 bash
``` 
这样基于ubuntu18.04启动了一个容器，并且进入了容器的bash，也就是命令行啦。

创建容器的时候可以指定容器名
```sh
docker run --name webserver -it ubuntu:18.04 bash
```
将容器命名为`webserver`

### 查看容器
```sh
docker container ls
docker container ls -a
```
第一行查看正在运行的容器，第二行查看所有容器。

### 启动与停止容器运行
```sh
docker start webserver
docker stop webserver
```
只有当容器启动了才可以进入。

### 进入容器
创建容器一节使用`docker run`来创建容器。容器创建好了之后，我可能想多次进入做一次探索和修改，进入容器的命令
```sh
docker exec -it webserver bash
```
因为我创建了一个`webserver`容器，所以我就用它的名字。当然也可以用ID，可是ID不好记。

### 保存修改以及查看历史修改
我们进入容器之后，可能做了一些修改，比如下载某些软件，执行了一些命令，想保存到镜像，可以使用
```sh
docker commit \
    --author "mory" \
    --message "add python3.6" \
    webserver \
    ubuntu: 18.04v2
```

查看commit的历史。
```sh
docker history ubuntu:18.04v2
```

### 其他

#### 1. 常用命令
```sh
docker info  # 会输出系统信息，以及镜像，容器等情况
docker rename oldname newname # 容器重命名
docker system df # 查看磁盘占用
```

#### 2. 使用国内镜像源
由于docker服务默认使用国外的源，下载速度比较慢，所以对于ubunut系统，在`/etc/docker/daemon.json`写入
```json
{
    "registry-mirrors": [
        "https://registry.docker-cn.com"
    ]
}
```
之后重启服务
```sh
sudo systemctl daemon-reload
sudo systemctl restart dock
```
这样使用docker在国内的镜像，来提升下载速度。

#### 3. none镜像
none镜像是没有用的，一般是由于镜像命名冲突所导致的。可以是远程拉取镜像更新导致，也可以本地创建时导致。
```sh
docker image ls -f dangling=true
```
删除之
```sh
docker image prune
```

#### 4. 在容器和宿主机上进行文件传输
```bash
# from os to container
docker cp /os/path [container]:/docker/image/path
# from container to os
docker cp [container]:/docker/image/path /os/path
```

### 参考
+ https://yeasy.gitbooks.io/docker_practice/content/