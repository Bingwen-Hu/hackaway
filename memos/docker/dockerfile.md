# Dockerfile 精要

本篇介绍dockerfile的精要，docker常用命令见[这篇](./docker.md)。

#### 1. 指定基础镜像
```dockerfile
FROM ubuntu:18.04
FROM scratch
```
基本镜像是dockerfile的第一行，示例一指定了ubuntu作为基础镜像，示例二表明不需要任何基础镜像。

#### 2. 执行命令
```dockerfile
RUN apt update \
    && apt install -y python3.6 \
    && apt install 
``` 

#### 3. 从宿主系统复制文件到镜像内部
```dockerfile
COPY ./sources.list /etc/apt/
```
直观感觉，就是把当前路径下的`sources.list`复制到`/etc/apt/`。但这个当前路径，是相对于构建镜像时指定的`上下文路径`，而不是宿主系统的路径，更不是镜像里的路径，详见下一节。

#### 4. 创建镜像
```sh
# 目录格式1
# ├── Dockerfile
# └── sources.list
docker build -t myimage:v3 .

# 目录格式2
# ├── Dockerfile
# └── files
#     └── sources.list
docker build -t myimage:v3 file
```
`-t myimage`指定了镜像的仓库名，`v3`指标签。跟上一节对应，`.`指定了当前路径作为镜像中的上下文路径，`files`则指定了把当前路径下的文件夹`files`作为上下文路径。这样，上一节的`COPY`命令才能正确地找到`sources.list`文件。

#### 5. 执行命令
```dockerfile
CMD ["curl", "-s", "http://ip.cn"]
```
`CMD`通常放在`dockerfile`的最后一行，容器启动时常使用这个命令。

#### 6. 其他
```dockerfile
# 定义环境变量
ENV HOME /home/mory
# 声明使用的端口
EXPOSE 6379
# 指定工作目录，在这之后所有的操作都在这个目录下进行
WORKDIR /app
# 指定用户
USER redis
# 延迟执行
ONBUILD COPY . /app/
```