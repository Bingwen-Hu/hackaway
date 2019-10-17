# 将官方 Python 运行时用作父镜像
FROM ubuntu:16.04
ENV LANG C.UTF-8

# 更换apt国内源（阿里源）
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak \ 
    && mv ./sources.list /etc/apt/ \ 
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -y update \

# 安装python3.6
RUN apt-get -y install software-properties-common \ 
    && add-apt-repository ppa:jonathonf/python-3.6 \ 
    && apt-get -y update  \
    && apt-get -y install python3.6 \
    && apt-get -y install python3-pip
