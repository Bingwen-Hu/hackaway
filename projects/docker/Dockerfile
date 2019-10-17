# 将官方 Python 运行时用作父镜像
FROM ubuntu:18.04
ENV LANG C.UTF-8

# 更换apt国内源（阿里源）
COPY sources.list /var/local/

RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak \ 
    && mv /var/local/sources.list /etc/apt/ \ 
    && apt-get -y clean \
    && apt-get -y update \
    # python 36
    && apt-get -y install software-properties-common \
    && add-apt-repository ppa:jonathonf/python-3.6 \ 
    && apt-get -y update  \
    && apt-get -y install python3.6 \
    && apt-get -y install python3-pip
