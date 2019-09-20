Pip 是一个Python的包管理工具，主要用于安装和删除第三方包，搭建开发环境依赖，是工作中最常用的工具，没有之一！赶紧让我们一起来了解下吧 :)


### Pip 的位置
在 Linux 和 MacBook 环境下，输入
```sh
which pip
```
可以得到 pip 的路径。因为有时候我们会配置多个 Python 环境，因而知道当前是哪个环境的 pip 就非常重要了。因为 pip 只会把包安装到它所在的环境中。如果你发现你用 pip 安装了包，导入时却总是说没找到这个包，那很可能是这个问题哦～

最新的 ubuntu 已经不带 Python2 的版本了。如果你电脑上有 Python2 和 Python3，那么通常 pip2 是 Python2 环境的，pip3 是 Python3 环境的。那 pip 呢？用 which 看看:-)

### 安装 Python 包
比如我们想安装一个叫 easydict 的 Python 包，可以这样
```sh
pip install easydict
```
如果想
安装一堆包，比如 requests, bs4, pandas, numpy ... 可以在一行中写完，如
```sh
pip install requests bs4 pandas numpy
```

### 修改安装源
[PyPi](https://pypi.org/) 是 Python 第三方包的仓库，地点在于美国。如果你安装的包特别大，比如 200MB 以上的，有时候会有网络连接中断的问题。当然取决于你的网速。如果有这种问题，修改安装源就非常有必要，毕竟中国国内就有好多优秀的安装源，跟老美的是一样一样的，同一个镜子照出来的，但网速却快得多！如何配置呢？

假设用户名是 mory，Windows下 win+R 打开运行界面，输入 cmd 启动命令行。Linux 或者 Mac 直接打开终端，输入 cd，直接跳到你的用户目录下。通常是 C:\Users\mory 或者 /home/mory 或者 /User/mory 这样的路径。我们把这个路径叫做家目录。

分步走，第一步创建一个叫 .pip 的文件夹。在 Linux 下以句号 . 开头的文件夹都是隐藏文件，不过在 Windows 下它们都是可见的。

```sh
mkdir .pip # create a directory named .pip
```

第二步创建一个 pip.conf 的文件。conf 是 configuration 的惯用缩写，也就是配置的意思。你可以用 Windows 的记事本或者 Linux 和 MacBook 下的 vi来创建，这里用 vi 创建。
```sh
vi pip.conf
```

输入以下内容
```sh
[global]
index-url=https://pypi.douban.com/simple
```
这里我们用的是可爱的豆瓣的镜像，当然清华，阿里和网易都有镜像，只是据说豆瓣比较快。

保存文件，大功告成！享受飞一样的下载速度吧！

### 删除包
废话少说，放码过来。

```sh
pip uninstall name-of-package
```

### 用户管理
有时候，几个人共享同一套环境，但环境里没有你要的安装包，而你又没有权限去修改大环境的包，这时候你可以把包只安装到自己的目录下，只对你自己可见，而其他人不可见。怎么做呢？

```sh
pip install name-of-package --user
```
通过这种方式安装的包，通常位于家目录下的 .local/lib/ 里面，类似这种样子

```sh
.local/lib/python3.6/site-packages/
```

### 开发者模式
如果你所安装的包，是你自己开发的，那就有一边开发一边测试的需求。如果每次做一小步修改，都做 pip uninstall 和 pip install 一次，那就有点低效了。Pip 支持开发者模式安装，它会创建一个连接，使得你可以轻松 import 你的包，同时你做的修改，也会立即生效。

```sh
cd name-of-package
pip install -e .
```
大部分人看不懂的人都不需要走到这一步，走到这一步的人都知道我在说什么:0


### setup.py
有时候一些开发者没有把包上传到 PyPi 上，所以你只能在他的 github 下找到这个包，然后通常会有个 setup.py 的脚本，通过以下命令来安装。

```sh
python setup.py install
```
这个跟 pip 有什么关系吗？当然这是 setup.py 自家的事，不过 pip 也可以掺和一下下嘛

以上的命令等于
```sh 
pip install .
```

setup.py 下的开发者模式是这样的

```sh
python setup.py develop
```

以上的命令等于

```sh
pip install -e .
```
