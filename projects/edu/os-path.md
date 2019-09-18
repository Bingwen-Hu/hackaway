os.path 表示操作系统路径 (Operation System Path) ，因而封装的是一些路径操作相关的功能，比如获取文件的绝对路径，合并路径等等。下面我们一起来了解下吧 :)


### 文件是否存在
这个功能我们前一节已经前过了呀

```py
import os.path as osp
osp.exists('docs')
```

### 获取绝对路径
路径分绝对路径和相对路径。绝对路径有点像是：    
我住在地球村/中国/广东省/广州/天河/A街道/B路/123号/04房   
相对路径有点像是：   
我住在A街道/B路/123号/04房

通常新手写代码会处理不好路径的问题，常常发生编程事故～

比如说，edu文件夹下包括两个文件
```sh
edu
├── os.md
└── os-path.md
```

执行代码
```py
import os.path as osp
osp.exists('os.md') # False
osp.exists('edu/os.md') # True
```
第一个返回False是因为当前目录下确实没有 os.md 这个文件，正确的路径是第二个语句。

获取绝对路径
```py
import os.path as osp
osp.abspath('edu/os.md')
```

### 获取文件名
有时候你得到一个路径，但你需要处理路径最后端的文件名

```py
import os.path as osp
path = "/home/mory/somewhere/mylogo.png"
osp.basename(path) # mylogo.png
```

顺带地，如果你想获取前面的文件夹，这种情况比较少见。
```py
import os.path as osp
path = "/home/mory/somewhere/mylogo.png"
osp.dirname(path) # /home/mory/somewhere
```

### 合并路径
合并路径是工作中最常见的路径操作，没有之一。

```py
import os.path as osp
somedir = "/home/mory/somewhere"
somefile = "mylogo.png"
osp.join(somedir, somefile)
# => /home/mory/somewhere/mylogo.png
```

### 切分路径
切分路径其实就是分别获取路径的文件夹和文件部分

```py
import os.path as osp
path = "/home/mory/somewhere/mylogo.png"
somedir, somefile = osp.split(path)
```

### 系统分隔符
目前三大主流系统，Windows 跟其他两个的路径表示并不相同，如果需要写跨平台的代码，知道这个就有点用

```py
import os.path as osp
print(osp.sep) # OS-dependent
```