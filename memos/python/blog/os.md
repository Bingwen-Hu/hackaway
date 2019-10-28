## os 模块

os 表示操作系统 (Operation System ) ，因而封装的是一些操作系统相关的功能，比如创建文件夹，删除文件夹等等。下面我们一起来了解下吧 :)


### 创建文件夹

以下是最基本的操作，创建一个叫 docs 的文件夹。

```py
import os
os.mkdir('docs')
```

确认是否有某文件夹，如果没有，创建之
```py
import os
if os.path.exists("my_directory"):
    os.mkdir("my_directory")
```

如果文件夹有就不创建，没有就创建
```py
import os
os.makedirs("my_directory", exist_ok=True)
```

创建多级目录，比如 我的文档/读书笔记/Python学习笔记
```py
import os
os.makedirs("learning/notes/python")
```

### 切换当前目录

当前目录指的是你当前工作的路径，如果你是在Linux或者Mac上工作，在终端输入 pwd ，可以得到当前目录的完全路径。
```py
import os
os.getcwd() # 获取当前路径
os.chdir("/") # 切换到 / 路径
os.getcwd() # 再次获取当前路径
```

### 移动文件

如果想把文件 A 从文件夹 Da 移动到 Db，可以这么做
```py
import os
os.rename("Da/A", "Db/A")
```

你可能会奇怪，为什么移动文件用的是重命名的方法？理由是，对于操作系统来说，一切都是路径。

举个例子：   
假设文件A的完全路径为:   
/home/mory/documents/doc.md   
重命名为mydoc.md，路径变为   
/home/mory/documents/mydoc.md   
移动到其他文件，比如说   
/home/mory/mydoc.md   

对于操作系统说来，不管你是改变了文件名，还是存放文件的目录，实质上都只是文件的路径改变了而已，所以移动操作只是一种重命名目录的操作罢了。


### 删除文件
```py
import os
os.remove("myfile")
```

### 删除文件夹
```py
import os
os.removedirs('my_directory')
```

### 调用外部系统

os 模块还可以直接调用外部操作系统的终端命令。我们刚刚演示的那些，基本上可以用以下的操作代替。注意，不同平台提供的命令不一致，以下命令基于 Linux 系统。
```py
import os
os.system("mkdir docs")
os.system("mv Da/A Db/A")
os.system("cd /")
os.system("rm myfile")
os.system("rmdir mydirectory")
```
