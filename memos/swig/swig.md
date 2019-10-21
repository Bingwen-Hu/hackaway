# 使用swig封装C++代码

### 环境
+ Linux (ubuntu 1804)
+ swig 3.0
+ python3.6
+ 目的：封装C++代码，供python调用的一个示例

### 项目

#### 1. 准备C++代码
我们的小项目是用 C++ 定义一个打印向量的函数，从 Python 端进行调用。目录
```sh
├── test-vec.cpp
├── vec.cpp
├── vec.h
```
先在 vec.h 中定义接口
```c++
// vec.h
#include <vector>

class My {
    public:
        void print_vector(std::vector<int> const &v);
};
```
然后是 vec.cpp 中实现
```c++
#include "vec.h"
#include <vector>
#include <iostream>

void My::print_vector(std::vector<int> const &v){
    for (int i : v) {
        std::cout << i << std::endl;
    }
}
```

我们写一个测试 test-vec.cpp
```c++
#include "vec.h"
#include <vector>

using std::vector;

int main(){
    My m = My();
    vector<int> v{};
    v.push_back(1);
    v.push_back(2);
    m.print_vector(v);
}
```
编译一下
```sh
g++ test-vec.cpp vec.cpp vec.h -o test
```
运行
```sh
./test
```

#### 2. 使用swig封装
使用swig时，需要定义一个接口(interface)文件，我们的比较简单，如下

```interface
%module vec // module name

// parse std header for vector
%include <std_vector.i>

// using template, for vector<int>
using std::vector;
namespace std{
    %template(vectori) vector<int>; 
}

// include header 
%{
    #include "vec.h" 
%}

// parse header for print_vector
%include "vec.h" 
```
第一行定义了模块名，我们在Python导入的时候就是用这个。第二行解析标准库的接口文件，swig提供了标准库的接口文件，所以我们可以直接使用。接下来定义了向量模板 vector<int>。因为我们使用了这个模板。然后是
```c++
%{
    #include "vec.h"
%}
```
所有的自定义的头文件必须放在这里面。最后是解析自定义的头文件。万事俱备，开始编译。

```sh
swig -c++ -python vec.i
```
解析接口文件，生成 vec_wrap.cxx 和 vec.py 文件。-c++ 表示启用 c++ 预处理器，-python 表示编译成 Python。
```sh
g++ -fpic -c vec.h vec.cpp vec_wrap.cxx  -I/usr/include/python3.6
```
这一步编译源文件成目标(.o)文件。因为我们的目标平台是 python3.6，所以包含了 python3.6 的头文件。
```sh
g++ -shared vec.o vec_wrap.o -o _vec.so -lstdc++
```
这一步生成一个动态链接库 _vec.so 。vec.py 是其上层接口，使用的时候应该使用 vec.py，而不是直接使用 _vec.so 。

好了，现在我们开始在 python 中使用这个库吧！

#### 3. 测试
```py
import vec

my = vec.My()

# use native list
lst = [1, 2, 3, 4]
my.print_vector(lst)

# use C++ vector<int>
v = vec.vectori([5, 6, 7, 8])
my.print_vector(v)
```
我们可以给 print_vector 传入 Python 列表！当然，因为我们导入了 vector<int>，也可以使用它！

#### 4. 总结
swig 封装代码还是特别简洁的，我们的全部源代码目录如下：
```sh
├── test-vec.cpp
├── test-vec.py
├── vec.cpp
├── vec.h
└── vec.i
```
我们只是编写了简单的接口文件，就可以把我们的 c++ 代码导入 Python 了！

### Reference & Thanks
+ http://swig.org/tutorial.html

