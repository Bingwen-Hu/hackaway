# install package into .local 
make sure there is a .pip directory exist!

if not
```bash
mkdir .pip
```

then 
```bash
cd .pip & touch pip.conf
```

and add the following content 
```conf
[install]
install-option=--prefix=~/.local
```

# setup pip source
+ create a ~/.pip/pip.conf
+ put following code in `pip.conf`

```conf
[global]
index-url=https://pypi.douban.com/simple
```

# using another mirror

```bash
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ opencv_python
```

