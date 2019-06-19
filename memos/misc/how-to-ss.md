# 如何配置shadowsocks翻墙

## 前置条件
一台已经配置好的shadowsocks服务器，即你拥有在墙外shadowsocks服务的帐号，端口号及密码

### Linux or Mac

1. 安装shadowsocks客户端
`pip`使用`python2`的版本
```bash
pip install shadowsocks
# 验证安装是否成功，在命令行输入sslocal或者ss-local
sslocal
# ss-local
# 打印出一系列使用帮助即为安装成功
```
2. 创建shadowsocks配置文件
使用`vim`打开，推荐配置文件放在`/etc`目录下
```bash
sudo vim /etc/shadowsocks.json
```
写入以下内容
```json
{
    "server":"113.223.121,212",  // 远程服务器ip
    "server_port":1995,  // 远程服务器端口
    "local_address": "127.0.0.1",  // 本地ip
    "local_port":1080,  // 本地代理端口
    "password":"NDRiOTM1MD",  // 远程服务器密码
    "timeout":300,  // 超时时间
    "method":"aes-256-cfb",  // 加密方式，与远程一致
    "fast_open": false
}
```

3. 启动服务
这里使用反引号(Esc键下面)将`which sslocal`括起来，是为了自动查找`sslocal`的位置
```bash
sudo `which sslocal` -c /etc/shadowsocks.json -d start
```
如果想停止服务
```bash
sudo `which sslocal` -c /etc/shadowsocks.json -d stop
```

4. chrome配置
从[SwithcyOmega](https://github.com/FelisCatus/SwitchyOmega/releases)下载[插件](https://github.com/FelisCatus/SwitchyOmega/releases/download/v2.5.20/SwitchyOmega_Chromium.crx)并解压
```bash
mv SwitchyOmega_Chromium.crx SwitchyOmega_Chromium.zip
unzip -x SwitchyOmega_Chromium.zip
```

打开谷歌浏览器，点`右上角三个点的位置`->`更多工具`->`拓展程序`，打开拓展程序页面，在拓展程序的右侧，打开`开发者模式`，这时会出面三个按钮，点击`加载已解压的拓展程序`,选择我们刚刚解压的`SwitchyOmega`的文件夹。

5. SwitchyOmega配置
从百度网盘获取配置的备份`bak`文件
> https://pan.baidu.com/s/1PnRxW4LLj1g1Mul-Ww7OKQ  spno 
在SwithyOmega左侧选择`导入/导出`栏，选择`从文件中恢复`，导入`bak`文件

6. 启动SwitchyOmega
在谷歌浏览器的右上方，点击SwitchyOmega按钮，选择`自动切换`即可

### Windows(win10)

1. 下载客户端

[下载地址](https://github.com/shadowsocks/shadowsocks-windows/releases)

2. 配置shadowsocks

双击shadowsocks.exe，初次运行会弹出配置页，将服务器ip，密码，端口好填好即可，参考以上linux的配置

3. 配置Chrome和SwitchyOmega与Linux/Mac相同

> 注：如果shadowsocks无法运行，可能是因为缺少对应版本的.NET Framework运行库，从[微软官网](https://www.microsoft.com/en-us/download/developer-tools.aspx)下载安装