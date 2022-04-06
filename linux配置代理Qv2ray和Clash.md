# Qv2ray: 
#tool #proxy #linux
一款linux上很好用的图形话界面科学上网工具-Qv2ray。
🌟 Linux/Windows/macOS 跨平台 v2ray GUI 🔨 使用 c++ 编写，支持订阅，扫描二维码，支持自定义路由编辑 🌟。使用 Qt 框架的跨平台 v2ray 客户端。支持 Windows, Linux, macOS

来自 https://www.zsxcool.com/7137.html

[Ubuntu 20.04系统 使用 V2ray GUI 界面软件配置翻墙上网](https://www.youtube.com/watch?reload=9&v=fwvUjQJHmgk&app=desktop) 

![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/1.png)
 

https://www.youtube.com/watch?reload=9&v=fwvUjQJHmgk&app=desktop 

`$ gnome-session-properties`打开开机自启动应用程序
 
记得可以改语言成中文，然后在首选项里面把固定连接对号打上（不然每次开机启动后连接列表里都是空的，都得重新扫码新建连接）
 
再有就是一旦proxy失败导致Ubuntu中整个连不上网，直接在系统的设置里面找到proxy关掉就能连上了（如下图改成Disabled即可） 
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/2.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/3.png)

下面是我好使的Qv2ray配置界面 
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/4.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/5.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/6.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/7.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/8.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/9.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/10.png)

主界面： 
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/11.png)


右上角最小化界面 

![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/12.png)

---

# Clash
该Clash项目在https://github.com/Dreamacro/clash进行维护
安装文档在https://github.com/Dreamacro/clash/wiki
安装步骤如下：
保证本地Golang版本大于等于1.16
成功后用go env命令查看`$GOPATH`: 
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/13.png)

在终端运行命令：go install github.com/Dreamacro/clash@latest
这一步会产生两组东西：
1. `$GOPATH`的bin目录下会有用go本地编译好的clash执行程序
2. ~/.config/文件夹下会生成一个文件夹clash，其中包含两个生成的文件：config.yaml和Country.mmdb，前者是空的代理信息配置文件，我们用机场订阅链接对应的yaml文件进行替换，名称一定要是config.yaml（如果不会找，那就在windows无脑配置好clash然后复制他的config.yaml过来），后者是大家都一样的地理信息表，如果生成失败了，同样可以在windows下安装配置clash然后复制其生成好的Country.mmdb过来。
如图（注意这里地址栏的Home相当于`$Home`，我这里也就是/home/zjp）
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/14.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/15.png)

运行命令clash，默认的clash命令调用的是 ~/.config/clash/ 下的配置文件（config.yaml和Country.mmdb），如下图（订阅节点用的是我的monocloud）： 
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/16.png)

利用curl命令测试是否fq成功（用ping命令失败的原因是：ping基于ICMP协议，在网络层，而socks代理在传输层，无法代理比它低层的协议，所以应当采用curl google.com），如下图： 
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/17.png)

clash远程管理界面：
clash连接成功后，用浏览器访问clash.razord.top，即可作为linux下的GUI界面来管理clash代理，如图： 
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/18.png)


如果是多组订阅链接的话，可以通过这样的方式进行管理和连接：
例如我现在也有Youtube工具大师的免费机场订阅节点，可以新建一个文件夹用来存放这个订阅链接的配置文件，文件夹下的目录结构如图（其中config.yaml同样是从订阅链接url获取的，Country.mmdb用的就是默认生成的）： 
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/19.png)

这次运行的命令需要用-d选项指定代理的配置文件（config.yaml和Country.mmdb）的位置，如clash -d ~/.config/clash/gongjvdashi，连接效果如图： 
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/20.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/21.png)

如果想开机自启动的话可以借助`gnome-session-properties`这个命令。
如果想通过命令切换代理可以参考：https://clash.gitbook.io/doc/restful-api/proxies

参考资料：
<https://www.cnblogs.com/sundp/p/13541541.html>
<https://zhuanlan.zhihu.com/p/369344633>
<https://www.cnblogs.com/sundp/p/13541541.html>
<https://github.com/yuanlam/Clash-Linux>
