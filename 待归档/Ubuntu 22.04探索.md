## 关于安装Ubuntu和分区方案
首选要注意我的固态从MBR转成了GPT格式，因为GPT比较快，所以我安装系统都要采取uefi方式引导。
同理安装windows系统是烧制镜像也是选gpt格式，不建议用ultraISO，它只能刻录MBR格式的，建议都用rufus最方便，只需要记得在用rufus的时候选GPT即可，另外注意电脑的BIOS里最好把传统的Lagacy的boot方式禁用（其实就是MBR，淘汰了最好不用）
用rufus烧制镜像的时候要注意选GPT选项而不是MBR。另外在安装Ubuntu的时候，选择这样的分区方案：
1024Mb(1G) 逻辑分区 EFI
8192Mb(8G) 主分区 Swap
30720Mb(30G) 逻辑分区 /
剩下的大部分空间 逻辑分区 /Home

安装好ubuntu22.04之后，发现系统自带的snap-store无法删除系统自带的麻将、扫雷等游戏，所以用apt去卸载
先利用apt list | grep 游戏名
找到gnome-游戏名 开头，且后面带有jammy和automatic的真正的包名
然后sudo apt-get remove 该游戏包名即可

---

## 关于ubuntu自己下载的tar软件包如何添加图标到启动器
将软件包文件夹移动到/opt下，举个例子：
例如/opt/zotero
然后仿照这个脚本编写一个脚本，放到该zotero文件夹里面，然后运行该脚本`sudo ./set_launcher_icon`
```shell
#!/bin/bash -e

#
# Run this to update the launcher file with the current path to the application icon
#

APPDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -w "$APPDIR"/zotero.desktop ]; then
sed -i -e "s@^Icon=.*@Icon=$APPDIR/chrome/icons/default/default256.png@" "$APPDIR"/zotero.desktop
else
echo "$APPDIR"/zotero.desktop is not writable
exit 1
fi
```
然后运行命令即可：
```shell
ln -s /opt/zotero/zotero.desktop ~/.local/share/applications/zotero.desktop
```
> 参考：[https://www.zotero.org/support/installation](https://www.zotero.org/support/installation)

---
## 常用命令
安装.snap结尾的包：sudo dpkg -i snap-pkg-name.snap --dangerous
安装.deb结尾的包：sudo dpkg -i deb-name.deb
创建链接（快捷方式）：ln -s 链接文件路径 存储位置

---

## 安装英伟达显卡驱动
首先禁用nouveau：执行
```shell
sudo vi /etc/modprobe.d/blacklist-nouveau.conf
```
在文件里写入：
```shell
blacklist nouveau
options nouveau modeset=0
```
保存后执行，更新使禁用nouveau生效:
```shell
sudo update-initramfs -u
```
重启系统（一定要重启），然后验证：
```shell
lsmod | grep nouveau
```
没有信息显示，说明nouveau已被禁用，接下来可以安装nvidia驱动。
重启，在启动ubuntu的时候grub界面按e，找到quiet splash $vt_handoff，删除$vt_handoff之后写上nomodeset然后按f10（这一步是为了禁用ubuntu自带的nvidia x server驱动）
进入系统后，按alt ctrl f3进入全黑没有图形界面的tty（按alt ctrl f1会返回图形界面），然后正常填写用户名和密码，安装gcc和make
运行英伟达.run结尾的驱动（运行run中途有可能跳转到图形界面，需要我们手动按alt ctrl f3回来，保证tty3里的.run正常跑完）
安装完成后重启，在终端运行nvidia-smi命令查看是否安装成功
> 参考[https://blog.csdn.net/qq_37424778/article/details/123380322](https://blog.csdn.net/qq_37424778/article/details/123380322)

---
## Ubuntu 22.04不支持AppImage软件包 
AppImage用到了FUSE（Filesystem in Userspace）技术，它可以允许非root用户挂载文件系统，但是默认情况下，Ubuntu 22.04将不再附带libfuse2包。AppImage发行版（更一般地说，所有现有AppImage发行版）的构建都期望得到libfuse2支持。这意味着默认情况下AppImage不会在Ubuntu 22.04上运行。直接双击运行会报错：
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/AppImage%E5%9C%A8Ubuntu22.04%E4%B8%8A%E8%BF%90%E8%A1%8C%E6%8A%A5%E9%94%99.png)

需要参考[https://github.com/AppImage/AppImageKit/wiki/FUSE](https://github.com/AppImage/AppImageKit/wiki/FUSE) 安装FUSE，需要注意的是该文档提了一个warning：在Ubuntu22.04装fuse可能会破坏系统（我觉得他的点在非root用户挂载文件系统可能会误操作），参考[https://itsfoss.com/cant-run-appimage-ubuntu/#comments](https://itsfoss.com/cant-run-appimage-ubuntu/#comments)用sudo apt install libfuse2安装libfuse2即可

---

## Ubuntu 22.04 截图快捷键
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Ubuntu22.04%E6%88%AA%E5%9B%BE%E5%BF%AB%E6%8D%B7%E9%94%AE.png)

---
## Ubuntu Qv2ray
ubuntu下的qv2ray本身不支持trojan协议，需要通过插件来实现：
下载[https://github.com/Qv2ray/QvPlugin-Trojan](https://github.com/Qv2ray/QvPlugin-Trojan)的[QvPlugin-Trojan.v3.0.0.linux-x64.so](https://github.com/Qv2ray/QvPlugin-Trojan/releases/download/v3.0.0/QvPlugin-Trojan.v3.0.0.linux-x64.so)，将其放到/Home/.config/qv2ray/plugins文件目录下，重启qv2ray，重启系统即可

---
## Obsidian on Ubuntu 22.04
完成Qv2ray代理配置后，安装Obsidian的deb版本（Obsidian需要代理才能打开插件商店）
参考[https://gitee.com/help/articles/4181](https://gitee.com/help/articles/4181)配置好本地的git
克隆项目用Obsidian打开即可
关于图床：参考[https://picgo.github.io/PicGo-Doc/en/guide/](https://picgo.github.io/PicGo-Doc/en/guide/) 安装PicGo的AppImage，另外PicGo需要依赖包xclip，用`sudo apt install xclip`安装xclip，然后参考[[图床搭建及配置]]对PicGo进行配置，完成后即可实现：在Ubuntu 22.04下与windows一样的“粘贴截图到笔记并上传到图床“的体验。

---
## Ubuntu tty用法
w -s查看所有激活的tty的情况
ps -t tty4查看目标tty中CMD的PID号，再sudo kill pid

---
## ZSH
安装zsh比较简单，直接用：`sudo apt install zsh`即可，然后切换默认shell为zsh：`sudo chsh -s $(which zsh)`，然后log out重新登录ubuntu系统用户即可。
>参考zsh官方文档：https://github.com/ohmyzsh/ohmyzsh/wiki

开始时zsh下命令`ll`是不能用的，需要`vim ~/.zshrc然后添加一行：
```shell
alias ll='ls -alF'
```
保存退出然后source一下：`source ~/.zshrc

>其他插件参考ZSH官方文档：https://github.com/ohmyzsh/ohmyzsh/wiki/Plugins-Overview

---
### ZSH的FS Jumping工具
FS Jumping是指文件系统的跳转工具
官方文档给了7个：**autojump**, fasd, jump, pj, wd, z, zoxide（我选autojump）
>参考ZSH官方文档：https://github.com/ohmyzsh/ohmyzsh/wiki/Plugins-Overview#fs-jumping

### ZSH主题
#### spaceship
spaceship主题很强大，git repo有17k的star，（不过我怕过重了没有选用）
>参考：https://github.com/spaceship-prompt/spaceship-prompt

#### jovial（我选择用的这个）
jovial采用苹果风格，响应式设计（responsive-design），显示信息比较丰富，集成了几个主流的插件。
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/jovial.png)

##### 集成插件：
- **[jovial](https://github.com/zthxxx/jovial/blob/master/jovial.plugin.zsh)**: jovial plugin defined some utils functions and alias, you can see in [jovial.plugin.zsh](https://github.com/zthxxx/jovial/blob/master/jovial.plugin.zsh)
-   **[git](https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/git)**: some short alias for commonly used command
-   **[autojump](https://github.com/wting/autojump)**: make you can use `j <keyword>` to jump to the full path folder
-   **[bgnotify](https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/bgnotify)**: background notifications for long running commands
-   **[zsh-history-enquirer](https://github.com/zthxxx/zsh-history-enquirer)**: widget for history searching, enhance `Ctrl+R`
-   **[zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)**: shell auto-completion
-   **[zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting)**: user input syntax highlighting

##### 安装和配置
安装时需要注意先安装nvm（参考[https://github.com/nvm-sh/nvm#installing-and-updating]），然后source一下.zshrc
然后按照[官网](https://github.com/zthxxx/jovial#glance)执行：
```shell
curl -sSL https://github.com/zthxxx/jovial/raw/master/installer.sh | sudo -E bash -s $USER
```
然后更新一下npm：`npm install -g npm@8.12.1`
然后再跑一遍
```shell
curl -sSL https://github.com/zthxxx/jovial/raw/master/installer.sh | sudo -E bash -s $USER
```
完成。
然后参考[教程](https://github.com/cstrap/monaco-font)安装苹果字体monaco（我是在[这个链接]([https://github.com/todylu/monaco.ttf/blob/master/monaco.ttf?raw=true](https://github.com/todylu/monaco.ttf/blob/master/monaco.ttf?raw=true))下载的monaco.ttf，而且安装ttf后需要Update一下font cache：用`sudo fc-cache -f -v`），然后调整终端背景色号*\#282a36*，安装gnome-tweaks后修改字体即可：
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%BB%88%E7%AB%AF%E9%85%8D%E7%BD%AE%E8%8B%B9%E6%9E%9C%E5%AD%97%E4%BD%93monaco.png)
>自定义配置参考：https://github.com/zthxxx/jovial#customization
>官方项目地址：https://github.com/zthxxx/jovial


#### zeta
zeta比较简洁，该显示的信息基本都有
https://github.com/skylerlee/zeta-zsh-theme


小技巧：
在任何界面按alt + space可以调出截图、移动、缩放等功能目录
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/alt%E5%92%8C%E7%A9%BA%E6%A0%BC%E8%B0%83%E5%87%BA%E6%88%AA%E5%9B%BE%E7%A7%BB%E5%8A%A8%E7%BC%A9%E6%94%BE%E7%AD%89%E5%8A%9F%E8%83%BD%E7%9B%AE%E5%BD%95.png)
鼠标放在浏览器标签栏滚动滚轮可以切换标签页
