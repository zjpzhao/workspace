#learning/Git
# Git
## 嵌套git repo的push
当我外层项目目录是一个git repo，目录需要用到另外一个git repo的内容时，git add时会出现 "fatal: in unpopulated submodule XXX" 错误。
解决方法是：
```shell
git rm -rf --cached [内层git仓库的项目名]
```
然后正常push三联即可

---

## 撤销或修改Commit
执行完commit后想撤回commit，执行：
```shell
git reset --soft HEAD^
```
成功的撤销了上次commit，之前做的修改仍然保留。这里我的理解是：HEAD^的意思是上一个版本，也可以写成`HEAD~1`，如果你进行了2次commit，想都撤回，可以使用`HEAD~2`
至于这几个参数：
- --mixed ：不删除工作空间改动代码，撤销commit，并且撤销git add . 操作。这个为默认参数，git reset --mixed HEAD^ 和 git reset HEAD^ 效果是一样的。
- --soft  ：不删除工作空间改动代码，撤销commit，不撤销git add . 
- --hard:：删除工作空间改动代码，撤销commit和git add . 注意完成这个操作后，就恢复到了上一次的commit状态。

需要修改commit的message（注释）执行：
```shell
git commit --amend
```
此时会进入默认vim编辑器，修改注释完毕后保存就好了。

---

# Github

## Github高级搜索
按`s`键直接聚焦到搜索框
springboot vue stars:>1000 pushed:>2022-05-02 language:Java
也可以在搜索框下方点击advanced search，用可视化表单也能实现高级搜索
其他高级功能可以参考[https://docs.github.com/en/search-github]

## Github文件查看技巧
1. 在repo里按`t`键进行文件检索
2. 点进文件后按`l`键可以快速跳转到某一行
3. 点击行号就可以：
- 选择Copy line就可以快速复制这行代码
- 选择Copy permalink生成永久链接
4. 按`b`键可以快速查看该文件的改动记录
5. 打开命令面板：`ctl + k`
6. 在repo按`.`键：将该仓库在网页版本vscode打开
7. 在线运行github仓库：在网址url中github前面添加gitpod.io/#/，gitpod自动识别项目的语言类型并安装依赖包，可以在这里随心所欲，也可以一键构建镜像
8. 项目推送：打开Github Explore：
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/open github explore.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/open%20github%20explore.png)
在explore页面可以查看根据你的浏览记录智能推送给你的repo，或者点击Get email updates获取邮件更新
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Github%20Explore.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/GitHub%20get%20email%20updates.png)


9. 更多github快捷键参考：[https://docs.github.com/en/get-started/using-github/keyboard-shortcuts]


---

# Gitee

---

## 强制同步Github仓库到Gitee同名仓库
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%BC%BA%E5%88%B6%E5%90%8C%E6%AD%A5Github%E4%BB%93%E5%BA%93%E5%88%B0Gitee%E5%90%8C%E5%90%8D%E4%BB%93%E5%BA%93.png)

---

## Gitee与Github仓库双向同步
参考<https://gitee.com/help/articles/4336>和<https://blog.csdn.net/yi_rui_jie/article/details/111357163>

---

## Gitee repo init
### A simple command-line tutorial:
#### Git global settings
```bash
git config --global user.name "zjpzhao"
git config --global user.email "1284610325@qq.com"
```

#### Create git repository
```bash
mkdir my-book-notes
cd my-book-notes
git init 
touch README.md
git add README.md
git commit -m "first commit"
git remote add origin https://gitee.com/simonhouhou/my-book-notes.git
git push -u origin "master"
```

#### Existing local folder
现在gitee上新建一个repo
```bash
cd existing_git_repo
git init
git add .
git commit -m "first commit"
git remote add origin https://gitee.com/simonhouhou/my-book-notes.git
git push -u origin "master"
```
