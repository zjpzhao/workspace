---
id: e8dac33a-edb1-4561-b721-ff2f0ff136fb
---
#doc #tool/notes
# 插件
### Obsidian Git
因为我的Obsidian笔记项目workspace放在了Onedrive的本地文件夹下，通过Onedrive进行同步，另外通过git进行版本控制和同步。需要注意的一点是：修改完笔记文件之后一定在**Onedrive完成同步状态后**，再进行commit和pull操作，否则会在.git生成一个FETCH开头的文件并提示副本冲突。

---

# 快捷键
调出控制台：ctrl + shift + i

---

# 技巧
## 双向链接
选中关键词，在英文输入法下连按两次做方括号\[即可转化为链接
创建双向链接的方式有四种：[04:24](https://www.bilibili.com/video/BV1nR4y157kd/?spm_id_from=333.788#t=264.166756)
1. 在反向链接面板将提到的关键词点击按钮转为链接
2. 使用双中括号框住现有的关键词
3. 使用双中括号插入新的关键词
4. 在右侧outLine面板点击按钮

---

## 快速将一行转变为任务
选择一行按住ctrl按一下回车变成任务，再按一下回车完成任务

---

## 快速生成presentation
slide插件是obsidian原生内置的插件，在设置界面enable插件后，通过`start presentation`命令，即可将当前markdown文件以ppt的形式展示出来，插件根据md文件中的---分隔符对文件进行分隔，来作为每一页展示的内容（添加的图片目前无法显示）。
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%BF%AB%E9%80%9F%E7%94%9F%E6%88%90presentation.png)

---
## 将图片内嵌到文字段落中
```markdown
![[image.png|inlL|100]]  
- 这里  
- 是
- 文字
```

```markdown
![[image.png|inlR|100]]  
- 这里  
- 是
- 文字
```

---

## 块引用
```Markdown
[[文章名|块名]]
e.g. [[Obsidian用法|块引用]]
```
