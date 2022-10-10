---
id: e8dac33a-edb1-4561-b721-ff2f0ff136fb
---
#doc #tool/notes
# 插件
### Obsidian Git
因为我的Obsidian笔记项目workspace放在了Onedrive的本地文件夹下，通过Onedrive进行同步，另外通过git进行版本控制和同步。需要注意的一点是：修改完笔记文件或者执行pull，commit和push之后都会激发Onedrive的同步状态（右下角onedrive转圈），我们一定要在**Onedrive完成同步状态**后再进行commit或者push操作，否则会在.git生成一个FETCH开头的文件并提示副本冲突，而且之后会不定期总是弹出来。
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/FETCH%E5%86%B2%E7%AA%81%E5%88%9B%E5%BB%BA%E5%89%AF%E6%9C%AC.png)

如果实在不好使就把笔记推上去，删除本地项目文件夹然后重新clone下来，这样做的原因是本地项目文件夹下的.git不会同步到gitee上，因为被gitignore给忽略了，所以冲突就不在了。
所以**强烈建议**将Onedrive图标拽出来到右下角，先关注右下角Onedrive的同步状态，每次都确保是在**Onedrive同步完成的状态**下进行pull或commit或push。

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
---
## 快速添加日期
采用obsidian插件Natural Language Dates
用法：@日期
e.g. @today回车得到2022-07-12

---
