# Python base
## 1. 补齐整数
```python
print('{:03d}'.format(i))
```
这里`{:03d}`的意思是三位整数用0补齐

## 2.排序
```python
import os
sorted(glob.glob('*.png'), key=os.path.getmtime)# 按照修改时间排序
sorted(glob.glob('*.png'), key=os.path.getsize)# 按照文件大小排序
```


# Pandas
## 1. 转excel
```python
df.to_excel('{:03d}.xlsx'.format(i))
```

---
## 2. 合并两个list为二维数组
```python
df=pd.DataFrame(np.array([x,y].T, columns=['x','y'])
```
这里的.T是转置的意思，上述代码效果如：


___
## 文件操作
1. 列举当前文件夹下所有以.pdf结尾的文件名
```python
# 采用glob.glob()函数
pdflist=glob.glob("*\\*.pdf", root_dir=".\\Zotero\\storage\\", recursive=True)
```

2.列举当前文件夹下所有文件（夹）名
采用os.walk()函数
```python
# 文件
_, _, filenames = next(os.walk(path), (None, None, []))

#文件夹
_, dirnames, _ = next(os.walk(path), (None, [], None))

# 文件和文件夹
_, dirnames, filenames = next(os.walk(path), (None, [], []))
```
采用glob()函数
