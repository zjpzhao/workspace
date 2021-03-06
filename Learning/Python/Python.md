---
id: 6cdd0919-5e72-4495-9f39-8a0fa6f8fb85
---
# Python base
#doc #learning/python
## 1. 补齐整数
```python
print('{:03d}'.format(i))
```
这里`{:03d}`的意思是三位整数用0补齐

## 2.文件排序
```python
import os
sorted(glob.glob('*.png'), key=os.path.getmtime)# 按照修改时间排序
sorted(glob.glob('*.png'), key=os.path.getsize)# 按照文件大小排序
```

## 3. 多线程
参考[docs.python.org](https://docs.python.org/3/library/multiprocessing.html)
```python
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))

# 返回[1, 4, 9]
```

## 4. 多线程+偏函数（参数打包）
pool.map只支持单参数，参考[python.omics.wiki](https://python.omics.wiki/multiprocessing_map/multiprocessing_partial_function_multiple_arguments)采用如下方式解决
### 方法一：采用偏函数
```python
#方法一：利用partial()进行参数打包
# Example: multiply all numbers in a list by 10
import multiprocessing
from functools import partial
data_list = [1, 2, 3, 4]

def prod_xy(x,y):
    return x * y

def parallel_runs(data_list):
    pool = multiprocessing.Pool(processes=4)
    prod_x=partial(prod_xy, y=10) # prod_x has only one argument x (y is fixed to 10)
    result_list = pool.map(prod_x, data_list)
    print(result_list)

if __name__ == '__main__':
    parallel_runs(data_list)
    #返回[10, 20, 30, 40]
```
我自己写的一个更复杂的例子是[[Code_对神经网络模型参数进行多线程单比特位故障注入]]

### 方法二：外套列表
多个参数的列表可以通过pool.map传递给子任务函数 （该函数需要接受列表作为单个参数）
```python
# 本例是给定多个列表，求每个列表的乘积（production）
import multiprocessing
import numpy as np

data_pairs = [ [3,5], [4,3], [7,3], [1,6] ]
def myfunc(p):
    product_of_list = np.prod(p)
    return product_of_list

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    result_list = pool.map(myfunc, data_pairs)
    print(result_list)
    #返回[15, 12, 21, 6]
```

---

## 5. 除法取商和余数
```python
[q, r] = divmod(a, b)
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

## 3. 读取.xlsb文件
```python
# 需要安装pyxlsb（pip install pyxlsb）
df = pd.read_excel(XLSB_Path, engine='pyxlsb')
```
或者不用Pandas
```python
from pyxlsb import open_workbook as xlsb
# 获取xslb 的第一张表
with xlsb('1.xlsb').get_sheet(1) as data:
    # 按行获取数据
    for row in data.rows():
        # 读取每行单元格内位置 和 数据
        for ro in row:
            # 得到 list 的数据
            print(ro.v)
```

---
# 文件操作
## 1. 列举当前文件夹下所有以.pdf结尾的文件名
```python
# 采用glob.glob()函数
pdflist=glob.glob("*\\*.pdf", root_dir=".\\Zotero\\storage\\", recursive=True)
```

## 2.列举当前文件夹下所有文件（夹）名
采用os.walk()函数
```python
path="."
# 文件
_, _, filenames = next(os.walk(path), (None, None, []))

#文件夹
_, dirnames, _ = next(os.walk(path), (None, [], None))

# 文件和文件夹
_, dirnames, filenames = next(os.walk(path), (None, [], []))
```
采用glob()函数
```python
list = glob.glob('.\**', recursive=True)
```
参考[link](https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/)

---

## 3. 将当前文件夹下所有pdf文件放到对应的子文件夹中
文件结构形如：
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/tree%E8%BE%93%E5%87%BA%E6%96%87%E4%BB%B6%E7%BB%93%E6%9E%84.png)
```python
import os
import glob
import shutil
# 采用glob.glob()函数
# pdflist=glob.glob(".\\**\\*.pdf",recursive=True)
_, dirnames, _ = next(os.walk("."), (None, [], None))
# print(dirnames)
for subfolder in dirnames:
    # print(subfolder)
    pdflist=glob.glob(".\\"+subfolder+"\\**\\*.pdf",recursive=True)
    # print(pdflist)
    for item in pdflist:
        name=item.split("\\")[-1]
        shutil.move(item,".\\"+subfolder+"\\"+name)```


---

## 4. excel读操作
```python
import xlrd
checked_students=set()
data = xlrd.open_workbook("./本科生每日打卡数据表.xls")
table = data.sheet_by_name("Sheet0")#选择sheet
for r in range(2, table.nrows):
	checked_students.add(table.cell(r, 2).value)
```


# 其他
## 1. 微信给好友发送消息脚本
```python
import itchat
# itchat不支持python3.10和3.9，该脚本在3.7.13下跑通
name="聂晨飞"
msg = "——飞哥今天是红的还是绿的\n——今天不喝"

itchat.auto_login(hotReload=True)   #登录
info = itchat.search_friends(name=name) #查找朋友

if len(info) == 0:
    print("No such friend!")
else:
    info = info[0]
    if info["RemarkName"] != name:
        print("Name incorrect")
    else: 
        print(msg)
        cmd = input("确定发送?(y/n): ")
        if cmd != "y" and cmd != "Y":
            print("Terminated")
        else:
            print("Sended to {}".format(name))
            itchat.send(msg, info["UserName"]) #用来真正发送消息
# itchat.logout()
```



# 可视化
## 数据可视化视频/动图
采用celluloid库
参考<https://pypi.org/project/celluloid/>

---
