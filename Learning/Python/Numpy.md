#learning/python/numpy

## 1. 广播对齐机制
当运算中的 2 个数组的形状不同时，numpy 将自动触发如图的广播机制：
![](https://www.runoob.com/wp-content/uploads/2018/10/image0020619.gif)
```python
import numpy as np 
a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([0,1,2])
print(a + b)

# 输出：
# [[ 0  1  2]
#  [10 11 12]
#  [20 21 22]
#  [30 31 32]]
```
如上：4x3 的二维数组与长为 3 的一维数组相加，等效于把数组 b 在二维上重复 4 次再运算。
```python
import numpy as np
a = np.array([[ 0, 0, 0], [10,10,10], [20,20,20], [30,30,30]])
b = np.array([1,2,3])
bb = np.tile(b, (4, 1))# 重复 b 的各个维度
print(a + bb)
```
>参考：https://www.runoob.com/numpy/numpy-broadcast.html

2. 按行累加
```python
sumvector=np.sum(arr,axis=1)
```


3. 数组拼接
```python
# 垂直拼接（即axis=0）

# 方法1
arr1=np.append(arr1,arr2,0)
# 方法2
arr1=np.vstack((arr1,arr2))
```

4. 求矩阵的逆
```python
def getInversion(M):
    inverM=lg.inv(M)
    # print(np.around(M*inverM,decimals=3))
    return inverM
```

5. 求对角化矩阵
```python
def getDiag(M):
    p = np.linalg.eig(M)[1]
    DiagM = np.dot(np.dot(np.linalg.inv(p),M),p)
    DiagM = np.around(DiagM,decimals=5)
    return DiagM
```

6. 数组升维/降维
```python
# e.g.将数组的形状从(10,)变成(10,1)
arr=arr.reshape(-1,1)
# e.g.将数组的形状从(10,1)变成(10,)
arr=np.squeeze(arr)
```

7. 数组除法考虑除零
```python
Aarray=np.divide(Z, X, out=np.zeros_like(Z), where=X!=0)
```
8. 放在对角线
```python
np.diag()
```