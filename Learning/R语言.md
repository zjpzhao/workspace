## 查看所有变量
```r
ls()
```

## 查看变量类型
```R
mode(a) 
class(a)
typeof(a) #展现数据的细节上，mode<class<typeof

```

## 像excel一样查看数据

```r
View(data)
```

## 查看字段的数据类型

```r
str(data)
```

## 查看字段维度

```r
dim(data)
length(data)#一维数据查看数据长度
```

## 查看数据字段名，还可以显示列表对象各个元素的名字

```r
names(data)
```

## 查看数据大致的分布情况

```r
summary(data)
```

## 查看前/后6条数据
```R
head(data)
tail(data)
```

## 去除列表固定的行
```R
wiot[ ,-c(1,2,3,4,5) ]
```

