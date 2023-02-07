cudaMemcpy已经包含了同步过程，就不需要显式调用sycn了

用空间换时间
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%BA%8C%E7%BB%B4grid%E5%92%8C%E4%BA%8C%E7%BB%B4block.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%AE%97%E4%B8%80%E7%BB%B4%E5%9D%90%E6%A0%87.png)

cudaMallocHost与malloc不同的一点是：避免内存虚拟化技术从内存移到disk


## 错误检测和事件

可以查看Cuda error的四个函数：

```C++
__host__​__device__​const char*     cudaGetErrorName ( cudaError_t error )
Returns the string representation of an error code enum name.  

__host__​__device__​const char*     cudaGetErrorString ( cudaError_t error )
Returns the description string for an error code.  

__host__​__device__​cudaError_t     cudaGetLastError ( void )
Returns the last error from a runtime call.  

__host__​__device__​cudaError_t     cudaPeekAtLastError ( void )
Returns the last error from a runtime call.  
```


![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%B8%80%E4%B8%AACUDA%E8%BF%90%E8%A1%8C%E6%97%B6%E6%A3%80%E6%B5%8B%E5%87%BD%E6%95%B0%E5%AE%9E%E4%BE%8B.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%B8%80%E4%B8%AACUDA%E8%BF%90%E8%A1%8C%E6%97%B6%E6%A3%80%E6%B5%8B%E5%87%BD%E6%95%B0%E5%AE%9E%E4%BE%8B2.png)


![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Check%E5%87%BD%E6%95%B02.png)


等待事件完成，设立flag：

```C++
cudaError_t cudaEventSynchronize(cudaEvent_t event);//阻塞（事件完成才记录）
cudaError_t cudaEventQuery(cudaEvent_t event);//非阻塞（事件没完成也会记录）
```


[(含代码)利用GPU版的Numpy---cuNumeric加速Python数值计算\_扫地的小何尚的博客-CSDN博客](https://blog.csdn.net/kunhe0512/article/details/128908418)


## 下午实验前置知识

### CUDA存储单元
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%A4%9A%E7%A7%8DCUDA%E5%AD%98%E5%82%A8%E5%8D%95%E5%85%83.png)

右边单向箭头表示是可读的，双向箭头表示是可读可写的（这里考试会有小陷阱）

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%A4%9A%E7%A7%8DCUDA%E5%AD%98%E5%82%A8%E5%8D%95%E5%85%83%E8%AF%A6%E8%A7%A3.png)

Local mem属于是on board mem（而不是on chip），但却是GPU线程私有的，空间比较大，而读写比较慢。（这一点需要注意）

#### Register
寄存器最快，我们要尽量让更多的block主流在SM中，以增加Occupancy，省着点使用寄存器资源。

#### Shared Memory
- 比Register稍慢一点
- On-chip
- 拥有高的多带宽和低很多的延迟
- 同一个Block中的线程共享一块Shared Memory
- 用__syncthreads()进行同步
- 比较小，要节省使用，否则会限制活动warp的数量。
- SMem被分成32个逻辑块（banks）
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Smem%20bank%20conflict.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E9%81%BF%E5%85%8Dbank%20conflict%EF%BC%88%E6%97%A0bank%20conflict%E7%9A%84%E6%83%85%E5%86%B5%EF%BC%89.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E9%81%BF%E5%85%8Dbank%20conflict%EF%BC%88%E6%9C%89bank%20conflict%E7%9A%84%E6%83%85%E5%86%B5%EF%BC%89.png)

或者用Memory Padding的方式避免bank conflict
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%94%A8Memory%20Padding%E7%9A%84%E6%96%B9%E5%BC%8F%E9%81%BF%E5%85%8Dbank%20conflict.png)
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/2%E7%94%A8Memory%20Padding%E7%9A%84%E6%96%B9%E5%BC%8F%E9%81%BF%E5%85%8Dbank%20conflict.png)
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%94%A8Memory%20Padding%E7%9A%84%E6%96%B9%E5%BC%8F%E9%81%BF%E5%85%8Dbank%20conflict%E7%9A%84%E4%BB%A3%E7%A0%81.png)

连续的数据存储，第一行的x就变成了

#### Local Memory
Reg不够的时候就会用Local Mem来替代
- On-board
- --ptxas-options=-v

#### Constant Memory
- 全局（对所有kernel可见）
- kernel从CM只能读而不能写，因此初始化必须在host端使用cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src,)
- 当一个warp中所有的thread都从同一个Memory地址读取数据时，constant Memory表现会非常好，会触发广播机制。

常量内存举例-光线追踪
这里的4表示每个点的r，g，b和透明度信息需要四个存储单元，存储在一个一维数组中。


#### Texture Memory
- 驻留在device Mem中，属于On-board，并使用一个只读cache。

#### Global Memory
- 空间最大，latency最高，是GPU中最基础的memory。
- On-board，驻留在Device memory中
- memory transction 对齐，合并访存。

按列读快，如下左边是按行读，右边是按照列读（例如thread0用四次迭代分别读取B00，B10，B20，B30）。
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%90%88%E5%B9%B6%E8%AE%BF%E5%AD%98.png)


