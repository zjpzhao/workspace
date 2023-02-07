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


## 打卡题目和其他知识：
1. GPU上多个线程的原子操作的多个分解步骤物理上不可能并行这句话对么？
答：这句话不完全正确。GPU上多个线程可以同时进行原子操作，但是对于特定的存储单元，只有一个线程能够成功地完成原子操作。因此，多个线程同时对相同的存储单元执行原子操作时，其中的一些线程可能会遇到冲突，并且仅有一个线程的操作会被执行，其他的线程的操作将被阻塞直到它们能够完成操作为止。
2. 原子操作所使用的存储单元，必须是shared memory吗？
答：不是必须是shared memory。原子操作可以在多种存储单元中进行，包括global memory和shared memory，具体取决于操作的需求和性能要求。

3. cudaMallocHost和使用malloc的区别
回答：驱动程序跟踪用这个函数分配的虚拟内存范围，并自动加速对cudaMemcpy()等函数的调用。由于内存可以被设备直接访问，因此可以用比用malloc()等函数获得的可翻页内存高得多的带宽来读取或写入。另外：1. 存储位置：cudaMallocHost 分配的内存位于 Host 内存，而 malloc 分配的内存位于操作系统管理的堆中。2.可访问性：cudaMallocHost 分配的内存可以被 Host 和 GPU 访问，而 malloc 分配的内存仅能被 Host 访问。3.性能：cudaMallocHost 可以提高数据传输的性能，因为它允许 GPU 和 Host 共享内存，减少了数据传输时间；而 malloc 分配的内存必须在 GPU 和 Host 之间传输，可能导致更长的数据传输时间。总的来说，如果需要在 GPU 和主机之间共享内存，则应使用 cudaMallocHost。但如果仅需要在主机上分配内存，则可以使用 malloc。