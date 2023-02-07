
## 实验前置知识

## 错误检测和事件

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%B8%80%E4%B8%AACUDA%E8%BF%90%E8%A1%8C%E6%97%B6%E6%A3%80%E6%B5%8B%E5%87%BD%E6%95%B0%E5%AE%9E%E4%BE%8B.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%B8%80%E4%B8%AACUDA%E8%BF%90%E8%A1%8C%E6%97%B6%E6%A3%80%E6%B5%8B%E5%87%BD%E6%95%B0%E5%AE%9E%E4%BE%8B2.png)


![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Check%E5%87%BD%E6%95%B02.png)

等待事件完成，设立flag：

```C++
cudaError_t cudaEventSynchronize(cudaEvent_t event);//阻塞（事件完成才记录）
cudaError_t cudaEventQuery(cudaEvent_t event);//非阻塞（事件没完成也会记录）
```
注意cudaEventSynchronize是阻塞的，需要等待时间完成，而cudaEventQuery是非阻塞的，即使事件未完成也会立即执行。

CUDA编程模型中的错误检测中常用cuda error的四个函数：
```C++
__host__​__device__​const char*     cudaGetErrorName ( cudaError_t error )
Returns the string representation of an error code enum name.  

__host__​__device__​const char*     cudaGetErrorString ( cudaError_t error )
Returns the description string for an error code.  

__host__​__device__​cudaError_t     cudaGetLastError ( void )
Returns the last error from a runtime call.  

__host__​__device__​cudaError_t     cudaPeekAtLastError ( void )
```

为了CUDA程序的debug方便，我们可以采用这里的cudaGetErrorString函数，将其封装在error.cuh中：
```C++
#pragma once
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

```
然后我们就可以在自己的.cu文件中引用该检错头文件，#include "error.cuh"，利用其定义的CHECK()函数对我们使用的CUDA api进行检错，形如：
`CHECK(cudaMallocHost((void **) &h_a, sizeof(int)*m*n));`
`CHECK(cudaEventCreate(&start));`
`CHECK(cudaEventCreate(&stop));`
`CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice));`
`CHECK(cudaEventRecord(stop));`
`CHECK(cudaEventSynchronize(stop));`
`CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));`
下面的实验有检错回显，这个封装思路很巧妙，用来Debug确定出错的位置很好用，具体可以参考樊老师的《CUDA编程基础与实践》一书的git repo：
[CUDA-Programming/src/04-error-check at master · brucefan1983/CUDA-Programming · GitHub](https://github.com/brucefan1983/CUDA-Programming/tree/master/src/04-error-check)

### CUDA存储单元
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%A4%9A%E7%A7%8DCUDA%E5%AD%98%E5%82%A8%E5%8D%95%E5%85%83.png)

右边单向箭头表示是可读的，双向箭头表示是可读可写的（据说这里考试会有小陷阱）

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%A4%9A%E7%A7%8DCUDA%E5%AD%98%E5%82%A8%E5%8D%95%E5%85%83%E8%AF%A6%E8%A7%A3.png)

下面我们大体上从由快到慢的顺序介绍GPU的各种存储单元，最后讨论主机端的存储器内存。

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

注意这里的bank conflict，只有在warp中的线程都访问同一个bank的资源的时候才不会存在bank conflict

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E9%81%BF%E5%85%8Dbank%20conflict%EF%BC%88%E6%97%A0bank%20conflict%E7%9A%84%E6%83%85%E5%86%B5%EF%BC%89.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E9%81%BF%E5%85%8Dbank%20conflict%EF%BC%88%E6%9C%89bank%20conflict%E7%9A%84%E6%83%85%E5%86%B5%EF%BC%89.png)

或者用Memory Padding的方式避免bank conflict
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%94%A8Memory%20Padding%E7%9A%84%E6%96%B9%E5%BC%8F%E9%81%BF%E5%85%8Dbank%20conflict.png)
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/2%E7%94%A8Memory%20Padding%E7%9A%84%E6%96%B9%E5%BC%8F%E9%81%BF%E5%85%8Dbank%20conflict.png)
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%94%A8Memory%20Padding%E7%9A%84%E6%96%B9%E5%BC%8F%E9%81%BF%E5%85%8Dbank%20conflict%E7%9A%84%E4%BB%A3%E7%A0%81.png)

连续的数据存储，第一行的x就变成了

#### Local Memory
属于On-board（而不是on chip），但却是GPU线程私有的，空间比较大，而读写比较慢。（这一点需要注意）
Register不够的时候就会用Local Mem来替代，但更多地是在以下情况使用Local Memory：
1. 无法确定其索引是否为常量的数组
2. 会消耗太多寄存器空间的大型结构或数组
3. 如果内核使用了多于可用寄存器的任何变量（这也称为寄存器溢出）
4. `--ptxas-options=-v`

#### Constant Memory
- 固定内存空间驻留在设备内存中，并缓存在固定缓存中（constant cache）。
- 范围是全局（对所有kernel可见）
- kernel从CM只能读而不能写，因此初始化必须在host端使用`cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count);`
- 当一个warp中所有的thread都从同一个Memory地址读取数据时，constant Memory表现会非常好，会触发广播机制。

##### 常量内存应用举例-光线追踪
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%85%89%E7%BA%BF%E8%BF%BD%E8%B8%AA.png)

这里是效果图，我们要在某个位置显示球的颜色或者黑色，其中球之间可能存在遮盖，所以需要经过计算距离来确定哪个球在最前面，然后显示这个球的颜色。

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%85%89%E7%BA%BF%E8%BF%BD%E8%B8%AAhit%E6%96%B9%E6%B3%95.png)

这里采用的是hit方法，计算光线是否与球面相交，若相交则返回光线到命中球面的距离。

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%85%89%E7%BA%BF%E8%BF%BD%E8%B8%AA%E6%B8%B2%E6%9F%93.png)

这里我们将需要大量访问的内容放到常量内存中，也就是说将球体的位置数据部分放到`__constant__ Sphere s[SPHERES];`中。实现光线追踪部分的代码，首先将threadIdx映射到像素的位置，每个线程都干自己的事情，然后让图像坐标偏移DIM/2，使z轴穿过图像中心，初始化背景颜色为黑色，距离初始化为负无穷-INF，然后开始计算距离：遍历每一个球体，调用上面的hit方法计算光线和球面的距离，如果距离更近则将距离更新为此值，否则不用修改距离值。完成对球面相交的判断后，将当前的颜色等信息保存到我们输出的图像中，`ptr[offset*4 + 0]`这里的4表示每个点的r，g，b和透明度信息共需要四个存储单元，这些信息存储在一个一维数组中，所以组织的时候需要引入offset偏移量进行索引。

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%85%89%E7%BA%BF%E8%BF%BD%E8%B8%AA%E7%94%9F%E6%88%90bitmap.png)

最后生成球面的中心坐标颜色和半径，通过球面数据生成bitmap（这里讲的比较粗略）。


#### Texture Memory
- 驻留在device Mem中，属于On-board，并使用一个只读cache。
- 与global Memory在一块，但是有自己专有的只读cache
- on-chip，所以比DRAM上取数据减少了内存请求和提高带宽
- 专门为那些在内存访问模式中存在大量空间局部性的图形应用程序而设计的。（也就是说，一个thread读的位置可能与临近的thread读的位置非常接近，如下）。
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/texture%20memory%E5%92%8C%E6%95%B0%E6%8D%AE%E5%B1%80%E9%83%A8%E6%80%A7.png)

举个texture memory的应用实例-热传导模型：
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/texture%20memory%E5%BA%94%E7%94%A8%E4%BA%8E%E7%83%AD%E4%BC%A0%E5%AF%BC%E6%A8%A1%E5%9E%8B.png)


#### Global Memory
- 空间最大，latency最高，是GPU中最基础的memory。
- On-board，驻留在Device memory中
- memory transction 对齐，合并访存。
- 合并访存机制，如下图的矩阵乘法：
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/global%20memory%E7%9A%84%E5%90%88%E5%B9%B6%E8%AE%BF%E5%AD%98%E6%9C%BA%E5%88%B6%E4%B8%BE%E4%BE%8B%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95.png)

我们可以让线程按照行或者列进行读取，那么哪一种更快呢？答案是**按列读快**，如下左边是按行读（例如thread0用四次迭代分别读取A00，A01，A02，A03），右边是按照列读（例如thread0用四次迭代分别读取B00，B10，B20，B30）。白色空格部分指的是访存取出来的数据空间部分，显然按行取效率低下，而这里按照列读取的话，我们每一次迭代只需要一次访存即可满足四个线程的取数据操作。
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%90%88%E5%B9%B6%E8%AE%BF%E5%AD%98.png)

### Host Memory
主机端存储器主要是内存可以分为两类：可分页内存（Pageable）和页面 （Page-Locked 或 Pinned）内存。
可分页内存通过操作系统 API(malloc/free) 分配存储器空间，该内存是可以换页的，即内存页可以被置换到磁盘中。可分页内存是不可用使用DMA（Direct Memory Acess)来进行访问的，普通的C程序使用的内存就是这个内存。

### 总结
不同的应用可能适用于不同的存储单元或他们的各种组合，我们要掌握好每种存储单元的特点并合理架构，所有这些常用的GPU存储单元的特性汇总如下表：
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%A4%9A%E7%A7%8DGPU%E5%AD%98%E5%82%A8%E5%8D%95%E5%85%83%E7%9A%84%E7%89%B9%E6%80%A7%E6%B1%87%E6%80%BB.png)


## 实验内容
### CUDA实现矩阵乘法

核心思想就是用空间换时间
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%BA%8C%E7%BB%B4grid%E5%92%8C%E4%BA%8C%E7%BB%B4block.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%AE%97%E4%B8%80%E7%BB%B4%E5%9D%90%E6%A0%87.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/CUDA%E5%AE%9E%E7%8E%B0%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95.png)

这部分内容比较简单，不赘述，回显对比了用cpu进行计算和用gpu计算的矩阵乘法结果是否一致：
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/CUDA%E5%AE%9E%E7%8E%B0%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E5%9B%9E%E6%98%BE.png)


### 检错和计时
Cuda编程模型中的事件。事件的本质就是一个标记，它与其所在的流内的特定点相关联。可以使用时间来执行以下两个基本任务：
-   同步流执行
-   监控设备的进展
流中的任意点都可以通过API插入事件以及查询事件完成的函数，只有事件所在流中其之前的操作都完成后才能触发事件完成。默认流中设置事件，那么其前面的所有操作都完成时，事件才出发完成。 事件就像一个个路标，其本身不执行什么功能，就像我们最原始测试c语言程序的时候插入的无数多个printf一样。
声明:

```C++
cudaEvent_t event;
```

创建：

```C++
cudaError_t cudaEventCreate(cudaEvent_t* event);
```

销毁：

```C++
cudaError_t cudaEventDestroy(cudaEvent_t event);
```

添加事件到当前执行流：

```C++
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
```

等待事件完成，设立flag：

```C++
cudaError_t cudaEventSynchronize(cudaEvent_t event);//阻塞
cudaError_t cudaEventQuery(cudaEvent_t event);//非阻塞
```

当然，我们也可以用它来记录执行的事件：

```C++
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop);
```

下面来测试一下核函数执行的时间，另外也引入我们的错误检测头文件error.cuh，.cu代码这么写：
```C++
#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int m=100;
    int n=100;
    int k=100;

    int *h_a, *h_b, *h_c, *h_cc;
    CHECK(cudaMallocHost((void **) &h_a, sizeof(int)*m*n));
    CHECK(cudaMallocHost((void **) &h_b, sizeof(int)*n*k));
    CHECK(cudaMallocHost((void **) &h_c, sizeof(int)*m*k));
    CHECK(cudaMallocHost((void **) &h_cc, sizeof(int)*m*k));
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));


    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **) &d_a, sizeof(int)*m*n));
    CHECK(cudaMalloc((void **) &d_b, sizeof(int)*n*k));
    CHECK(cudaMalloc((void **) &d_c, sizeof(int)*m*k));


    // copy matrix A and B from host to device memory
    CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice));

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    CHECK(cudaEventRecord(start));
    //cudaEventQuery(start);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms.\n", elapsed_time);

     CHECK(cudaEventDestroy(start));
     CHECK(cudaEventDestroy(stop));
    CHECK(cudaMemcpy(h_c, d_c, (sizeof(int)*m*k), cudaMemcpyDeviceToHost));
    //cudaThreadSynchronize();
    

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    int ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if(fabs(h_cc[i*k + j] - h_c[i*k + j])>(1.0e-10))
            {
                
                ok = 0;
            }
        }
    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    // free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));
    CHECK(cudaFreeHost(h_cc));
    return 0;
}
```
其中引用的error.cuh参考[CUDA-Programming/error.cuh at master · brucefan1983/CUDA-Programming · GitHub](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/04-error-check/error.cuh)

PS: 这里我发现了给的答案的一处小错误，h_cc没有在最后进行释放。我们需要进行`CHECK(cudaFreeHost(h_cc));`

Makefile这么写：
```makefile
TEST_SOURCE = matrix_mul.cu

TARGETBIN := ./matrix_mul

CC = /usr/local/cuda/bin/nvcc

$(TARGETBIN):$(TEST_SOURCE)
	$(CC)  $(TEST_SOURCE) -o $(TARGETBIN) -I ./
    

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
```
这里由于我们引用的头文件error.cuh，所以编译命令里需要写-I选项，表明include路径包含本文件夹，确保能找到error.cuh。

PS: 这里由于是我自己在jupyter上手敲的内容，按tab会默认加四个空格而非是制表符，所以一直会报下面这个错，复制别的地方的tab过来以后就好了。
![e5b99bd52f76ce49d2a796a6c61ba78.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/makefile%E9%87%8C%E6%89%93tab%E7%9A%84%E6%97%B6%E5%80%99%E6%98%AF%E5%9B%9B%E4%B8%AA%E7%A9%BA%E6%A0%BC%E8%80%8C%E6%8A%A5%E9%94%99.png)

那么编译运行后，如果程序出错，回显是这样的：
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%94%A8error.cuh%E8%BF%9B%E8%A1%8C%E7%A8%8B%E5%BA%8F%E6%A3%80%E9%94%99%E6%97%B6%E7%A8%8B%E5%BA%8F%E5%87%BA%E9%94%99%E7%9A%84%E5%9B%9E%E6%98%BE.png)

正常的回显会打出来调用GPU计算矩阵乘法时的耗时：
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/GPU%E8%AE%A1%E7%AE%97%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E6%97%B6%E7%9A%84%E8%80%97%E6%97%B6.png)


真的是十分方便！

### 用shared memory加速矩阵乘法和bank conflict
当我们在处理矩阵乘法时，假设矩阵$M(m,k)*N(k,n) = P(m,n)$。那么，矩阵M中的一个数值m(x,y),就要被grid中所有满足$threadIdx.y+blockIdx.y*blockDim.y = y$的线程从Global Memory中读一次，一共就是K次。那么，我们看到这么多重复读取，就可以把这个变量放在Shared Memory中，极大地减少每次的读取时间。
我们采用分tile块的方式，将一部分子矩阵加载到SMem中，如图中的蓝色块和橙色块。
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/shared_memory%20tile%E5%8A%A0%E9%80%9F%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95.png)
代码这么写：
```C++
#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

__global__ void gpu_matrix_mult_shared(int *d_a, int *d_b, int *d_result, int n) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        tile_a[threadIdx.y][threadIdx.x] = row<n && (sub * BLOCK_SIZE + threadIdx.x)<n? d_a[idx]:0;
        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        tile_b[threadIdx.y][threadIdx.x] = col<n && (sub * BLOCK_SIZE + threadIdx.y)<n? d_b[idx]:0;

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int m=1000;
    int n=1000;
    int k=1000;

    int *h_a, *h_b, *h_c, *h_cc, *h_cs;
    CHECK(cudaMallocHost((void **) &h_a, sizeof(int)*m*n));
    CHECK(cudaMallocHost((void **) &h_b, sizeof(int)*n*k));
    CHECK(cudaMallocHost((void **) &h_c, sizeof(int)*m*k));
    CHECK(cudaMallocHost((void **) &h_cc, sizeof(int)*m*k));
    CHECK(cudaMallocHost((void **) &h_cs, sizeof(int)*m*k));
    
    cudaEvent_t start, stop,stop_share;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventCreate(&stop_share));


    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = 1;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = 0;
        }
    }

    int *d_a, *d_b, *d_c, *d_c_share;
    CHECK(cudaMalloc((void **) &d_a, sizeof(int)*m*n));
    CHECK(cudaMalloc((void **) &d_b, sizeof(int)*n*k));
    CHECK(cudaMalloc((void **) &d_c, sizeof(int)*m*k));
    CHECK(cudaMalloc((void **) &d_c_share, sizeof(int)*m*k));

    CHECK(cudaEventRecord(start));
    // copy matrix A and B from host to device memory
    CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice));

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    

    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m,n,k);    

    CHECK(cudaMemcpy(h_c, d_c, (sizeof(int)*m*k), cudaMemcpyDeviceToHost));
    //cudaThreadSynchronize();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    gpu_matrix_mult_shared<<<dimGrid, dimBlock>>>(d_a, d_b, d_c_share, n);
    CHECK(cudaMemcpy(h_cs, d_c_share, (sizeof(int)*m*k), cudaMemcpyDeviceToHost));
    
    CHECK(cudaEventRecord(stop_share));
    CHECK(cudaEventSynchronize(stop_share));
    
    float elapsed_time, elapsed_time_share;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CHECK(cudaEventElapsedTime(&elapsed_time_share, stop, stop_share));
    printf("Time_global = %g ms.\n", elapsed_time);
    printf("Time_share = %g ms.\n", elapsed_time_share);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));    

    cpu_matrix_mult(h_a, h_b, h_c, m, n, k);

    int ok = 1;
    for (int i = 0; i < m; ++i)
    { 
        for (int j = 0; j < k; ++j)
        {
            if(fabs(h_cs[i*k + j] - h_c[i*k + j])>(1.0e-10))
            {
                printf("hcs: %d hc: %d  ",h_cs[i*k + j], h_c[i*k + j]);
                ok = 0;
            }
        }
    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }
    
    // free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));
    CHECK(cudaFreeHost(h_cc));
    return 0;
}
```

这里规模改成了1000，能让加速效果看得更明显一些。
![f1d7f1f9b3dc4b51f321e62b28d7a46.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%94%A8shared%20memory%E5%8A%A0%E9%80%9F%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95.png)

#### bank conflict
为了人为制造bank conflict，内核代码这样写，其他的不变
```C++
__global__ void gpu_matrix_mult_shared(int *d_a, int *d_b, int *d_result, int m, int n, int k) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        // 没有 bank conflict
        //idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        //tile_a[threadIdx.y][threadIdx.x] = row<n && (sub * BLOCK_SIZE + threadIdx.x)<n? d_a[idx]:0;
        //idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        //tile_b[threadIdx.y][threadIdx.x] = col<n && (sub * BLOCK_SIZE + threadIdx.y)<n? d_b[idx]:0;
        
        //生成 st bank conflict
        idx = row * n + sub * BLOCK_SIZE + threadIdx.y;
        tile_a[threadIdx.x][threadIdx.y] = row<n && (sub * BLOCK_SIZE + threadIdx.y)<n? d_a[idx]:0;
        idx = (sub * BLOCK_SIZE + threadIdx.x) * n + col;
        tile_b[threadIdx.x][threadIdx.y] = col<n && (sub * BLOCK_SIZE + threadIdx.x)<n? d_b[idx]:0;
        
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}
```
使用命令查看bank conflict情况，引入的是--events shared_ld_bank_conflict,shared_st_bank_conflict参数
`sudo /usr/local/cuda/bin/nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict ./matrix_mul`

结果如下所示，可以看到我们制造了st_bank_conflict：
![a83bf7d3d7744ffb4ec7a9f850d4bc2.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%88%B6%E9%80%A0bank%20conflict.png)

没有bank conflict的长这样：
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E6%B2%A1%E6%9C%89bank%20conflict%E7%9A%84%E5%9B%9E%E6%98%BE.png)



## 打卡题目和其他知识：
1. GPU上多个线程的原子操作的多个分解步骤物理上不可能并行这句话对么？
答：这句话不完全正确。GPU上多个线程可以同时进行原子操作，但是对于特定的存储单元，只有一个线程能够成功地完成原子操作。因此，多个线程同时对相同的存储单元执行原子操作时，其中的一些线程可能会遇到冲突，并且仅有一个线程的操作会被执行，其他的线程的操作将被阻塞直到它们能够完成操作为止。
2. 原子操作所使用的存储单元，必须是shared memory吗？
答：不是必须是shared memory。原子操作可以在多种存储单元中进行，包括global memory和shared memory，具体取决于操作的需求和性能要求。

3. cudaMallocHost和使用malloc的区别
回答：驱动程序跟踪用这个函数分配的虚拟内存范围，并自动加速对cudaMemcpy()等函数的调用。由于内存可以被设备直接访问，因此可以用比用malloc()等函数获得的可翻页内存高得多的带宽来读取或写入。另外：1. 存储位置：cudaMallocHost 分配的内存位于 Host 内存，而 malloc 分配的内存位于操作系统管理的堆中。2.可访问性：cudaMallocHost 分配的内存可以被 Host 和 GPU 访问，而 malloc 分配的内存仅能被 Host 访问。3.性能：cudaMallocHost 可以提高数据传输的性能，因为它允许 GPU 和 Host 共享内存，减少了数据传输时间；而 malloc 分配的内存必须在 GPU 和 Host 之间传输，可能导致更长的数据传输时间。4.避免内存虚拟化技术从内存移到disk。总的来说，如果需要在 GPU 和主机之间共享内存，则应使用 cudaMallocHost。但如果仅需要在主机上分配内存，则可以使用 malloc。

4. 注意cudaMemcpy已经包含了同步过程，就不需要显式调用sync了
5. [(含代码)利用GPU版的Numpy---cuNumeric加速Python数值计算\_扫地的小何尚的博客-CSDN博客](https://blog.csdn.net/kunhe0512/article/details/128908418)