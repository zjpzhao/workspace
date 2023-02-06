## 课程说明
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/CUDA2023%E8%AE%AD%E7%BB%83%E8%90%A5%E8%AF%BE%E7%A8%8B%E8%AF%B4%E6%98%8E.png)


## 实验平台
Jetson Nano
Jetson Nano's Compute Capability: 5.3（所以架构是SM53）
[CUDA GPUs - Compute Capability | NVIDIA Developer](https://developer.nvidia.com/cuda-gpus)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/jestonnano%E5%8F%82%E6%95%B0.png)


### 优点
- 统一存储，减少了耗时的PCIE传输数据操作
- 对于个人用户来说，一个低成本的小jetson设备可能是比较适合入门的。价格便宜，能学习GPU开发，还能学习Linux上的CPU开发，和熟悉ARM CPU等等。


## 下午实验前置知识
### Linux
比较简单，所以不细致记录
#### Ubuntu的文件管理：目录结构
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Ubuntu%E7%9A%84%E6%96%87%E4%BB%B6%E7%AE%A1%E7%90%86%E7%9B%AE%E5%BD%95%E7%BB%93%E6%9E%84.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Ubuntu%E6%9D%83%E9%99%90%E7%AE%A1%E7%90%86.png)

这里的d就是指这是一个文件夹，如果不是d是-，则说明是一个文件而非文件夹。

#### 权限控制
对于权限修改，有字母法和数字法
r  (read) -> 4
w (write) -> 2
x  (excute) -> 1

字母法个人感觉比较繁琐，形如：
`chmod (u g o a) (+ - =) (r w x) (文件名)
例如：
`chmod ugo+r file.txt`
`chmod ug+w, o-w file1.txt file2.txt`

#### 其他收获
内容比较多的文本文件建议用more来代替cat看，好处是分页，滚动查看（要不然行会很多，占用屏幕空间）

### device, global和host修饰符
`__device__ float Func()`执行位置device，调用位置device

`__global__ void KernelFunc()`执行位置是device，调用位置是host或device(arch>3.2)。
- `__global__`定义一个kernel函数，CPU上调用，GPU上执行。
- 必须返回的是void类型
- 2.x: Fermi，3.0/3.2: Kepler (3.2老式嵌入式的TK1）

`__host__ float HostFunc()`执行位置是host，调用位置是host。

问：device和host可以同时修饰一个函数吗？可以！同时有__device__和__host__等于分别生成了两份代码，一份在GPU上，一份在CPU上。方便你将一些通用的小例程，同时得到CPU+GPU的版本。
另外，host修饰的函数不能直接调用device修饰的函数。

### 变量一般的起名规则
d_xxx 的名字，常用来表示GPU上的变量(d = device, GPU)
h_xxx 的名字，常用来表来CPU上的变量(h = host, CPU)


### 编译
#### Makefile的用法
- 就是实现自动化编译，把编译命令都写在一个Makefile文件里，编译的时候只需要运行make（或make -f makefile的文件名）
- make时文件路径下如果有一个文件叫clean怎么办：在makefile里写一行.PHONY: clean，声明clean是个操作。./PHONY表示声明，有声明的会按照声明执行，没有声明的会寻找makefile同级的文件。这里.PHONY 意思表示 clean 是一个“伪目标”，在rm命令前面加了一个小减号的意思就是，也许某些文件出现问题，但不要管，继续做后面的事。注意 clean 的规则不要放在文件的开头，不然，这就会变成 make 的默认目标，一般我们都会放在文件结尾处。

关于Makefile高阶教程，参考[Makefile详细教程\_扫地的小何尚的博客-CSDN博客](https://blog.csdn.net/kunhe0512/article/details/128623790)

#### 单文件编译
.cu的编译器是nvcc
`/usr/local/cuda/bin/nvcc -arch=compute_53 -code=sm_53 hello_cuda.cu -o hello_cuda -run`
code需要大于等于arch。不同的架构，可以理解成不同代的卡（或者嵌入式的带有GPU的Jetson设备）。通过这里的arch和code设定，可以告诉编译器，在哪种显卡上生成代码。我们实验课采用的Jetson Nano算力是5.3，所以架构是SM53。

#### 多文件编译
主文件main.c，引用了add.h，sub.h，div.h，mul.h
```c
OBJ = main.o add.o sub.o mul.o div.o
CC = gcc
app: $(OBJ)
	$(CC) -o app $(OBJ)
main.o: main.c
	$(CC) -c main.c
add.o: add.c
	$(CC) -c add.c
sub.o:sub.c
	$(CC) -c sub.c
mul.o: mul.c
	$(CC) -c mul.c
div.o: div.c
	$(CC) -c div.c
.PHONY : clean //防止有个文件叫clean
clean :
	-rm $(OBJ) app  
```

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%A4%9A%E6%96%87%E4%BB%B6%E7%BC%96%E8%AF%91.png)

#### 头文件
（1）.h里放声明
（2）.cu里面放实现代码
（3）用nvcc进行编译和链接操作，得到可执行文件（注意命令行）。

.cuh和.h等头文件
回答：头文件扩展名无所谓。可以用.h, .hpp, .cuh, 或者.txt也可以。（因为头文件是被#include 后，在包含它的那个文件中.c或者.cpp或者.cu中参与编译的, 最终的文件类型看宿主文件。所以它自身无所谓）

### CUDA线程

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/CUDA%E7%9A%84%E6%89%A7%E8%A1%8C%E6%B5%81%E7%A8%8B.png)

#### Thread
线程层次
- threadIdx为三维向量
- 一个thread block由大量thread组成
- 二维block:
- Block size: (Dx,Dy), thread(x,y) 的ID=x+yDx
- 三维block:
- Block size: (Dx,Dy,Dz), thread(x,y,z) 的
ID=x+yDx+zDxDy

#### Block
- Block也叫线程协作组（CTA）
- 线程以block为单位分配到GPU处理核心（SM）中
- SM内部计算和存储资源有限
- 在目前的NVIDIA GPU中，单程序多数据一个block中最多包含1024个thread
- 同一kernel中的所有block大小都一致
- 一个kernel可以被大量block执行
- 总线程数=block数量\*一个block中的线程数量
- 通常block数量大于GPU SM数量
- Block组成一维、二维、三位的grid
```C
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N){
		C[i][j] = A[i][j] + B[i][j];
	}
	
int main()
{
	...
	// Kernel invocation
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
	...
}
```

- 在硬件中，单程序多数据每个block独立执行
- 不同block可以并行或串行执行
- 不同block可以按照任意顺序被调度和执行
- 保证了程序的可扩展性（scalability）

![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/block%20execute%20in%20SM.png)

- 同一个block中的thread可以通过shared memory共享数据
	- Shared memory是由程序员控制的高速访问存储空间，单程序多数据访问速度类似L1 cache
- 同一个block中的thread可以进行同步
	- 通过在kernel中调用__syncthreads()函数设置同步点
	- 同步表示必须等待所有thread均到达该路障后才可以继续执行之后的代码
```
threadIdx.x 是执行当前kernel函数的线程在block中的x方向的序号  

blockIdx.x 是执行当前kernel函数的线程所在block，在grid中的x方向的序号
```

threadIdx整体是3个uint分量的向量类型（有threadIdx.x, threadIdx.y, threadIdx.z三个分量），类型是unsigned int, 或者说，uint32_t，32位的无符号的整数。

dim3类型等于是一个特殊的uint3(没有用到的元素是1，而不是0）。

#### Grid
一个Grid里有多个Block。
1个block里面能容纳的线程数量较少，x/y/z方向上安排的线程数量分别有限制（1024个，1024个，和64个），但是总大小（它们的乘积，例如我32行，每行32个线程）有限制，最多不能超过1024个线程（三个方向的乘积），不过grid里的blocks数量就比较大了。

我们谈论warp的时候，AMD上叫Wavefront，大小有两种，64（老A卡）和32（新A卡）。


### 一些其他工具的使用
在CUDA samples里的devicequery可以查看GPU的一些详细信息。（现在新的cuda不带samples包了，可以在github上自己找）

![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/deviceQuery%E6%9F%A5%E7%9C%8BCUDA%E9%A9%B1%E5%8A%A8%E7%A8%8B%E5%BA%8F%E5%92%8C%E8%BF%90%E8%A1%8C%E6%97%B6%E7%89%88%E6%9C%AC.png)






## 实验内容

1. 写一个helloc_cuda.cu
```C
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```
编译用`/usr/local/cuda/bin/nvcc -arch=compute_53 -code=sm_53 hello_cuda.cu -o hello_cuda -run`，得到一个叫hello_cuda的执行文件，用`./hello_cuda`即可执行这个文件。
另外，可以采用make的方式完成编译，Makefile这么写：
```makefile
TEST_SOURCE = hello_cuda.cu
TARGETBIN := ./hello_cuda
CC = /usr/local/cuda/bin/nvcc
$(TARGETBIN):$(TEST_SOURCE)
	$(CC)  $(TEST_SOURCE) -o $(TARGETBIN)
.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
	-rm -rf *.o   
```

2. makefile编译多文件的Cuda程序
要编译hello_cuda02-test.cu，它引用的hello_from_gpu.cuh，hello_from_gpu.cu
makefile这么写：
```makefile
TEST_SOURCE = hello_cuda02-test.cu

TARGETBIN := ./hello_cuda_multi_file

CC = /usr/local/cuda/bin/nvcc

$(TARGETBIN):hello_cuda02-test.cu hello_from_gpu.o
	$(CC)  $(TEST_SOURCE) hello_from_gpu.o -o $(TARGETBIN)

hello_from_gpu.o:hello_from_gpu.cu
	$(CC) --device-c hello_from_gpu.cu -o hello_from_gpu.o

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
	-rm -rf *.o
```

查看程序性能：
`sudo /usr/local/cuda/bin/nvprof  ./hello_cuda`回显中的Profiling result：是GPU（kernel函数）上运行的时间，API calls是在cpu上测量的程序调用API的时间。
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/nvprof%E6%9F%A5%E7%9C%8B%E7%A8%8B%E5%BA%8F%E6%80%A7%E8%83%BD%E7%9A%84%E5%9B%9E%E6%98%BE.png)

3. CUDA 线程组织

我们如何能够得到一个线程在所有的线程中的索引值？比如：我们申请了4个线程块，每个线程块有8个线程，那么我们就申请了32个线程，那么我需要找到第3个线程块（编号为2的block）里面的第6个线程（编号为5的thread）在所有线程中的索引值怎么办？  
这时，我们就需要blockDim 和 gridDim这两个变量：  
- gridDim表示一个grid中包含多少个block  
- blockDim表示一个block中包含多少个线程  

也就是说，在上面的那个例子中，gridDim.x=4, blockDim.x=8  
那么，我们要找的第22个线程（编号为21）的唯一索引就应该是，index = blockIdx.x * blockDim.x + threadIdx.x
```C
#include<stdio.h>

__global__ void printid(){
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    
    printf("Hello World from block %d and thread %d!\n", blockId, threadId);
}

int main()
{
    printid<<<5,65>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

编译运行，回显是：
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/nvpro%E7%9A%84api%20trace%E5%8F%82%E6%95%B0%E6%9F%A5%E7%9C%8Bapi%E8%B0%83%E7%94%A8.png)

4. sobel.cu边缘检测kernel优化

```c
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//GPU实现Sobel边缘检测
//             x0 x1 x2 
//             x3 x4 x5 
//             x6 x7 x8 
__global__ void sobel_gpu(unsigned char* in, unsigned char* out, int imgHeight, int imgWidth)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int index = y * imgWidth + x;
    int Gx = 0;
    int Gy = 0;
    unsigned char x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (x > 0 && x < imgWidth && y>0 && y < imgHeight)
    {
        x0 = in[(y - 1) * imgWidth + x - 1];
        x1 = in[(y - 1) * imgWidth + x ];
        x2 = in[(y - 1) * imgWidth + x + 1];
        x3 = in[(y) * imgWidth + x - 1];
        x4 = in[(y ) * imgWidth + x ];
        x5 = in[(y ) * imgWidth + x + 1];
        x6 = in[(y + 1) * imgWidth + x - 1];
        x7 = in[(y + 1) * imgWidth + x ];
        x8 = in[(y + 1) * imgWidth + x + 1];
        Gx = (x0 + x3 * 2 + x6) - (x2 + x5 * 2 + x8);
        Gy = (x0 + x1 * 2 + x2) - (x6 + x7 * 2 + x8);
        out[index] = (abs(Gx) + abs(Gy)) / 2;
    }
}

//CPU实现Sobel边缘检测
void sobel_cpu(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth)
{
    int Gx = 0;
    int Gy = 0;
    for (int i = 1; i < imgHeight - 1; i++)
    {
        uchar* dataUp = srcImg.ptr<uchar>(i - 1);
        uchar* data = srcImg.ptr<uchar>(i);
        uchar* dataDown = srcImg.ptr<uchar>(i + 1);
        uchar* out = dstImg.ptr<uchar>(i);
        for (int j = 1; j < imgWidth - 1; j++)
        {
            Gx = (dataUp[j - 1] + 2 * data[j - 1] + dataDown[j - 1])-(dataUp[j + 1] + 2 * data[j + 1] + dataDown[j + 1]);
            Gy = (dataUp[j - 1] + 2 * dataUp[j] + dataUp[j + 1]) - (dataDown[j - 1] + 2 * dataDown[j] + dataDown[j + 1]);
            out[j] = (abs(Gx) + abs(Gy)) / 2;
        }
    }
}

int main()
{
    //利用opencv的接口读取图片
    Mat img = imread("1.jpg", 0);
    int imgWidth = img.cols;
    int imgHeight = img.rows;

    //利用opencv的接口对读入的grayImg进行去噪
    Mat gaussImg;
    GaussianBlur(img, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);
    //CPU结果为dst_cpu, GPU结果为dst_gpu
    Mat dst_cpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    Mat dst_gpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));


    //调用sobel_cpu处理图像
    sobel_cpu(gaussImg, dst_cpu, imgHeight, imgWidth);

    //申请指针并将它指向GPU空间
    size_t num = imgHeight * imgWidth * sizeof(unsigned char);
    unsigned char* in_gpu;
    unsigned char* out_gpu;
    cudaMalloc((void**)&in_gpu, num);
    cudaMalloc((void**)&out_gpu, num);
    //定义grid和block的维度（形状）
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //将数据从CPU传输到GPU
    cudaMemcpy(in_gpu, img.data, num, cudaMemcpyHostToDevice);
    //调用在GPU上运行的核函数
    sobel_gpu<<<blocksPerGrid,threadsPerBlock>>>(in_gpu, out_gpu, imgHeight, imgWidth);

    //将计算结果传回CPU内存
    cudaMemcpy(dst_gpu.data, out_gpu, num, cudaMemcpyDeviceToHost);
    imwrite("save.png", dst_gpu);
    //显示处理结果, 由于这里的Jupyter模式不支持显示图像, 所以我们就不显示了
    //imshow("gpu", dst_gpu);
    //imshow("cpu", dst_cpu);
    //waitKey(0);
    //释放GPU内存空间
    cudaFree(in_gpu);
    cudaFree(out_gpu);
    return 0;
}

```

编译命令：
`/usr/local/cuda/bin/nvcc sobel.cu -L /usr/lib/aarch64-linux-gnu/libopencv*.so -I /usr/include/opencv4 -o sobel`

这里的so文件也可以先将LD_PRELOAD变量赋值为需要的.so文件，用预加载的方式使用so库文件。(LD_PRELOAD可以影响程序的运行时的链接，它允许你定义在程序运行前优先加载的动态链接库，这个功能主要就是用来有选择性的载入不同动态链接库中的相同函数。)

运行后的回显：
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/sobel%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8BCUDA%E5%AE%9E%E7%8E%B0%E7%9A%84%E5%9B%BE%E5%83%8F%E7%BB%93%E6%9E%9C.png)
