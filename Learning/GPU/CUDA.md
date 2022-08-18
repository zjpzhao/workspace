#learning/GPU/cuda

# 概述
一个经典的CUDA程序结构包含五个部分[^1]
1. 分配GPU空间
2. 将数据从CPU端复制到GPU端
3. 调用CUDA kernel执行计算
4. 计算完成后从GPU拷贝数据回CPU
5. 清理GPU内存空间

## Two Memory
异构的Host和Device都有各自的存储空间，由PCI-Express 总线区分开：
Host：CPU and CPU memory (Host memory)
Device: GPU and GPU memory (Device memory)
所以代码中一般以h_为前缀表示host memory，d_为前缀表示device memory。

## kernel()
kernel函数是CUDA编程的关键，主要对拷贝到GPU的数据进行操作，在GPU中执行，函数用标识符`__global__`修饰。

## 异步性
CUDA编程的异步性体现在：CPU和GPU都属于计算机的计算部件，各有存储空间。GPU执行程序时启动kernel()后控制权返还给CPU，串行代码C由Host执行，并行代码CUDA C由Device执行。



```CPP
#include<stdio.h>

__global__ void helloFromGPU (void){
    printf("Hello World from GPU!\n");
}
 int main(void) {
    printf("Hello World from CPU!\n");
    helloFromGPU<<<1,1>>>();
    // helloFromGPU<<<1,10>>>();    output 10 times
    //一个kernel是由一组线程执行，所有线程执行相同的代码。上面一行三对尖括号中的1和10 表明了该function将有10个线程
    cudaDeviceReset();
    return 0;
}
```


## NVCC编译
#learning/cuda/nvcc  
CUDA Compilation Trajectory
![](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/graphics/cuda-compilation-from-cu-to-executable.png)
nvcc编译cuda代码的时候，Host和Device是分开进行的，nvcc --keep选项可以保存编译.cu的过程文件（如.ptx, .cubin等），--dryrun是只看编译过程但不真正编译，PTX是每一个线程都需要执行的，我猜测需要执行该PTX的线程号是通过链接.cubin文件而分配的。具体需要参考和探索CUDA binary
- [ ] 离线编译和在线编译参考[《CUDA C Programming Guide》(《CUDA C 编程指南》)导读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/53773183)
https://www.findhao.net/easycoding/2039.html

# 环境搭建
![[插件用法#^bd19ab]]




nvprof 分析工具使您能够从命令行收集和查看分析数据。请注意，Visual Profiler 和 nvprof 将在将来的 CUDA 版本中弃用。NVIDIA Volta 平台是最后一个完全支持这些工具的架构。建议使用下一代工具 NVIDIA Nsight Systems 进行 GPU 和 CPU 采样和跟踪，并使用 NVIDIA Nsight Compute 进行 GPU 内核分析。

---
使用`deviceQuery`命令确定系统上的 CUDA 驱动程序和运行时版本。deviceQuery 命令在 CUDA SDK 中可用。
`cd /usr/local/cuda/samples/1_Utilities/deviceQuery`
`./deviceQuery`
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/deviceQuery%E6%9F%A5%E7%9C%8BCUDA%E9%A9%B1%E5%8A%A8%E7%A8%8B%E5%BA%8F%E5%92%8C%E8%BF%90%E8%A1%8C%E6%97%B6%E7%89%88%E6%9C%AC.png)



[^1]: 参考https://www.cnblogs.com/1024incn/p/4537177.html



# NVTX标签
在nsys中显示的部分函数与原分析代码怎么对应起来呢？就可以采用打NVTX标签的方法：
```C
#include "nvToolsExt.h"
...
void myfunc(int n, double * x*)
{
	nvtxRangePushA("init_host_data");//你想标注的名字
	//init x on host
	init_host_data(n,x,x_d,y_d);
	nvtxRangePop();
}
...
```
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/NVTX%20instrumentation%20example.png)

# NCU
 ![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/nsight%20compute.png)
可视化工具：nv-nsight-cu，命令行工具：nv-nsight-cu-cli
用API stream做交互式分析
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/nvprof%20and%20ncu%20events.png)

![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/nvprof%20and%20ncu%20metrics.png)

### Speed of Light reports
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/sol%20sec%20compute%20bound.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/SOL%20memory%20bound.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/SOL%20latency%20bound.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/SOL%20sections.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/SOL%20Section%20Unit%20details.png)



待看教程
- [ ] https://zhuanlan.zhihu.com/p/34587739