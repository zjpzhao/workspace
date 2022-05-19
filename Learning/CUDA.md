#learning/cuda

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

# 环境搭建
![[插件用法#^bd19ab]]



nvprof 分析工具使您能够从命令行收集和查看分析数据。请注意，Visual Profiler 和 nvprof 将在将来的 CUDA 版本中弃用。NVIDIA Volta 平台是最后一个完全支持这些工具的架构。建议使用下一代工具 NVIDIA Nsight Systems 进行 GPU 和 CPU 采样和跟踪，并使用 NVIDIA Nsight Compute 进行 GPU 内核分析。





[^1]:参考https://www.cnblogs.com/1024incn/p/4537177.html