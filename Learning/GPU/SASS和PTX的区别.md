**官方支持**：PTX是NVIDIA官方支持的最底层，有相关的文档（见[Parallel Thread Execution ISA](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/parallel-thread-execution/index.html)）和完善的工具链（NVCC，cuobjdump，PTXAS等等），也可以在driver api中load，甚至支持cuda C中[inline PTX assembly](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/inline-ptx-assembly/index.html)。而SASS这层只有非常简略的介绍[SASS Instruction Set Reference](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-binary-utilities/index.html%23instruction-set-ref)，虽然其中也提供了一些工具如nvdisasm和cuobjdump做一些分析，但也非常局限。Debug上两者倒是差别不大，NSight功能比较完善了，现在应该是可以支持cuda C/PTX/SASS三个层级的debug。

> 参考：[CUDA微架构与指令集（2）-SASS指令集概述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/161624982)




PTX代码的兼容性远强于二进制代码。只要不涉及到不同架构上的特性差异，PTX可以在任何架构上运行。

不过PTX代码在两种情况下其兼容性会受限：  
1. 若PTX代码使用了较高级别架构的特有特性，则无法在较低架构上运行。例如若PTX代码用到了计算能力3.0以上才能使用的Warp Shuffle特性，则无法在2.x或1.x平台上运行。 2. 若PTX在较低架构上生成，则虽然能够在所有更高级别的架构上运行，但无法充分利用这些架构的硬件特性，造成性能无法最大化的问题。

在编译时，可以通过`-arch`来指定生成的PTX代码的版本，如`-arch=compute_30`。
为了保证应用程序的兼容性，最好是将代码编译成PTX代码，然后依靠各个计算能力的驱动程序在线编译成对应平台的二进制代码cubin。

除了使用`-arch`和`-code`来分别指定C->PTX和PTX->cubin的计算能力外，还可以用`-gencode`关键字来操作，如下例：

```text
nvcc x.cu
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_60,code=\'compute_60,sm_60\'
```

使用上述编译指令后，会生成3.5/5.0/6.0的cubin文件，以及6.0的PTX代码。具体内容请参考nvcc user manual。

对于主机端代码，会自动编译，并在运行时决定调用哪一个版本的执行。对于上例，主机端代码会编译为：3.5/5.0/6.0的二进制文件，以及7.0的PTX文件。

另外，在程序中可以使用`__CUDA_ARCH__`宏来指定计算能力(只能用于修饰设备端代码)。计算能力3.5在程序中对应的`__CUDA_ARCH__`为350。

有一点需要注意的是，7.0以前，都是以线程束为单位在调度，线程束内指令永远是同步的，被成为**锁步**。而Volta架构(计算能力7.x)引入了Independent Thread Scheduling，破坏了线程束内的隐式同步。因此，如果老版本的代码里面有默认锁步的代码，在Volta架构下运行时可能会因为锁步的消失而出问题，可以指定`-arch=compute_60 \-code=sm_70`，即将PTX编到Pascal架构下以禁用Independent Thread Scheduling特性。（当然，也可以修改代码来显示同步）

另外，版本相关编译指令有缩写的情况，具体看手册。
> 参考：https://www.cvmart.net/community/detail/6486