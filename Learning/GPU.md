# GPU是怎么计算的
#learning/GPU 
>参考视频：https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/
## 计算密集型
为什么FLOPs不重要?
内存将数据传到GPU速度是200GBytes/s（＝25Giga-FP64/s，即每秒提供250亿个FP64值，∵FP64=8bytes），CPU大约能达到2000GFLOPs FP64（两万亿次双精度运算），所以内存不能以所需的速度供给CPU。
$$
Compute\ Intensity=\frac{FLOPs}{Data\ Rate}=\frac{2000}{25}=80
$$
在这个例子中，我需要CPU对每一个数据进行80此操作，否则CPU就会空闲，这样还不如买一个更便宜的CPU
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/CPU%E5%92%8CGPU%E8%AE%A1%E7%AE%97%E5%BC%BA%E5%BA%A6%E5%AF%B9%E6%AF%94.png)
我们需要尽可能降低计算强度，因为没有算法能在每次load时做100次，所以当增加FLOPs的速度大于增加内存带宽的速度时，*Compute Intensity*就会上升，所以需要在程序算法上不断创新来保持这个值，所以我们认为**FLOPs不重要**，因为我们已经有足够的FLOPs了。（四分之三的程序都被内存带宽完全限制住了，所以内存是瓶颈）
## 关心延迟
我们更应该关心的是延迟。处理器有很多专用的指令集比如乘加运算FMA，可以在一个指令里完成乘和加运算。加载数据的时间可以抵消FLPOs
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/CPU%E6%8C%87%E4%BB%A4%E6%B5%81%E6%B0%B4%E7%BA%BF.png)
电流在硅中的传播速度只有光的五分之一（6万公里/s），根据经验一个时钟周期电流的移动只有20mm，所以通过内存和CPU之间的总线传递数据时就需要用5到10个左右时钟周期才能返回，总而言之memory bus占用了99%以上的时间
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%B8%8D%E5%90%8C%E8%8A%AF%E7%89%87%E7%9A%84%E5%B8%A6%E5%AE%BD%E5%92%8C%E6%95%88%E7%8E%87.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/daxpy%E5%85%85%E5%88%86%E5%88%A9%E7%94%A8%E5%86%85%E5%AD%98.png)

### 并行和并发
并行性（parallelism）比并发性（concurrency）强：
- 并行的关键是有同时处理多个任务的能力；并发的关键是有处理多个任务的能力但不一定要同时。
- 并行性是指在硬件限制下每个线程同时执行一个操作
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%B8%8D%E5%90%8C%E8%8A%AF%E7%89%87%E7%9A%84%E7%BA%BF%E7%A8%8B%E6%AF%94.png)
- 并发应用：比如循环展开（Loop Unrolling）：
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%B9%B6%E5%8F%91%E6%96%B9%E5%BC%8F%E5%87%8F%E5%B0%91%E8%BF%AD%E4%BB%A3%E6%AC%A1%E6%95%B0.png)

### GPU和CPU设计的根本区别
GPU具有更高的延迟和更多的线程，也就是说GPU相当于是**吞吐机**而CPU是**延迟机**：GPU设计师将所有资源投入到添加更多线程中而不是减少延迟，CPU的期望是一个线程基本完成所有的工作，将这些线程从一个切换到另一个是非常昂贵的，就像操作系统中进程的上下文切换一样 ，所以你只需要足够线程就可以解决延迟问题，而CPU设计者把所有的资源都投入到减少延迟上了，不是增加线程而是使用完全相反的方法来解决延迟问题，这就是GPU和CPU设计的根本区别

## GPU的分层存储架构
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/gpu%20cache.png)
这里寄存器也作为Cache的一种，这是一个非常重要的GPU细节，寄存器需要更靠近内存并且需要足够大的内存空间来完成FLOPs请求，能做的内存操作与寄存器数量直接相关。A100 GPU原则上可以维持27兆的高速数据（如上图），这是A100中总寄存器的数量，可以存330万个双精度数据——我们可以看出GPU使用寄存器缓存数据来解决高延迟问题，以及通过靠近数据来减少延迟