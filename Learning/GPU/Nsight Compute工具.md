# 配置踩坑
说点其他的，在配置Nsight Compute Remote的时候，连接session一直报错：
`Searching for attachable processes on 101.42.156.117:49152-49215`
查遍了全网的debug经验贴结果也没找到解决方法，最终我灵机一动发现问题可能出现在FRP代理，因为我们用于反向代理的服务器没有对Nsight Compute连接会话的端口（49152，可改）进行转发，所以只需要服务器设置本地端口49152和远程端口49152进行绑定，如[[内网穿透 | 1. 创建配置文件目录和配置文件]]进行49152-50215的批量端口转发）。
然后连接时会报Error: ERR_NVGPUVCTRPERM，解决方法是：现在我们要连的目标服务器上用超级管理员权限启动ncu session：
```bash
(base) s3090@b216[20:03:42]:~/zjp/experiments/nsight_compute/target/linux-desktop-glibc_2_11_3-x64$ sudo ./ncu --mode=launch /WdHeDisk/users/s3090/zjp/experiments/benchmarks/PolyBench-ACC/CUDA/linear-algebra/kernels/2mm/2mm
```
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/sudo%E5%90%AF%E5%8A%A8ncu%20session.png)

但是这样执行了命令一直报warning（是NVIDIA的bug）
```bash
==WARNING== Could not deploy stock section files to "/WdHeDisk/users/s3090/Documents/NVIDIA Nsight Compute/2022.3.0/Sections".
Set the HOME environment variable to a writable directory.
==WARNING== Using "/WdHeDisk/users/s3090/zjp/experiments/nsight_compute/target/linux-desktop-glibc_2_11_3-x64/../../sections" instead.
==WARNING== See https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq.
==PROF== Waiting for profiler to attach on ports 49152-49215.
```
解决方法是
```bash
cp /usr/local/cuda-11.1/nsight-compute-2020.2.0/sections /WdHeDisk/users/s3090/zjp/experiments/nsight_compute/target/linux-desktop-glibc_2_11_3-x64/../../
```

连接后还是有问题：查找进程时找到一个进程后，还在一直在查找进程却不进行连接，但是我们起码有一个能用的（如图），所以我们可以这么连：在上一步sudo执行ncu之后，在连接界面点Attach，找到那个能用的Process然后Attach即可

![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E9%80%9A%E8%BF%87Attach%E7%9A%84%E6%96%B9%E5%BC%8F%E8%BF%9E%E6%8E%A5Nsight%20Compute.png)

然后依次点击1和2按钮即可分析kernel
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/ncu%E5%88%86%E6%9E%90kernel.png)

## Register Dependencies
这里*SASS*级别的Register Dependency很爽（如图），但是无法导出到csv（导出后会是空的）。于是我通过查找NVIDIA的文档发现：这个功能是基于CUDA Binary Utilities的[[CUDA Binary Utilities|Register liveness information]]功能实现的，这个可以导出到csv，只需要写一个脚本就可以得到SASS寄存器的依赖对（还没有写），结合Binary Utilities的[[CUDA Binary Utilities|Basic Block Control flow graph]]功能，我们可以比较方便地建图。
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Nsight%20Compute%20Register%20Dependencies.png)




>[官方文档](https://docs.nvidia.com/nsight-compute/2022.3/NsightCompute/index.html)
