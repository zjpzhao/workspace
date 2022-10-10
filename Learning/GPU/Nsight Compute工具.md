# 配置踩坑
说点其他的，在配置Nsight Compute Remote的时候，连接session一直报错：
`Searching for attachable processes on 101.42.156.117:49152-49215`
查遍了全网的debug经验贴结果也没找到解决方法，最终我灵机一动发现问题可能出现在FRP代理，因为我们用于反向代理的服务器没有对Nsight Compute连接会话的端口（49152，可改）进行转发，所以只需要服务器设置本地端口49152和远程端口49152进行绑定，如[[内网穿透 | 1. 创建配置文件目录和配置文件]]进行49152-50215的批量端口转发）。
然后连接时会报Error: ERR_NVGPUVCTRPERM，解决方法是：现在我们要连的目标服务器上用超级管理员权限启动ncu session：
```bash
(base) s3090@b216[20:03:42]:~/zjp/experiments/nsight_compute/target/linux-desktop-glibc_2_11_3-x64$ sudo ./ncu --mode=launch /WdHeDisk/users/s3090/zjp/experiments/benchmarks/PolyBench-ACC/CUDA/linear-algebra/kernels/2mm/2mm
```
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/sudo%E5%90%AF%E5%8A%A8ncu%20session.png)

连接后还是会一直在查找进程，但是起码有一个能用的（如图），所以我们可以这么连：在上一步sudo执行ncu之后，在连接界面点Attach，找到那个能用的Process然后Attach即可

![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E9%80%9A%E8%BF%87Attach%E7%9A%84%E6%96%B9%E5%BC%8F%E8%BF%9E%E6%8E%A5Nsight%20Compute.png)

然后依次点击1和2按钮即可分析kernel
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/ncu%E5%88%86%E6%9E%90kernel.png)
