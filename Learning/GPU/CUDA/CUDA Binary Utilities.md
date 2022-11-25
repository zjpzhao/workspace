# Basic Block Control flow graph 
关于提取SASS级别的Basic Block Control Flow Graph（输出可以是DOT格式，然后用[Graphviz](http://www.graphviz.org/)进行可视化，或者直接输出.png文件）可以参考文档：
[3.1.1. Control flow graph information](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#nvdisasm-usage-cfg)

# Register liveness information
关于SASS指令寄存器依赖的静态提取：参考[3.1.2. Register liveness information](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#nvdisasm-usage-liveness)，Nsight Compute的[[Nsight Compute工具|Register Dependencies]]功能就是通过本功能实现的

主要是通过nvdisasm工具的`-plr (--print-life-ranges)`和`-lrm (--life-range-mode)`两个参数，来分析.cubin文件得到该信息（另外，可以用参数`-b (--binary <SMxy>)`来控制architecture。)

注意：nvdisasm需要完整的重定位信息来做控制流分析。如果CUDA二进制文件中缺少这种信息，那么要么使用nvdisasm选项"-ndf "来关闭控制流分析，要么使用ptxas和nvlink选项"-preserve-relocs "来重新生成cubin文件。