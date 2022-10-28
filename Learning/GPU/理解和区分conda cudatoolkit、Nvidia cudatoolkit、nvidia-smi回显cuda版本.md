nvidia-smi上显示的CUDA版本是当前driver所支持的最高的cudatoolkit版本
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/nvidia-smi%E4%B8%8A%E6%98%BE%E7%A4%BA%E7%9A%84CUDA%E7%89%88%E6%9C%AC.png)
下面介绍彼此的关系和区别

参考[https://stackoverflow.com/questions/56470424/nvcc-missing-when-installing-cudatoolkit](https://stackoverflow.com/questions/56470424/nvcc-missing-when-installing-cudatoolkit)和[https://www.zhihu.com/question/344950161](https://www.zhihu.com/question/344950161)

当在conda中安装cudatoolkit 的时候，可能会没有CUDA的编译器nvcc，原因是conda的cudatoolkit是Nvidia的cudatoolkit的一个子集，是包含了一部分支持pytorch、tensorflow的内容。所以想安装完整版本的cudatoolkit，有两种选择：

-   要么在conda外面安装完整的Nvidia cudatoolkit，可以参考[https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)。
-   要么就是在conda中安装conda-forge的cudatoolkit-dev，参考[https://anaconda.org/conda-forge/cudatoolkit-dev](https://anaconda.org/conda-forge/cudatoolkit-dev)，例如：conda install -c conda-forge cudatoolkit-dev=10.1即可。
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E5%AE%89%E8%A3%85cudatoolkit.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/NVCC路径.png)
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/NVCC%E8%B7%AF%E5%BE%84.png)

