
# 输入相关
- **CUDA**: NVIDIA. 2011. CUDA C/C++ SDK Code Samples. http://developer.nvidia.com/cuda-cc-sdk-code-samples
- **Polybench**: Scott Grauer-Gray, Lifan Xu, Robert Searles, Sudhee Ayalasomayajula, and John Cavazos. 2012. Auto-tuning a high-level language targeted to GPU codes. In Innovative Parallel Computing (InPar), 2012. IEEE, 1–10
- **Rodinia**: Shuai Che, Michael Boyer, Jiayuan Meng, David Tarjan, Jeremy W Sheaffer, Sang-Ha Lee, and Kevin Skadron. 2009. Rodinia: A benchmark suite for heterogeneous computing. In 2009 IEEE International Symposium on Workload Characterization (IISWC). Ieee, 44–54.
- **AxBench**: Amir Yazdanbakhsh, Divya Mahajan, Hadi Esmaeilzadeh, and Pejman Lotfi-Kamran. 2017. Axbench: A multiplatform benchmark suite for approximate computing. IEEE Design & Test 34, 2 (2017), 60–68.
- [ ] CTA是什么？

# 图结构优化角度
下游任务冗余信息——如何设计GNN的聚合函数（参考YEF2022北航工作：Graph Structure Learning with Variational Information Bottleneck）

# GPGPU特性
block-数据局部性

# SDC的精度（近似trade off）

# Warp整合/聚类

# 故障注入工具NVBITFI
nvcc编译cuda代码的时候，Host和Device是分开进行的，nvcc --keep选项可以保存编译.cu的过程文件（如.ptx, .cubin等），PTX是每一个线程都需要执行的，我猜测需要执行该PTX的线程号是通过链接.cubin文件而分配的。具体需要参考和探索CUDA Binary Utilities。