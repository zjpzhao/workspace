
# 输入相关
- **CUDA**: NVIDIA. 2011. CUDA C/C++ SDK Code Samples. http://developer.nvidia.com/cuda-cc-sdk-code-samples
- **Polybench**: Scott Grauer-Gray, Lifan Xu, Robert Searles, Sudhee Ayalasomayajula, and John Cavazos. 2012. Auto-tuning a high-level language targeted to GPU codes. In Innovative Parallel Computing (InPar), 2012. IEEE, 1–10
- **Rodinia**: Shuai Che, Michael Boyer, Jiayuan Meng, David Tarjan, Jeremy W Sheaffer, Sang-Ha Lee, and Kevin Skadron. 2009. Rodinia: A benchmark suite for heterogeneous computing. In 2009 IEEE International Symposium on Workload Characterization (IISWC). Ieee, 44–54.
- **AxBench**: Amir Yazdanbakhsh, Divya Mahajan, Hadi Esmaeilzadeh, and Pejman Lotfi-Kamran. 2017. Axbench: A multiplatform benchmark suite for approximate computing. IEEE Design & Test 34, 2 (2017), 60–68.

CTA：协调线程数组（Block的术语）
线程的动态指令数（*DI*）决定了GPGPU应用的错误弹性：怎么通过工具统计？——原文采用GPGPU-sim

# 图结构优化角度
下游任务冗余信息——如何设计GNN的聚合函数（参考YEF2022北航工作：Graph Structure Learning with Variational Information Bottleneck）

# 故障注入工具NVBITFI
nvcc编译cuda代码的时候，Host和Device是分开进行的，nvcc --keep选项可以保存编译.cu的过程文件（如.ptx, .cubin等），PTX是每一个线程都需要执行的，我猜测需要执行该PTX的线程号是通过链接.cubin文件而分配的。具体需要参考和探索CUDA Binary Utilities。


原来的想法是对每一个线程做一层图，主要是**太大了**只能到指令级别（也可以做出来用作baseline，然后和裁剪优化后的进行对比讲故事），现在可以在不同level裁剪之后得到具有代表性的点进行建图，提特征和学习，然后学出来的model跨模型应用到别的应用上进行测试。
另外结合多输入或者从GNN优化角度也可以一下，不过工作就有点多了。（待定）
主要还是考虑裁剪凝练图，这样才能体现出学习局部特征的效果（当然得看具体结果），目前的想法都是未完全考虑实验工具和实现难度的。
5个level修剪来减少冗余——让图神经网络带标签的点少一些，然后结合注入错误的时间和GNN的准确率进行Trade off，另外也是优化图神经网络的基本结构，去除了一些冗余信息
1. CTA级别：修剪冗余块，选取代表块
2. 线程级别：修剪冗余线程，内核中的许多线程具有类似的错误恢复特征，减少冗余的故障注入点.
3. 指令级别：修剪冗余指令，不需要考虑所有指令，并且线程之间复制的子集只能考虑一次
4. 循环级别：修剪冗余循环，我怎么能知道循环多少次才能代表所有循环，MICRO比较生猛，选了一堆benchmark，然后循环不同次数看多少次的时候跟FI的结果分布比较接近了（也就是循环次数收敛值），然后这些benchmark求了个平均大概是7次，以后每个benchmark都7次（我觉得吧，循环总次数也需要考虑进去，所以说要不用百分比可能好一点？可以找一找分析循环level的文章）
5. bit级别：修剪冗余fault site——基于高位出错后果可能更严重，对于32位寄存器是这样的，但是对于pred并不是：这就是 4 位谓词系统的本质，寄存器类型 .pred 中最高的三位分别用于溢出标志、携带标志和标志标志，而最低位表示零标志。在本书中研究的应用进程的上下文中，只有零标志用于分支条件，因此我们可以自信地修剪寄存器类型.pred中的其他三个位位置。也就是说我们可以给寄存器分类，

有一个点就是这篇MICRO：用很多故障注入来给CTA和thread分别进行分类，所以用DI count来作为分类的依据可以替代“在给CTA和thread级别分类时候采用的故障注入”。对于别的level裁剪也都是用这种方法：用故障注入来找一些这个级别的分布特征来作为进行裁剪的依据（比如DI数，7次循环等等），我觉得这一点发现是他们冲MICRO成功的一个非常重要的点，但是比较粗暴，对于**跨复杂应用**的话可能还是有点担心，还需要看最终结果。

必须要先进行CTA分组然后进行thread分组（来自不同CTA的具有相同iCnt（动态指令数）的线程可能具有不同的指令）
具有不同 iCnt 的线程之间的故障注入结果分布如此相似的一个潜在原因是这些线程**共享大的相同代码块**
我觉得这篇MICRO主要还是写对benchmark的实验观察发现一些规律，具体这些规律是每个benchmark的kernel内和间的独特的写法导致的（这一部分在SUGAR里可以得到对每一个benchmark的详细的分析，而且结合了输入规模）

- [ ] 再有一个点就是polybench这类应用都很简单，可以拿简单的benchmark去跑GNN，然后在比较复杂的benchmark（像宝哥的那种）去测试，来说明是确实学到了局部特征（也就是魏老师说的子图）
- [ ] 注入level——**指令级别**or**位级别**？我觉得做裁剪的话可以到位级别。

# 依赖
依赖特征——结合到图神经网络的结构特性上：这里可以和我们传统的工作进行对比（有个图）
依赖涉及到建图层次：
- 线程内：寄存器依赖+控制依赖
- 线程间：共享内存
- kernel间依赖：一个kernel需要用到另外一个kernel的结果（比如3mm的kernel3需要用到前两个kernel计算的矩阵结果）

# 评价
跨应用精度（可以像DATE GLAIVE似的给应用分个类：比如DI敏感型啥的，结合考虑不同应用kernel的依赖关系，这样说平均准确率的时候可以把低一点的平均回来）
参考那些经典的我们设计的指标，方便比较


novelty


问题：
**整体：故障主注入点的选择，泛化能力，建图**
CPU-GPU 交互行为的表征
想法：弹性相似性融入图
- [ ] 跨应用。和挑战对应（chap 3&4），局部特性和共有的关系，学的是通用的知识，benchmark独特的特性套在学到的知识上（比较容易让人接受）——辅助工具
- [ ] 剪枝对于错误传播，对于PTX建一个图，不同线程在这一个图上的表征（激活顶点/边）
- [ ] 泛化：应用特点体现在图里
- [ ] 相似性和建图的关系
- [ ] 关键字图——剪枝依据做解释，关键点和边对错误传播的影响。


next：错误传播——图
图的层次
线程德相似性体现在选点还是建图上

本工作结束以后：GNN+FI+CNN→单调性binfi结合剪枝