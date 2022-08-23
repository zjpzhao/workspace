
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

## NVCC--keep选项
nvcc编译cuda代码的时候，Host和Device是分开进行的，nvcc --keep选项可以保存编译.cu的过程文件（如.ptx, .cubin等），PTX是每一个线程都需要执行的，~~我猜测需要执行该PTX的线程号是通过链接.cubin文件而分配的~~，tid blockid都可以在ptx中找到。具体需要参考和探索CUDA Binary Utilities。


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

1. 想怎么建图——多线程传播的表示，想清楚之后
2. 弹性相似在图上的表示
3. 泛化解释

challenge——GPU弹性和错误传播（GPU-Trident团队、Nie Bin）
预测问题用图解决宏观上的好处：准，自动化（不提泛化，看DATE21的说法）
线程的相似性在GNN上表示如果做得好可以作为一个用GNN相对于解决传统问题好在哪的点——insight

引DATE和DATE引的文章
解决了哪些挑战

next：错误传播——图
图的层次
线程的相似性体现在选点还是建图上->建图的剪枝上

后续工作：GNN+FI+CNN→单调性binfi结合剪枝






# 需要思考和解决的问题
- [x] GAT能做inductive吗——能
- [x] 能否将注意力融入GraphSage *or* 直接采用GAT——GAT
- [x] 选用哪种图框架：DGL *or* PyG (参考https://github.com/cornell-zhang/GLAIVE)——DGL
- [x] GNN Train前需要构建自定义数据集，DGL *or* PyG 方便些？——用networkx或者csv文件构建图然后DGL直接导入即可
- [ ] 注错层次在*PTX*/*PTX plus*/*SASS* ？
- [ ] 在哪一章的那一块介绍三种软错误（在Fault model行么）
- [ ] 要修改GAT本身么（以适应我们的背景）：添加注意力阈值，小于这个数的就不参与message passing了


# 文章细节翻译
from the ... perspective
The learned W matrix is commonly shared and updated by all nodes, indicating a weight sharing philosophy similar to convolutional neural networks in DL.
这里体现的是类似于深度学习中卷积神经网络的权值共享的哲学

启发式的
我们考虑了线程、指令、比特级别的弹性相似性，遵循这篇文章提出的方法对程序进行建图

三种软错误：
GPGPU Application Resilience Profile. For each fault injection experiment, there are three possible outcomes:
- masked output: the application output is identical to that of fault-free execution.
- silent data corruption (SDC) output: the fault injection run exits successfully without any error, but the output is incorrect. 
- other: the fault injection run results in a crash or hang.

Recent commercial GPUs protect the DRAM, shared memory, cache, and register file with single-error-correction double-error-detection (SEC-DED) ECC [17, 42]. Therefore, this study only considers errors occurring in the functional units of GPGPUs (e.g., arithmetic logic units and load-store units). Soft errors may manifest as single or multiple-bit flips. In this paper, we adopt the single-bit flip model as it is typically considered the most common error type in microprocessors [7, 33, 41]. Moreover, multiple-bit flips generally have a similar effect on program resiliency as single-bit flip [9, 17, 37]. Therefore, the single-bit flip model is sufficient to capture the view of the resilience characteristics.

We assume that register files and other components such as caches and memory are protected by ECC (which is the case in almost all GPUs). We simulate commonly occurring computation-related errors due to transient faults (known as soft errors) in ALUs/LSUs. These faults can lead to wrong ALU output which would then be stored in destination registers, or corrupted variables loaded by an LSU. This erroneous computing operation is what we emulate by injecting faults directly to destination register values. This is a standard experimental methodology for GPGPU reliability studies [18], [24], [33]–[35]. 

In this paper, we consider transient hardware faults that occur in the computational elements of the GPU, including architectural registers and functional units, and affect the program’s execution. We assume these faults manifest as a single bit flip. Many studies [4], [5], [11], [31] have shown that there is little difference between the SDC probability of single and multiple bit flips. Moreover, previous work in this area [15], [20], [26], [36] also uses the single-bit flip model. We do not consider faults in the GPU’s control logic, nor do we consider faults in the instructions’ encoding. We also do not consider faults in the memory or caches, as we assume that these are protected with error correction codes (ECC) - this is the case for most modern GPUs used in HPC applications. *However, an error can propagate to memory, if an erroneous value is stored by a store instruction into memory, resulting in subsequent loads being faulty (these faults are considered).* Finally, similar to most other work in the area [7]–[10], [12], [18], we assume that the program does not jump to arbitrary illegal addresses due to faults during the execution, as this can be detected by control-flow checking techniques [28]. However, the program may take a faulty legal branch (the execution path is legal but the branch direction is wrong due to faults propagating to the branch condition).

The fault injection methodology used here closely follows the one used in [24], [36]: we flip a bit at a destination register identified by the thread id, the instruction id, and a bit position. We perform our reliability evaluations on GPGPU-Sim [37] with PTXPlus mode. GPGPU-Sim is a widely-used cyclelevel GPU architectural simulator, and its PTXPlus mode provides a one-to-one mapping of instructions to actual ISA for GPUs [36], [37]. Any fault injection tool or technique. (e.g., SASSIFI [18] or NVBitFI [38]) can be used for evaluating the application reliability, i.e., the technique presented in this paper does not depend on GPGPU-Sim.
	引用自：*Enabling Software Resilience in GPGPU Applications via Partial Thread Protection*

The proposed methodology can be readily extended to multi-bit fault models [39]

我们提出的方法可以很容易地扩展到多比特的错误模式上

we construct the feature vector corresponding to a fault site, denoted as 𝑣, formulated in Equation 5: 𝑣 = ⟨𝐹instruction-type, 𝐹bit-position, 𝐹bit-flip-direction, 𝐹slice, 𝐹shared, 𝐹commonality ⟩

