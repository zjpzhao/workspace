#learning/GNN 

---
# 两种GNN
GNN可以在两种环境中进行训练：直推式学习（transductive learning）和归纳式学习（inductive learning）[^1]。
- transductive learning：在固定图上训练网络，不需要泛化到看不见的数据。
- inductive learning：为了推广到看不见的节点和图，网络在每次迭代中都在不同的图上进行训练。


[^1]:Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. In Advances in neural information processing systems. 1024–1034.

“Inductive learning”意为归纳学习，“Transductive learning”意为直推学习。两者的区别就体现在你所说的对于unseen node的处理。
unseen node指测试集出现了训练集未学习过的节点，即图结构（拉普拉斯矩阵）发生了变化。
GCN由于本质是频域卷积，一次卷积更新所有节点，计算过程涉及表征图结构的拉普拉斯矩阵，所以一旦出现了没有见过的图结构，拉普拉斯矩阵随之变化，以前训练好的基于原图结构的模型也就失效了。
GAT是图卷积在空域的表现形式，这使得其能够逐节点运算实现“卷积”，虽然也用到拉普拉斯矩阵信息，计算过程却已经脱离了拉普拉斯矩阵的束缚，其训练目标是中心节点与邻居节点的“聚和”关系，所以就算出现了unseen node，图结构改变了，训练好的“聚和”关系仍然能够适用，所以是一种Inductive learning方法。
某种意义上来说，GCN是一种考虑了整体图结构的方法；而GAT一定程度上放弃了整体结构，这使得其能够完成Inductive任务[^2]。 

[^2]: https://www.zhihu.com/question/409415383/answer/1361505060  

# 关键子图检测
原始图→采样编码→子图→强化筛选→关键子图→重构→骨架图

>**WWW 2021**
>SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism

# 更好的图结构
## 结构学习信息瓶颈理论
结构学习信息瓶颈可以学习更好的图结构，提高表征质量；结构学习信息平静对结构扰动具有极强的鲁棒性
>**AAAI 2022**
>Graph Structure Learning with Variational Information Bottleneck

## 低失真的结构表征
图结构嵌入欧氏空间会产生严重失真，引入黎曼流形来代替欧氏空间
>**ICDM 2021**
>ACE-HGNN: Adaptive Curvature Exploration Hyperbolic Graph Neural Network

## 加速
所提出的方法FastGCN比原始GCN和GraphSAGE快得多，同时保持了可比的预测性能。GraphSAGE是GCN对于大型和密集图表（例如Reddit）的实质性改进，尽管对于较小的图表（Cora和Pubmed），GCN的训练速度更快。FastGCN是最快的。
>**ICLR 2018** (770 Citations)
>FASTGCN: FAST LEARNING WITH GRAPH CONVOLU- TIONAL NETWORKS VIA IMPORTANCE SAMPLING

## Co-Design
本文的GNN动态剪枝算法：*Algorithm 1 GNN algorithm with dynamic pruning.* 可以参考一下
>**DAC 2021**
>DyGNN: Algorithm and Architecture Support of Dynamic Pruning for Graph Neural Networks

## 参数有效性分析
随着训练的推进，GNN 前面节点的 embedding 层越来越没用，但边的表示一直对最后的预测准确率有很大影响。通过实验我们可以发现，GNN 中边的表示，以及信息传递和聚合，都是很重要的；其它诸如图注意力、节点表示，都可有可无。于是作者只留下两个结构：Edge encoder 用来构建边的表示，Graph Soft Counter layer 用来做信息传递和聚合。
>**arXiv**
>GNN IS A COUNTER? REVISITING GNN FOR QUESTION ANSWERING

---

## 深度GNN
gcn 增加深度会降低模型效果主要是因为过度平滑的问题。现在解决这个问题的方法主要就是 skip-connection 的方法，其中包括残差网络。推荐几篇论文：
1. DeepGCNs: Can GCNs Gobas Deep as CNNs? 这篇论文主要讨论了 GCN 的深度问题，文中才用了 ResGCN，DenseGCN 和 Dilation 等方法，最后效果比较明显。网络层数可以达到 56 层，并且有 3.7 个点的提升
2. Deep insights into Graph Convolution Networks for Semi-supervised Learning. 这篇论文只看前面对于过度平滑的分析即可。
3. Representation learning on graphs with jumping knowledge networks. 这篇论文建立一个相对比较深的网络，在网络的最后当一个层聚合器来从所有层的输出中进行选择，来抑制 noise information 的问题。

> 参考[网站](https://www.codetd.com/article/11303197)

# GAT
## 注意力机制
在归纳节点分类问题中，GaAN可以优于GAT以及其他具有不同聚合器的GNN模型。原论文：GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs
在文章*fuseGNN*[^3]的图10中得知：
GCN和GAT都可以用作Inductive和Transductive
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/GCN%E5%92%8CGAT%E9%83%BD%E5%8F%AF%E4%BB%A5%E7%94%A8%E4%BD%9CInductive%E5%92%8CTransductive.png)

[^3]:fuseGNN: Accelerating Graph Convolutional Neural Network Training on GPGPU




The GNNs can be trained in two settings: transductive learning and inductive learning [6]. The former one trains the network on a fixed graph and generalizing to unseen data is not required. In inductive learning, in order to generalize to unseen nodes and graphs, the network is trained on a different graph in each iteration.

Although the trained ML models can infer average power for a new workload on the same design, a new ML model must be trained for each new design encountered.

The goal of such graph-based semi-supervised learning problems is to classify the nodes in a graph using a small subset of labeled nodes and all the node features.（from 'Attention-based Graph Neural Network for Semi-supervised Learning'）

---

Attention的引入目的：在处理局部信息的时候同时能够关注整体的信息，不是用来给参与计算的各个节点进行加权的，而是表示一个全局的信息并参与计算。**对于一个样本来说只利用邻域内的样本计算注意力系数和新的表示，即仅将注意力分配到节点的一阶邻居节点集上**。


GAT模型的特点
自动学习节点之间互相的影响度
计算高效：self-attention层的操作可以在所有的边上并行，输出特征的计算可以在所有顶点上并行。没有耗时的特征值分解。单个的GAT计算F ′ F'F 
′
 个特征的时间复杂度可以压缩至$O(|V|FF'+|E|F')$，F是输入的特征数，|V|和|E|是图中顶点数和边数。复杂度与Kipf & Welling, 2017的GCN差不多。
尽管 multi-head 注意力将存储和参数个数变为了K倍，但是单个head的计算完全独立且可以并行化。
鲁棒性更强：和GCN不同，本文的模型可以对同一个 neighborhood的node分配不同的重要性，使得模型的capacity大增。
注意力机制以一种共享的策略应用在图的所有的边上，因此它并不需要在之前就需要得到整个图结构或是所有的顶点的特征（很多之前的方法的缺陷）。因此GAT 也是一种局部模型。也就是说，在使用 GAT 时，无需访问整个图，而只需要访问所关注节点的邻节点即可，解决了之前提出的基于谱的方法的问题。因此这个方法有几个影响：
图不需要是无向的，可以处理有向图（若$j\to i$不存在，仅需忽略$\alpha_{ij}$即可）
可以直接应用到 inductive learning：包括在训练过程中在完全未见过的图上评估模型的任务上。

原文链接：https://blog.csdn.net/yyl424525/article/details/100920134

