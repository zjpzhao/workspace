#learning/GNN 

---
# 两种GNN
GNN可以在两种环境中进行训练：直推式学习（transductive learning）和归纳式学习（inductive learning）[^1]。
- transductive learning：在固定图上训练网络，不需要泛化到看不见的数据。
- inductive learning：为了推广到看不见的节点和图，网络在每次迭代中都在不同的图上进行训练。

[^1]:Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. In Advances in neural information processing systems. 1024–1034.

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
在文章*fuseGNN*[^2]的图10中得知：
GCN和GAT都可以用作Inductive和Transductive
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/GCN%E5%92%8CGAT%E9%83%BD%E5%8F%AF%E4%BB%A5%E7%94%A8%E4%BD%9CInductive%E5%92%8CTransductive.png)

[^2]:fuseGNN: Accelerating Graph Convolutional Neural Network Training on GPGPU


