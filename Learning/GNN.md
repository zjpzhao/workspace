#learning/GNN 

---

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