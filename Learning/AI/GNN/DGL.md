DGL在内部将SciPy矩阵和NetworkX图转换为张量来创建图。因此，这些构建方法并不适用于重视性能的场景。

DGL使用一个一维的整型张量（如，PyTorch的Tensor类，TensorFlow的Tensor类或MXNet的ndarray类）来保存图的点ID，DGL称之为”节点张量”。为了指代多条边，图的edge数据结构是两个tensor构成的，DGL使用一个包含2个节点张量的元组：
edges = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])

每个节点和边可以拥有多组特征，每组特征都是一个tensor向量或者矩阵

对于加权图，用户可以将权重储存为一个边特征

对于转换成双向边的图有两种方法：
1. newg = dgl.add_reverse_edges(g)
2. bg = dgl.to_bidirected(g) 适用于原来的图是无向图的情况

DGL支持使用 32 位或 64 位的整数作为节点ID和边ID。节点和边ID的数据类型必须一致。如果使用 64 位整数， DGL可以处理最多 $2^{63}-1$ 个节点或边。不过，如果图里的节点或者边的数量小于 $2^{63}-1$ ，用户最好使用 32 位整数。 这样不仅能提升速度，还能减少内存的使用。DGL默认使用int64，但是建议用int32构建小图：
```python
g32 = dgl.graph(edges, idtype=th.int32)  # 使用int32构建图
>>> g32.idtype
torch.int32
```

## GNN可视化
A DGL graph can be converted to a `networkx` graph, so to utilize its rich functionalities such as visualization.
```python
# visualization via nx
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
f = plt.figure()
subax1=plt.subplot(121)
nx_g1 = g.to_networkx().to_undirected()
pos1 = nx.kamada_kawai_layout(nx_g1)
nx.draw(nx_g1, pos1, with_labels=True, node_color=[[.7, .7, .7]])

subax2 = plt.subplot(122)
nx_g2 = g.to_networkx()#.to_undirected()
pos2 = nx.kamada_kawai_layout(nx_g2)
nx.draw(nx_g2, pos2, with_labels=True, node_color=[[.7, .7, .7]])
f.savefig("graph.png")
```
生成训练轮次可视化视频：
https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/basic_tasks/2_gnn.ipynb

其他DGL常用方法详见：https://zhuanlan.zhihu.com/p/536737592
