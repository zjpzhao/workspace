t-distributed stochastic neighbor embedding（T分布随机邻域嵌入）
原作博客：https://lvdmaaten.github.io/tsne/

For the standard t-SNE method, implementations in Matlab, C++, CUDA, Python, Torch, R, Julia, and JavaScript are available.

用CUDA加速的t-SNE：https://github.com/CannyLab/tsne-cuda

GNN模型学习到的节点嵌入向量可以用t-SNE工具进行可视化，可以直观地找出预测结果的成功程度。


```latex
@article{chan2019gpu,
  title={GPU accelerated t-distributed stochastic neighbor embedding},
  author={Chan, David M and Rao, Roshan and Huang, Forrest and Canny, John F},
  journal={Journal of Parallel and Distributed Computing},
  volume={131},
  pages={1--13},
  year={2019},
  publisher={Elsevier}
}
```

关于用法
https://medium.com/analytics-vidhya/using-t-sne-for-data-visualisation-8a83f46fbad3
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
# load dataset
data = datasets.fetch_openml(
    'Fashion-MNIST',
    version=1,
    return_X_y=True
)
# data returns a tuple (features, target)
features, target = data
target = target.astype(int)
# reshape the features for plotting image
image = features.iloc[0].values.reshape(28, 28)
plt.imshow(image, cmap='gray')
# plt.savefig('image.png', bbox_inches='tight')
# dimensionality reduction using t-SNE
tsne = manifold.TSNE(n_components=2, random_state=42)
# fit and transform
mnist_tr = tsne.fit_transform(features[:30000])
# transformed_data is a 2D numpy array of shape (30000, 2)
# create dataframe
cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
                       data=np.column_stack((mnist_tr, 
                                            target.iloc[:30000])))
# cast targets column to int
cps_df.loc[:, 'target'] = cps_df.target.astype(int)
cps_df.head()
clothes_map = {0:'T-shirt/top',
               1: 'Trouser',
               2: 'Pullover',
               3: 'Dress',
               4: 'Coat',
               5: 'Sandal',
               6: 'Shirt',
               7: 'Sneaker',
               8: 'Bag',
               9: 'Ankle Boot'}
# map targets to actual clothes for plotting
cps_df.loc[:, 'target'] = cps_df.target.map(clothes_map)
cps_df.target.value_counts().plot(kind='bar')
grid = sns.FacetGrid(cps_df, hue="target", height=8)
grid.map(plt.scatter, 'CP1', 'CP2').add_legend()
```