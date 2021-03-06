---
id: 7ccc909f-c126-4aa4-956f-4cc6705ba1ac
---
#code/python #learning/ParallelComputing/multithread #learning/ParallelComputing/multiprocessing

```python
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
import deepdish as dd
import random
import struct
from struct import *
from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import multiprocessing
from functools import partial
import threading
import copy

MODEL_PATH="models/mnist_0.99.pkl"
INJECT_NUM=3000
BATCH_SIZE=10000
EPOCH=100

#bit flip
def bitflip2(x, pos):
    #pack用于将数据转换成字节流
    fs = pack('f', x)
    #unpack用于将字节流转换成Python数据类型
    bval = list(unpack('BBBB', fs)) #得到输入数据的二进制表示
    # print(bval)
    [q, r] = divmod(pos, 8)  #divmod(a,b)计算a除以b的商和余数
    #异或（不同结果为１,相同结果为０），通过与１异或实现比特翻转
    bval[q] ^= 1 << r  #00000001按位左移r个单位
    fs = pack('BBBB', *bval)
    fnew = unpack('f', fs)
    return fnew[0]

def go(each,iter,model,test_x):
    model_dict=model.state_dict()##beifen 
    model_dict_backup=copy.deepcopy(model_dict)
    print(f"第{iter}个filter，注入错误第{each}次")
    rand_inject_channel = random.randint(0, filter_channel - 1)
    rand_inject_filter_x = random.randint((filter_size - 1)/2, filter_size - 1)
    rand_inject_filter_y = random.randint((filter_size - 1)/2, filter_size - 1)
    data2flip=float(filters[iter][rand_inject_channel][rand_inject_filter_x][rand_inject_filter_y])
    pos = random.randint(0, 31)
    result = bitflip2(data2flip,pos)
    model_dict[layer_name][iter][rand_inject_channel][rand_inject_filter_x][rand_inject_filter_y]=result
    model.load_state_dict(model_dict)
    model.eval()
    predict_y = model(test_x[0:].float()).detach()
    predict_ys_after = torch.argmax(predict_y, axis=-1)
    count = 0
    if predict_ys_after[0].item()!=7:	#选的这张图是第7类的
        count = 1
    model.load_state_dict(model_dict_backup)
    return count

model=torch.load(MODEL_PATH)
model_dict=model.state_dict()########################### which to modify
layer_name_list=(model_dict.keys())
test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
for idx, (test_x, test_label) in enumerate(test_loader):
    test_x=test_x[0,:,:,:]
    test_x=test_x[np.newaxis,:]

for layer_name in layer_name_list:
    if layer_name.startswith('conv')==False or layer_name.endswith('.weight')==False:
        continue
    filters=model_dict[layer_name]
    filter_num=filters.shape[0]
    filter_channel=filters.shape[1]
    filter_size=filters.shape[2]
    if filter_size == 1:
            continue
    print(layer_name)
    filters_wrongnum=[]
    filters_rank=[] #by importance desc
	
	#inject into per filter INJECT_NUM times
    for iter in range(filter_num): 
        wrongnum=0
        par = partial(go,iter=iter,model=model,test_x=test_x)
        cores = multiprocessing.cpu_count()
        p = multiprocessing.Pool(processes=cores)
        l = [each for each in range(INJECT_NUM)]         
        wrongnum=np.array(p.map(par,l)).sum()
        print(f"第{iter}个filter，{wrongnum}")
        filters_wrongnum.append(wrongnum)

    new_filters_wrongnum=filters_wrongnum[:]
    new_filters_wrongnum.sort(reverse=True)
    print(filters_wrongnum)
    print(new_filters_wrongnum)
    for item in new_filters_wrongnum:
        filters_rank.append(filters_wrongnum.index(item))
    print(filters_rank)
```