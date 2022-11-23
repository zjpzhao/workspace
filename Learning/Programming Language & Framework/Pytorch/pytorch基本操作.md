`to('cuda')` `.cuda()`

`to('cuda:1')` `.cuda(device=1)`

`to('cpu')` `.cpu()`

加速
return DataLoader(self.mnist_test, batch_size=50,num_workers=14,prefetch_factor =2)