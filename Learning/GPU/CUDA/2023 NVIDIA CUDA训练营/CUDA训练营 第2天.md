cudaMemcpy已经包含了同步过程，就不需要显式调用sycn了

用空间换时间
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%BA%8C%E7%BB%B4grid%E5%92%8C%E4%BA%8C%E7%BB%B4block.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E7%AE%97%E4%B8%80%E7%BB%B4%E5%9D%90%E6%A0%87.png)

cudaMallocHost与malloc不同的一点是：避免内存虚拟化技术从内存移到disk


## 错误检测和事件

可以查看Cuda error的四个函数：

```C++
__host__​__device__​const char*     cudaGetErrorName ( cudaError_t error )
Returns the string representation of an error code enum name.  

__host__​__device__​const char*     cudaGetErrorString ( cudaError_t error )
Returns the description string for an error code.  

__host__​__device__​cudaError_t     cudaGetLastError ( void )
Returns the last error from a runtime call.  

__host__​__device__​cudaError_t     cudaPeekAtLastError ( void )
Returns the last error from a runtime call.  
```


![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%B8%80%E4%B8%AACUDA%E8%BF%90%E8%A1%8C%E6%97%B6%E6%A3%80%E6%B5%8B%E5%87%BD%E6%95%B0%E5%AE%9E%E4%BE%8B.png)

![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/%E4%B8%80%E4%B8%AACUDA%E8%BF%90%E8%A1%8C%E6%97%B6%E6%A3%80%E6%B5%8B%E5%87%BD%E6%95%B0%E5%AE%9E%E4%BE%8B2.png)


![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Check%E5%87%BD%E6%95%B02.png)


等待事件完成，设立flag：

```C++
cudaError_t cudaEventSynchronize(cudaEvent_t event);//阻塞（事件完成才记录）
cudaError_t cudaEventQuery(cudaEvent_t event);//非阻塞（事件没完成也会记录）
```


[(含代码)利用GPU版的Numpy---cuNumeric加速Python数值计算\_扫地的小何尚的博客-CSDN博客](https://blog.csdn.net/kunhe0512/article/details/128908418)