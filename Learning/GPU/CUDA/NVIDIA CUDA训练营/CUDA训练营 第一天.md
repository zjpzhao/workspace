## 课程说明
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/CUDA2023%E8%AE%AD%E7%BB%83%E8%90%A5%E8%AF%BE%E7%A8%8B%E8%AF%B4%E6%98%8E.png)


## 实验平台
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/jestonnano%E5%8F%82%E6%95%B0.png)

Jeston Nano's Compute Capability: 5.3
SM53
[CUDA GPUs - Compute Capability | NVIDIA Developer](https://developer.nvidia.com/cuda-gpus)

# Linux


## Ubuntu的文件管理：目录结构
![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Ubuntu%E7%9A%84%E6%96%87%E4%BB%B6%E7%AE%A1%E7%90%86%E7%9B%AE%E5%BD%95%E7%BB%93%E6%9E%84.png)


![image.png](https://zjpimage.oss-cn-qingdao.aliyuncs.com/Ubuntu%E6%9D%83%E9%99%90%E7%AE%A1%E7%90%86.png)
这里的d就是指这是一个文件夹，如果不是d是-，则说明是一个文件而非文件夹。

对于权限修改，有字母法和数字法
r  (read) -> 4
w (write) -> 2
x  (excute) -> 1


内容比较多的文本文件建议用more来代替cat看，好处是分页，滚动查看（要不然行会很多，占用屏幕空间）


Makefile高阶教程
[Makefile详细教程\_扫地的小何尚的博客-CSDN博客](https://blog.csdn.net/kunhe0512/article/details/128623790)

make时文件路径下如果有一个文件叫clean怎么办?
在makefile里写一行.PHONY: clean，声明clean是个操作。./PHONY表示声明，有声明的会按照声明执行，没有声明的会寻找makefile同级的文件


## CUDA
**APOD**: Assess, Parallelize, Optimize, Deploy (评估、并行化、优化、部署) [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#assess-parallelize-optimize-deploy)





VectorAdd：
```C
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector A
  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}
```