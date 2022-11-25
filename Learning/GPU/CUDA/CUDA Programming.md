- CUDA=Compute Unified Device Architecture，统一计算设备架构。
- 由 NVIDIA 提出的通用并行计算平台和编程模型 
- CUDA 软件环境使用 C++ 作为高级编程语言
- 也支持其他编程语言和接口：Fortran, Python, OpenACC
- 具有灵活性和可扩展性
- Single Program Multiple Data ，单程序多数据单程序多数据
- 不同的进程 / 线程运行同一个程序源代码（ SP ），单程序多数据但是分别使用不同的输入数据进行各自的计算（ MD ）
	- 不同进程 / 线程相互独立，单程序多数据没有执行顺序的要求
- 常用的并行编程模型多数采用 SPMD 模式
- 例如： CUDA ，单程序多数据 MPI ，单程序多数据 openMP


## SPMD, SIMT, 和SIMD
- SPMD：GPU的编程模型
- SIMT：GPU的执行方式
- SIMD：GPU计算单元的处理方式（硬件计算部件）

SIMT的优点
- 编程灵活：任意大小的工作量vs任意硬件SIMD宽度
- 每个线程可以被单独对待：分支控制
- Warp细节对程序员透明：Warp大小、哪些线程由同一个warp执行

## Kernel
- CUDA C++中并行函数以kernel的形式定义
- 当一个kernel被调用时，单程序多数据N个不同的thread共同将该kernel并行执行N次
- 由__global__限定词声明，单程序多数据调用时由<<<...>>>执行配置句法说明执行该kernel的thread数目
```C
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
int i = threadIdx.x;
C[i] = A[i] + B[i];
}
int main()
{...
// Kernel invocation with N threads
VecAdd<<<1, N>>>(A, B, C);
...
}
```

<<<1, N>>>表示这个kerne用了1个CTA，每个含有N个thread。
举个例子：
```C
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{ 
	int i = threadIdx.x;
	int j = threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}
int main()
{ ...
	// Kernel invocation with one block of N * N * 1 threads
	int numBlocks = 1;
	dim3 threadsPerBlock(N, N);
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
	...
}
```

## Thread
线程层次
- threadIdx为三维向量
- 一个thread block由大量thread组成
- 二维block:
- Block size: (Dx,Dy), thread(x,y) 的ID=x+yDx
- 三维block:
- Block size: (Dx,Dy,Dz), thread(x,y,z) 的
ID=x+yDx+zDxDy

## Block
- 线程以block为单位分配到GPU处理核心（SM）中
- SM内部计算和存储资源有限
- 在目前的NVIDIA GPU中，单程序多数据一个block中最多包含1024个thread
- 同一kernel中的所有block大小都一致
- 一个kernel可以被大量block执行
- 总线程数=block数量\*一个block中的线程数量
- 通常block数量大于GPU SM数量
- Block组成一维、二维、三位的grid
```C
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N){
		C[i][j] = A[i][j] + B[i][j];
	}
	
int main()
{
	...
	// Kernel invocation
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
	...
}
```

- 在硬件中，单程序多数据每个block独立执行
- 不同block可以并行或串行执行
- 不同block可以按照任意顺序被调度和执行
- 保证了程序的可扩展性（scalability）

![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/block%20execute%20in%20SM.png)

- 同一个block中的thread可以通过shared memory共享数据
	- Shared memory是由程序员控制的高速访问存储空间，单程序多数据访问速度类似L1 cache
- 同一个block中的thread可以进行同步
	- 通过在kernel中调用__syncthreads()函数设置同步点
	- 同步表示必须等待所有thread均到达该路障后才可以继续执行之后的代码