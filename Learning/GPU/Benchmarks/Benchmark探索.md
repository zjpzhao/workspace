# 1.AxBench
## Boost library 1.79.0
–prefix
用来指定boost的安装目录，不加此参数的话默认将头文件安装在/usr/local/include/boost目录下，库文件在/usr/local/lib/目录下。
### 命令
sudo ./bootstrap.sh --with-libraries=all --show-libraries
### 回显
The Boost libraries requiring separate building and installation are:
    - atomic
    - chrono
    - container
    - context
    - contract
    - coroutine
    - date_time
    - exception
    - fiber
    - filesystem
    - graph
    - graph_parallel
    - headers
    - iostreams
    - json
    - locale
    - log
    - math
    - mpi
    - nowide
    - program_options
    - python
    - random
    - regex
    - serialization
    - stacktrace
    - system
    - test
    - thread
    - timer
    - type_erasure
    - wave
## FANN 2.2.0
make fann(2.2.0)库回显：
[sudo] password for s3090: 
[ 12%] Building C object src/CMakeFiles/floatfann.dir/floatfann.c.o
[ 25%] Linking C shared library libfloatfann.so
[ 25%] Built target floatfann
[ 37%] Building C object src/CMakeFiles/doublefann.dir/doublefann.c.o
[ 50%] Linking C shared library libdoublefann.so
[ 50%] Built target doublefann
[ 62%] Building C object src/CMakeFiles/fixedfann.dir/fixedfann.c.o
[ 75%] Linking C shared library libfixedfann.so
[ 75%] Built target fixedfann
[ 87%] Building C object src/CMakeFiles/fann.dir/floatfann.c.o
[100%] Linking C shared library libfann.so
[100%] Built target fann
Install the project...
-- Install configuration: ""
-- Installing: /usr/local/lib/pkgconfig/fann.pc
-- Installing: /usr/local/lib/libfloatfann.so.2.2.0
-- Installing: /usr/local/lib/libfloatfann.so.2
-- Installing: /usr/local/lib/libfloatfann.so
-- Installing: /usr/local/lib/libdoublefann.so.2.2.0
-- Installing: /usr/local/lib/libdoublefann.so.2
-- Installing: /usr/local/lib/libdoublefann.so
-- Installing: /usr/local/lib/libfixedfann.so.2.2.0
-- Installing: /usr/local/lib/libfixedfann.so.2
-- Installing: /usr/local/lib/libfixedfann.so
-- Installing: /usr/local/lib/libfann.so.2.2.0
-- Installing: /usr/local/lib/libfann.so.2
-- Installing: /usr/local/lib/libfann.so
-- Installing: /usr/local/include/fann.h
-- Installing: /usr/local/include/doublefann.h
-- Installing: /usr/local/include/fann_internal.h
-- Installing: /usr/local/include/floatfann.h
-- Installing: /usr/local/include/fann_data.h
-- Installing: /usr/local/include/fixedfann.h
-- Installing: /usr/local/include/compat_time.h
-- Installing: /usr/local/include/fann_activation.h
-- Installing: /usr/local/include/fann_cascade.h
-- Installing: /usr/local/include/fann_error.h
-- Installing: /usr/local/include/fann_train.h
-- Installing: /usr/local/include/fann_io.h
-- Installing: /usr/local/include/fann_cpp.h

## 使用Axbench
首先一定要sudo chmod -R 777 .
conda activate zjp_python2
(base) s3090@b216:~/zjp/experiments/axbench-gpu$ ./run.sh make convolution
Start making convolution...
./run.sh: line 50: ./log/Make.log: No such file or directory
grep: ./log/Make.log: No such file or directory
Yay! ** convolution ** has been successfully compiled.
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/error:%20libfann.so.2.png)

上面这个错误参考[网站](https://leenissen.dk/fann/wp/help/installing-fann/)，执行sudo lodconfig即可解决


注意跑的时候一定要用sudo赋权
```bash
(zjp_python2) s3090@b216[19:31:19]:~/zjp/experiments/benchmarks/axbench-gpu$ sudo ./run.sh run convolution
[sudo] password for s3090: 
#1: Collect the training data...
*********************************************
 CUDA Convolution Separable Starting... 
-------------------------------------------------------
 Input Image:  train.data/input/baboon.pgm
-------------------------------------------------------
*********************************************
#2: Aggregate the training data...
mkdir: cannot create directory ‘./train.data/output/kernel.data’: File exists
--------------------------------------------------------- 
# Total number of training data (Kernel = convolutionRowsKernel): 262144
---------------------------------------------------------
 >>>>>> Start Training for Kernel convolutionRowsKernel <<<<<<
 Do you want to perform training for this kernel?[y/N] y
#3: Read compile parameter from json file...
------------------ Compiler Parameters ------------------
Learning rate [0.1-1.0]: 0.1
Epoch number [1-10000]: 1
Sampling rate [0.1-1.0]: 0.1
Test data fraction [0.1-1.0]: 0.1
Maximum number of layers [3|4]: 3
Maximum number of neurons per layer [2-64]: 2
---------------------------------------------------------
#4: Explore different NN topologies for each kernel...
# Learning Rate:                0.100000 
# Epochs:                       1.000000 
# Sampling Rate:                0.100000 
--------------------------------------------------------- 
# Training Size: 23592
---------------------------------------------------------
#5: Find the best NN topology...
---------------------------------------------------------
# Best Topology: 17_2_1
# Best MSE:      24.0608000000
---------------------------------------------------------
#6: Replace the code with NN...
#7: Compile the code with NN...
---------------------------------------------------------
 The transformed code for ** convolution ** was successfully compiled 
---------------------------------------------------------
#8: Run the code on the test data...
---------------------------------------------------------
 CUDA Convolution Separable Starting... 
-------------------------------------------------------
 Input Image:  test.data/input/lena.pgm
-------------------------------------------------------
test.data/input/lena.pgm        *** Error: 95.20%
---------------------------------------------------------
```

blackscholes调好了好使了
convolution好使了
jmeint好使了

---

# Polybench

2MM MVT SYRK GEMM的kernel通过MICRO18（Nie Bin）的线程级剪枝后都只剩下一个有代表性的线程

---

# Rodinia
由于文件夹太大了，没法上传github和gitee采用git版本控制，所以我删除了几个用不到的文件夹：opencl, openmp, data/mummergpu, data/leukocyte/

编译hotspot保留中间文件：
```bash
/usr/local/cuda/bin/nvcc -DRD_WG_SIZE_0=16 hotspot.cu -o hotspot -I /usr/local/cuda/include -L /usr/local/cuda/lib --keep
```

---

# tensorcore相关benchmark
Accelsim自带的[gpu-app-collection](https://github.com/accel-sim/gpu-app-collection)里面有支持tensorcore的benchmark：*tensorcore-microbenchmarks*，下载之后用vscode打开然后在项目里搜索 `wmma::` 就知道哪些是有的了
