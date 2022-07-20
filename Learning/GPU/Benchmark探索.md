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
(base) s3090@b216:~/zjp/experiments/axbench-gpu$ ./uun.sh make convolution
Start making convolution...
./run.sh: line 50: ./log/Make.log: No such file or directory
grep: ./log/Make.log: No such file or directory
Yay! ** convolution ** has been successfully compiled.
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/error:%20libfann.so.2.png)

上面这个错误参考[网站](https://leenissen.dk/fann/wp/help/installing-fann/)，执行sudo lodconfig即可解决

blackscholes调好了好使了
convolution好使了
jmeint好使了

---

# Polybench


---

# Rodinia
由于文件夹太大了，没法上传github和gitee采用git版本控制，所以我删除了几个用不到的文件夹：opencl, openmp, data/mummergpu, data/leukocyte/

编译hotspot保留中间文件：
```bash
/usr/local/cuda/bin/nvcc -DRD_WG_SIZE_0=16 hotspot.cu -o hotspot -I /usr/local/cuda/include -L /usr/local/cuda/lib --keep
```

---

# tensorcore-microbenchmarks等
Accelsim自带的[gpu-app-collection](https://github.com/accel-sim/gpu-app-collection)里面有支持tensorcore的benchmark，下载之后用vscode打开然后在项目里搜索 `wmma::` 就知道哪些是有的了
