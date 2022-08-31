# NVCC
## 编译中断和续编
编译过程暴露中间文件（如PTX）后继续编译过程
1. 通过-dryrun选项输出编译子命令
```bash
nvcc -dryrun -arch=sm_60 -o ${EXECUTABLE} ${CUFILES} --keep 2>dryrun.out
```
2. 查看dryrun.out文件中的编译子命令
```bash
#$ _NVVM_BRANCH_=nvvm
#$ _SPACE_= 
#$ _CUDART_=cudart
#$ _HERE_=/usr/local/cuda/bin
#$ _THERE_=/usr/local/cuda/bin
#$ _TARGET_SIZE_=
#$ _TARGET_DIR_=
#$ _TARGET_DIR_=targets/x86_64-linux
#$ TOP=/usr/local/cuda/bin/..
#$ NVVMIR_LIBRARY_DIR=/usr/local/cuda/bin/../nvvm/libdevice
#$ LD_LIBRARY_PATH=/usr/local/cuda/bin/../lib:/usr/local/cuda-7.0/bin/lib64:/usr/local/cuda/lib64
#$ PATH=/usr/local/cuda/bin/../nvvm/bin:/usr/local/cuda/bin:/home/jlu/bin:/home/jlu/.local/bin:/home/jlu/anaconda3/bin:/home/jlu/anaconda3/bin:/usr/local/maven/bin:/usr/java/jdk1.8.0_221/bin:/usr/java/jdk1.8.0_221/jre/bin:/usr/local/hadoop/bin:/usr/local/hadoop/sbin:/usr/local/spark/bin:/usr/local/spark/sbin:/usr/local/scala/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/snap/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/snap/bin:/usr/local/hbase/bin:/usr/local/zookeeper/bin:/usr/local/Jobsubmitter/bin:-Xmx2048m -XX:MetaspaceSize=1024m -XX:MaxMetaspaceSize=1524m -Xss2m/bin:/usr/local/cuda/bin
#$ INCLUDES="-I/usr/local/cuda/bin/../targets/x86_64-linux/include"  
#$ LIBRARIES=  "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"
#$ CUDAFE_FLAGS=
#$ PTXAS_FLAGS=
#$ rm 2mm_dlink.reg.c
#$ gcc -D__CUDA_ARCH__=600 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=89 -include "cuda_runtime.h" -m64 "2mm.cu" -o "2mm.cpp1.ii" 
#$ cicc --c++14 --gnu_version=70500 --allow_managed   -arch compute_60 -m64 -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "2mm.fatbin.c" -tused -nvvmir-library "/usr/local/cuda/bin/../nvvm/libdevice/libdevice.10.bc" --gen_module_id_file --module_id_file_name "2mm.module_id" --orig_src_file_name "2mm.cu" --gen_c_file_name "2mm.cudafe1.c" --stub_file_name "2mm.cudafe1.stub.c" --gen_device_file_name "2mm.cudafe1.gpu"  "2mm.cpp1.ii" -o "2mm.ptx"
#$ ptxas -arch=sm_60 -m64  "2mm.ptx"  -o "2mm.sm_60.cubin" 
#$ fatbinary --create="2mm.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=60,file=2mm.sm_60.cubin" "--image3=kind=ptx,sm=60,file=2mm.ptx" --embedded-fatbin="2mm.fatbin.c" 
#$ gcc -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=89 -include "cuda_runtime.h" -m64 "2mm.cu" -o "2mm.cpp4.ii" 
#$ cudafe++ --c++14 --gnu_version=70500 --allow_managed  --m64 --parse_templates --gen_c_file_name "2mm.cudafe1.cpp" --stub_file_name "2mm.cudafe1.stub.c" --module_id_file_name "2mm.module_id" "2mm.cpp4.ii" 
#$ gcc -D__CUDA_ARCH__=600 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 "2mm.cudafe1.cpp" -o "2mm.o" 
#$ nvlink --arch=sm_60 --register-link-binaries="2mm_dlink.reg.c"  -m64   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "2mm.o"  -lcudadevrt  -o "2mm_dlink.sm_60.cubin"
#$ fatbinary --create="2mm_dlink.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=60,file=2mm_dlink.sm_60.cubin" --embedded-fatbin="2mm_dlink.fatbin.c" 
#$ gcc -c -x c++ -DFATBINFILE="\"2mm_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"2mm_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=89 -m64 "/usr/local/cuda/bin/crt/link.stub" -o "2mm_dlink.o" 
#$ g++ -m64 -Wl,--start-group "2mm_dlink.o" "2mm.o"   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"  -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group -o "2mm" 
```
The PTXAS is the abbr for Ptx optimizing assembler. Usage as follows.
```bash
ptxas --help

Usage : ptxas [options] <ptx file>,...
```

Start from `ptxas -arch=sm_60 -m64  "2mm.ptx"  -o "2mm.sm_60.cubin" `, the ptxas continue to compile .ptx files to .cubin files. After a few compiling insts, the exe file for .cu is generated. Accordingly, we can extract the insts below the `ptxas` inst to another dryrun file (we call dryrunpruned), after modifying the ptx file, we run the remaining insts in dryrunpruned file.
