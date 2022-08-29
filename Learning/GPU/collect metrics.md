>https://stackoverflow.com/questions/16791145/how-to-count-number-of-executed-thread-for-whole-the-cuda-kernel-execution

There are numerous ways to measure thread level execution efficiency. This answer provides a list of different collection mechanisms. Robert Crovella's answer provides a manual instrumentation method that allows for accurately collection of information. A similar technique can be used to collect divergence information in the kernel.

**Number of Threads Launched for Execution (static)**

gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z

**Number of Threads Launched**

gridDim.x * gridDim.y * gridDim.z * ROUNDUP((blockDim.x * blockDim.y * blockDim.z), WARP_SIZE)

This number includes threads that are inactive for the life time of the warp.

This can be collected using the PM counter threads_launched.

**Warp Instructions Executed**

The counter inst_executed counts the number of warp instructions executed/retired.

**Warp Instructions Issued**

The counter inst_issued counts the number of instructions issued. inst_issued >= inst_executed. Some instructions will be issued multiple times per instruction executed in order to handle dispatch to narrow execution units or in order to handle address divergence in shared memory and L1 operations.

**Thread Instructions Executed**

The counter thread_inst_executed counts the number of thread instructions executed. The metrics avg_threads_executed_per_instruction can be derived using thread_inst_executed / inst_executed. The maximum value for this counter is WARP_SIZE.

**Not Predicated Off Threads Instructions Executed**

Compute capability 2.0 and above devices use instruction predication to disable write-back for threads in a warp as a performance optimization for short sequences of divergent instructions.

The counter not_predicated_off_thread_inst_executed counts the number of instructions executed by all threads. This counter is only available on compute capability 3.0 and above devices.

not_predicated_off_thread_inst_executed <= thread_inst_executed <= WARP_SIZE * inst_executed

This relationship will be off slightly on some chips due to small bugs in thread_inst_executed and not_predicated_off_thread_inst_executed counters.

**Profilers**

Nsight Visual Studio Edition 2.x support collecting the aforementioned counters.

Nsight VSE 3.0 supports a new Instruction Count experiment that can collect per SASS instruction statistics and show the data in table form or next to high level source, PTX, or SASS code. The information is rolled up from SASS to high level source. The quality of the roll up depends on the ability of the compiler to output high quality symbol information. It is recommended that you always look at both source and SASS at the same time. This experiment can collect the following per instruction statistics:

a. inst_executed b. thread_inst_executed (or active mask) c. not_predicated_off_thread_inst_executed (active predicate mask) d. histogram of active_mask e. histogram of predicate_mask

Visual Profiler 5.0 can accurately collect the aforementioned SM counters. nvprof can collect and show the per SM details. Visual Profiler 5.x does not support collection of per instruction statistics available in Nsight VSE 3.0. Older versions of the Visual Profiler and CUDA command line profiler can collect many of the aforementioned counters but the results may not be as accurate as the 5.0 and above version of the tools.


参考手册（找了很久）：https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
看线程执行指令数：
sudo /usr/local/cuda-11.1/bin/ncu --metrics thread_inst_executed ./vectoradd

看启动了多少CTA：
sudo /usr/local/cuda-11.1/bin/ncu --metrics sm__ctas_launched ./gemm

其他非常多的metrix可以参考手册或者直接看
sudo ncu --query-metrics --csv > query-metrics.csv输出的csv文件![[query-metrics.csv]]