Structure: from *source code* to *intermediate representation(IR)*, then to *machine code*.
IR 级别可以有IR优化，从IR到汇编代码级别是有后端优化的。前端处理程序的源代码（如C/C++代码）并生成程序的中间表示（IR），而中间和后端分别对代码进行与平台无关的优化和特定平台的优化，例如[[NVCC]]中的-O0, -O1, -O2和-O3就是编译器的优化级别。
LLVM的IR文件以 *.ll* 结尾