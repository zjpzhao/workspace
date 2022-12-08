
## Observation框

![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/observation%E6%A1%86.png)

```latex
\begin{center}
\fbox{\parbox{.95\linewidth}{
      \textbf{Observation\#1}: 
      \textit{Instruction type and instruction function have profound impact on instruction SDC proneness.}
}}
\end{center}
```


```latex
\texttt
```

## 表格
![](https://zjpimage.oss-cn-qingdao.aliyuncs.com/latex%20table.png)

```latex
\begin{table}[]
\centering
\caption{The heuristic features of instruction inherent property and fault propagation information.}\label{tbl1}
\resizebox{85mm}{!}{%
\begin{tabular}{|c||c|c|c|}
\hline
\multicolumn{1}{|l|}{}                          & \textbf{No.} & \textbf{Type} & \textbf{Feature   Description}          \\ \hline
\multirow{13}{*}{\rotatebox{90}{Instruction Inherent Property}}
                                                & 1            & Bool          & Is multiplication or division operation \\ \cline{2-4} 
                                                & 2            & Bool          & Is add or sub operation                 \\ \cline{2-4} 
                                                & 3            & Bool          & Is floating-point or integer operation  \\ \cline{2-4} 
                                                & 4            & Bool          & Is logic operation                      \\ \cline{2-4} 
                                                & 5            & Bool          & Is shift operation                      \\ \cline{2-4} 
                                                & 6            & Bool          & Is load operation                       \\ \cline{2-4} 
                                                & 7            & Bool          & Is dead-store operation                 \\ \cline{2-4} 
                                                & 8            & Bool          & Is output relevant-store operation      \\ \cline{2-4} 
                                                & 9            & Bool          & Is data movement operation              \\ \cline{2-4} 
                                                & 10           & Bool          & Is address calculation                  \\ \cline{2-4} 
                                                & 11           & Bool          & Is arithmetic calculation               \\ \cline{2-4} 
                                                & 12           & Bool          & Is loop control operation               \\ \cline{2-4} 
                                                & 13           & Bool          & Is branch control operation             \\ \hline
\multirow{8}{*}{\rotatebox{90}{Fault Propagation}}              
                                                & 14           & Int           & Address calculation in IDS              \\ \cline{2-4} 
                                                & 15           & Int           & Shift operation in IDS                  \\ \cline{2-4} 
                                                & 16           & Int           & Shift bits                              \\ \cline{2-4} 
                                                & 17           & Int           & Logic operation in IDS                  \\ \cline{2-4} 
                                                & 18           & Int           & Data movement in IDS                    \\ \cline{2-4} 
                                                & 19           & Bool          & Is branch control instruction in IDS    \\ \cline{2-4} 
                                                & 20           & Bool          & Is terminal output-relevant store       \\ \cline{2-4} 
                                                & 21           & Int           & Length of IDS                           \\ \hline
\end{tabular}
}
\end{table}
```