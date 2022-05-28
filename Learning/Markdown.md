---
id: 50919e2f-a53d-4905-b3b4-73537442a528
---
#doc #learning/LaTeX #learning/markdown 

# 基础用法
划掉


# 数学符号和公式
参考<https://www.jianshu.com/p/25f0139637b7>

## 排列组合
1. `$$\binom{n}{m}$$`

$$\binom{n}{m}$$

2. `$$\tbinom{n}{m}$$`

$$\tbinom{n}{m}$$

3. `$$\dbinom{n}{m}$$`

 $$\dbinom{n}{m}$$
 ## 方程组
```latex
$$
\left \{ 
\begin{array}{c}
a_1x+b_1y+c_1z=d_1 \\ 
a_2x+b_2y+c_2z=d_2 \\ 
a_3x+b_3y+c_3z=d_3
\end{array}
\right.
$$
```
$$
\left \{ 
\begin{array}{c}
a_1x+b_1y+c_1z=d_1 \\ 
a_2x+b_2y+c_2z=d_2 \\ 
a_3x+b_3y+c_3z=d_3
\end{array}
\right.
$$
### 大括号 多行公式 和条件居右
array 中lrc表示各列内容的居左、居中、居右。（lr的意思是两端对齐）

```latex
\begin{equation}
\left\{
             \begin{array}{lr}
             x=\dfrac{3\pi}{2}(1+2t)\cos(\dfrac{3\pi}{2}(1+2t)), &  \\
             y=s, & 0\leq s\leq L,|t|\leq1.\\
             z=\dfrac{3\pi}{2}(1+2t)\sin(\dfrac{3\pi}{2}(1+2t)), &  
             \end{array}
\right.
\end{equation}
```
$$
\begin{equation}
\left\{
             \begin{array}{lr}
             x=\dfrac{3\pi}{2}(1+2t)\cos(\dfrac{3\pi}{2}(1+2t)), &  \\
             y=s, & 0\leq s\leq L,|t|\leq1.\\
             z=\dfrac{3\pi}{2}(1+2t)\sin(\dfrac{3\pi}{2}(1+2t)), &  
             \end{array}
\right.
\end{equation}
$$