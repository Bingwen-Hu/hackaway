## Moving Least Squares Deformation
移动最小二乘法变形算法

Notations:
+ 控制点: p
+ 映射点: q
+ 变形函数: f
+ 满足: f(p) = q

alias:
+ f = $l_v$
+ x = v



### 公式
给定图像上一个点$v$，其损失函数为：
$$ \sum \limits_i w_i |l_v(p_i) - q_i|^2 \tag 1$$
其中，$p_i$和$q_i$均为行向量。权重$w_i$的一般形式为：
$$
    w_i = \frac{1}{|p_i - v|^{2 \alpha}} \tag 2
$$

注意，权重$w_i$跟$v$无关，所以作者称这是移动最小二乘法(Moving Least Squares)。通过以上公式，对于不同的点$v$，有不同的人仿射变换$l_v(x)$

仿射变换
$$
    f(v) = l_v(v)
$$
注意到，当$v$趋近于$p_i$时，$w_i$趋于无穷大。这时候，最小化公式1将使得$|l_v(p_i) - q_i|^2$的值为0，也即$f(p_i)=q_i$。也就是说，如果$q_i=p_i$，则有$x, l_v(x)=x$

当$\alpha \le 1$时，$f$除了在控制点$p_i$处不平滑，在其他地方都是平滑的。

### 解构 $l_v(x)$
$l_v$由两部分组成：线性变换矩阵$M$以及平移量$T$
$$
    l_v(x) = xM + T     \tag 3
$$
代入公式1有
$$
   \sum \limits_i w_i |p_iM + T - q_i|^2 \tag 4 
$$
对$T$求偏导，令偏导为0，得
$$
    \sum \limits_i 2w_i \times (p_i M + T - q_i) = 0 \\
    \sum \limits_i w_i p_iM + \sum \limits_i w_i T - \sum \limits_i w_i q_i = 0 \\
    \sum \limits_i w_i T = \sum \limits_i w_i q_i -  \sum \limits_i w_i p_iM \\
    T =  \sum \limits_i w_i q_i / \sum \limits_i w_i - (\sum \limits_i w_i p_i / \sum \limits_i w_i) \times M \\
    let: q_* =  \sum \limits_i w_i q_i / \sum \limits_i w_i, \quad p_* = \sum \limits_i w_i p_i / \sum \limits_i w_i \\
    T = q_* - p_* M
$$

我们将上式$T$的公式代入公式3,得
$$
    l_v(x) = xM + q_* - p_* M \\
    l_v(x) = (x - p_*) M + q_* \tag 5
$$

将公式5代入公式1,得
$$
\sum \limits_i w_i |(p_i - p_*) M + q_*  - q_i|^2 \\ 
\sum \limits_i w_i |(p_i - p_*) M - (q_i  - q_*)|^2 \\
\sum \limits_i w_i |\hat{p_i} M - \hat{q_i}|^2 \tag{6} 
$$
其中，$\hat{p_i} = p_i - p_*$, $\hat{q_i} = q_i - q_*$

### 仿射变换(Affine Deformations)
这是最简单的，所以先来。但我们的目标是Rigid Deformation。

用公式法求公式6的矩阵$M$：
$$
    M = (\sum \limits_i \hat{p_i}^Tw_i\hat{p_i})^{-1} \sum \limits_j w_j \hat{p_j}^T\hat{q_j}
$$
将上式代入公式5, 可得
$$
    f_a(v) = (v - p_*)(\sum \limits_i \hat{p_i}^Tw_i\hat{p_i})^{-1} \sum \limits_j w_j \hat{p_j}^T\hat{q_j} + q_* \tag{7}
$$
对于图像上的每个点应用这个公式，可以得到新的图片。

> 为什么M是2x2的矩阵？

用户往往通过调整$q_i$来改变形变效果，而$p_i$部分保持固定，那$p_i$部分可以预先计算好，以加速形变。将公式7改写成
$$
    f_a(v) = \sum \limits_j A_j \hat{q_j} + q_*
$$
其中，$A_j$是一个简单的标量，由以下公式计算得到：
$$
    A_j = (v - p_*)(\sum \limits_i \hat{p_i}^T w_i \hat{p_i})^{-1} w_j \hat{p_j}^T
$$

### 相似变换(Similarity Deformations)
对$M$进行约束，令
$$
    M^T M = \lambda^2I
$$
写成分块矩阵的形式
$$
    M = (M_1 \quad M_2)
$$
其中，$M_1$，$M_2$是长为2的列向量。那么有
$$
    M_1^TM_1 = M_2^TM_2 = \lambda^2 \\
    M_1^TM_2 = 0
$$
这要求$M_1$与$M_2$正交。即$M_2 = M_1^{\perp}$。如果一来，可以用$M_1$来改写公式6
$$
    \sum \limits_i w_i |\begin{pmatrix} \hat{p_i} \\ -\hat{p_i}^{\perp} \end{pmatrix} M_1 - \hat{q_i}^T|^2 \tag{8}
$$
上式有唯一最小值，得
$$
    M = \frac{1}{\mu_s}\sum \limits_i w_i \begin{pmatrix} \hat{p_i} \\ -\hat{p_i}^{\perp} \end{pmatrix} 
        (\hat{q_i}^T \quad -\hat{q_i}^T)  
$$
其中，
$$
    \mu_s = \sum \limits_i w_i \hat{p_i} \hat{p_i}^T
$$

这里也有预计算部分，略去。

### 严格变换(Rigid Deformation)
严格变换认为，形变不应该涉及放缩。

这一部分特别难懂！
