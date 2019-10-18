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