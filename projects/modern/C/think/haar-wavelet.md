# 哈尔小波

因为看了好几次没看懂，所以做个笔记。

### $L^2(0,1)$空间
$L^2(0,1)$表定义在闭区间(0,1)的平方可积的函数的线性空间(linear space of square (Lebesgue) integrable real-valued functions on the interval (0, 1))。定义两个运算：

内积(inner product):
$$
    (f, g) = \int_0^1 f(x)g(x)dx
$$

求模(norm):
$$
    ||f||^2 = (f, f) = \int_0^1 |f(x)|^2dx, \quad f,g \in L^2(0,1)
$$

求模运算满足三角不等式：
$$
    ||f - g|| \le ||f - b|| + ||b - g|| \quad \forall f, g, h \in L^2(0,1)
$$
因此可见，$||f-g||$是对$f$和$g$之间距离的一种度量。

内积是对矩阵空间$R^n$的点乘运算的一种泛化。因此，如果$(f,g)=0$，则说$f,g$是正交的。正交集合$F \subset L^2(0,1)$表示，其中每一对函数都是正交的。如果对于$F$中的每一个函数，都满足$||f|| = 1, \forall f \in F$，则说$F$是标准正交集。

与矩阵空间的向量分解(vector decomposition)相似，如果对于任意一个函数$f \in L^2(0,1)$，可以表示成:
$$
    f = \sum \limits_{i=1}^\infin (f, f_i)f_i = (f, f_1)f_1 + (f, f_2)f_2  + ...,
$$
我们说标准正交集合$F=\{f_i\}^\infin_{i=1} \subset L^2(0,1)$是一个线性空间$L^2(0,1)$的完全标准正交基(complete orthonormal basis)。
在这里，这个无穷和(infinite sum)称为函数$f$基于$F$的广义傅里叶级数。

哈尔的贡献在于，给出了一种构造$F$的方法。

### 哈尔构造法
选取一个大于0的整数$j$，将闭区间(0,1)平均分成$2_j$份，这些断点构成了一个向量，如下
```py
j = 2
V_j = [i / 2**j for i in range(2**j+1)]
print(V_j)
# [0.0, 0.25, 0.5, 0.75, 1.0]
```
令$V_j$为常数分段函数(piecewise-constant function)的集合，其中的函数在断点$i/2^j$处不连续。很显然，$V_j \subset L^2(0,1)$。

现在，为了在$V_j$中构造一个标准正交基$F_j$，定义$φ: R \rightarrow R$ 如下
$$
    φ(x) = \left\{
        \begin{aligned}
            1 & \quad if \ 0 \le x \le 1  \\
            0 & \quad otherwise 
        \end{aligned}
    \right.
$$

并且，对于任意的 $i = 0,1,...,2^j-1$, 令
$$
    φ_i^j(x) = 2^{j/2}φ(2^jx - i), \quad 0 \le x \le 1
$$

然后令$F_j = \{φ_i^j\}_{i=0}^{2^j-1}$。注意，在$φ_i^j$中的$j$只是上标，不是指数。至此，$F_j$构造完毕。

```py
# φ definition
def Phi(x):
    if 0 <= x <= 1:
        return 1
    else:
        return 0
# continues with all the code block before
# we have j and V_j already
# we discard j in the name for fixed j.
def Phi_ifn(i):
    def Phi_i(x):
        val = 2**(j/2) * Phi(2**j * x - i)
        return val
    return Phi_i
# obviously, this φ_i^j a function
```
我们的标准正交基$F_j$中有$2^j$个函数。

### 研究$F_j$

接下来研究一下$F_j$。先研究$F_2 = \{φ_0^2, φ_1^2, φ_2^2, φ_3^2\}$
```py
F_2 = [Phi_ifn(i) for i in range(2**j)]

def visual_fn(F_j, figsize):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(*figsize)

    xs = np.linspace(0, 1, 50)
    for i, fn in enumerate(F_j):
        ax = plt.subplot(gs[i])
        y = [fn(x) for x in xs]
        ax.set_aspect('auto')
        plt.plot(xs, y)
    plt.show()

visual_fn(F_2, [1, 4])
```
得到的结果
![F_2](F_2.png)

### 参考

+ Programming Projects in C for Students of Engineering, Science, and Mathematics
