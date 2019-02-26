## 2.1 scalars, vectors, matrices and tensors
scalar:
$$\large s \in \R $$
vector:
$$\large x \in \R $$
matrix:
$$\large A_{i,j} \in \R^{mn}$$
transpose:
$$\large (A_{i,j})^T = A_{i,j}$$

## 2.2 multiplying matrices and vectors
$$ x^T y = y^T x$$
$$ (AB)^T = B^T A^T$$

## 2.3 identity and inverse matrix
identity matrix:
$$\large \forall x \in \R^n, I_n x = x $$
inverse matrix:
$$\large A^{-1} A = I_n $$
 
## 2.4 linear dependences and span
$$ Ax = b \tag 1$$
In order for $A^{−1}$ to exist, Eq. 1 must have exactly one solution for every value of b.

If both x and y are solutions then 
$$ z = \alpha x + (1-\alpha) y$$
is also a solution for any real $\alpha$.


Formally, a linear combination of some set of vectors $\{v^{(1)}, ... , v^{(n)}\}$ is given by multiplying each vector $v^{(i)}$ by a corresponding scalar coefficient and adding the results: 
$$\Large \sum \limits_i c_i v^{(i)}$$ 
The span of a set of vectors is the set of all points obtainable by linear combination of the original vectors.

In order for the system Ax = b to have a solution for all values of $b \in R^m$ , we therefore require that the column space of A be all of $R^m$ .

A set of vectors is *linearly independent* if no vector in the set is a linear combination of the other vectors. If we add a vector to a set that is a linear combination of the other vectors in the set, the new vector does not add any points to the set’s span.

In order for the matrix to have an inverse, the matrix must be square, that is, we require that m = n and that all of the columns must be linearly independent. A square matrix with linearly dependent columns is known as *singular*.

## 2.5 Norms
Formally, the $ L^P $ norm is given by:
$$ \large ||x||_P = \sum |x_i|^P)^{\frac{1}{p}}$$
for $p \in \R, p \ge 1$
More rigorously, a norm is any function f that satisfies the following properties:
+ $f(x) = 0 ⇒ x = 0$
+ $f(x + y) \le f(x) + f(y) \quad (the\ triangle\ inequality)$
+ $\forall \alpha \in \R, f(\alpha x) = |\alpha| f(x)$


The $L^1$ norm is commonly used in machine learning when the difference between zero and nonzero elements is very important.

Sometimes we may also wish to measure the size of a matrix. In the context of deep learning, the most common way to do this is with the otherwise obscure Frobenius norm
$$ \large ||A||_F = \sqrt{\sum \limits_{i,j}A^2_{i,j}}$$
which is analogous to the L 2 norm of a vector.

The dot product of two vectors can be rewritten in terms of norms. Specifically,
$$ x^Ty = ||x||_2 ||y||_2 cos \theta $$
where $\theta$ is the angle between x and y.

## 2.6 Special Kinds of Matrices and Vectors
### 2.6.1 Diagonal:
Diagonal matrices consist mostly of zeros and have non-zero entries only along the main diagonal. Formally, a matrix D is diagonal if and only if $D_{i,j}= 0$ for all $i \ne j$ . We have already seen one example of a diagonal matrix: the identity matrix, where all of the diagonal entries are 1. We write $diag(v)$ to denote a square diagonal matrix whose diagonal entries are given by the entries of the vector v.
$$ diag(v)x = v \circledast x$$
$$ diag(v)^{-1} = diag([1/v_1, ..., 1/v_n]^T)$$

### 2.6.2 Symmetric:
symmetric matrix is any matrix that is equal to its own transpose:
$$ A = A^T$$

### 2.6.3 unit vector
A unit vector is a vector with unit norm
$$ ||x||_2 = 1$$

### 2.6.4 orthogonal
A vector x and a vector y are orthogonal to each other if $x^T y = 0$. If both vectors have nonzero norm, this means that they are at a 90 degree angle to each other. In $R^n$ , at most n vectors may be mutually orthogonal with nonzero norm. If the vectors are not only orthogonal but also have unit norm, we call them *orthonormal*. \
An orthogonal matrix is a square matrix whose rows are mutually orthonormal
and whose columns are mutually orthonormal:
$$ A^TA = AA^T = I$$
this implies:
$$ A^T = A^{-1}$$

### 2.7 Eigendecomposition

Many mathematical objects can be understood better by breaking them into constituent parts, or finding some properties of them that are universal, not caused by the way we choose to represent them.

For example, integers can be decomposed into prime factors. The way we represent the number 12 will change depending on whether we write it in base ten or in binary, but it will always be true that 12 = 2 × 2 × 3. From this representation we can conclude useful properties, such as that 12 is not divisible by 5 , or that any integer multiple of 12 will be divisible by 3.

Much as we can discover something about the true nature of an integer by decomposing it into prime factors, we can also decompose matrices in ways that show us information about their functional properties that is not obvious from the representation of the matrix as an array of elements.

One of the most widely used kinds of matrix decomposition is called eigen- decomposition, in which we decompose a matrix into a set of eigenvectors and eigenvalues.

$$ Av = \lambda v$$
The scalar λ is known as the eigenvalue corresponding to this eigenvector.

decompose matrix A
$$ A = V diag(\lambda)V^{-1}$$
here, V is a matrix of eigenvectors per column $[v^{(1)}, ..., v^{(n)}]$, and $\lambda = [λ_1, ..., λ_n]^T$ 

Specifically, every real symmetric matrix can be decomposed into an expression using only real-valued eigenvectors and eigenvalues:
$$A = Q Λ Q^T $$
where Q is an orthogonal matrix composed of eigenvectors of A, and Λ is a diagonal matrix. The eigenvalue $Λ_{i,i}$ is associated with the eigenvector in column i of Q, denoted as $Q_{:,i}$.\
The eigendecomposition of a matrix tells us many useful facts about the
matrix. The matrix is singular if and only if any of the eigenvalues are 0.\
A matrix whose eigenvalues are all positive is called positive definite. A matrix whose eigenvalues are all positive or zero-valued is called positive semidefinite. \
Positive semidefinite matrices are interesting because they guarantee that $∀x , x^TAx ≥ 0$. Positive definite matrices additionally guarantee that $x^TAx = 0 ⇒ x = 0$.

## 2.8 Singular Value Decomposition
The singular value decomposition (SVD) provides another way to factorize a matrix, into singular vectors and singular values. Every real matrix has a singular value decomposition, but the same is not true of the eigenvalue decomposition. For example, if a matrix is not square, the eigendecomposition is not defined, and we must use a singular value decomposition instead.

$$ A = UDV^T $$
Suppose that A is an m × n matrix. Then U is defined to be an m × m matrix, D to be an m × n matrix, and V to be an n × n matrix.\
Each of these matrices is defined to have a special structure. The matrices U and V are both defined to be orthogonal matrices. The matrix D is defined to be a diagonal matrix. Note that D is not necessarily square.
The elements along the diagonal of D are known as the singular values of the matrix A. The columns of U are known as the left-singular vectors. The columns of V are known as as the right-singular vectors.\
We can actually interpret the singular value decomposition of A in terms of the eigendecomposition of functions of A. The left-singular vectors of A are the eigenvectors of $AA^T$. The right-singular vectors of A are the eigenvectors of $A^TA$. The non-zero singular values of A are the square roots of the eigenvalues of $A^TA$. The same is true for $AA^T$.