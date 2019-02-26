### 0.1 set terminology and notation
+ contain(elment): a object belong to a set, say the set contains the object
+ contain(set): all the element in a set belong to another set, say the second set contain the first one.
+ cardinality: number of elements a set contains.
$$ cardinality(A) = |A| $$

usually, *R* denote set of all real numbers, and *C* consists of all complex numbers.

### 0.2 cartesian product
(named Descartes)

the cartesian product of set A and B is the set of all pairs(a,b) where a->A and b->B
$$ cp(A, B) = A X B $$
$$ |A X B| = |A| X |B| $$

### 0.3 the function
For sets D and F , we use the notation $F^D$ to denote all functions from D to F

### 0.4 function composite
Given two functions $f : A \to B$ and $g : B \to C$, the function $g \circ f$ , called the composition of g and
$$ (g \circ f)(x) = g(f(x)) $$
for every $x \in A$ \
If the image of f is not contained in the domain of g then $g \circ f$ is not a legal expression.

### 0.5 functioanl inverse
the function that reverses the effect of the encryption function. This function is said to be the functional inverse of the encryption function.\

definition: 
+ $f \circ g$ is defined and is the identity function on the domain of g and
+ $g \circ f$ is defined and is the identity function on the domain of f

If f and g are invertible functions and $f \circ g$ exists then $f \circ g$ is invertible and $(f \circ g)^{−1} = g^{−1} \circ f^{−1}$