# 4 Numerical Computation
## 4.1 overflow and underflow
+ underflow: Underflow occurs when numbers near zero are rounded to zero.
+ overflow: Overflow occurs when numbers with large magnitude are approximated as ∞ or −∞. Further arithmetic will usually change these infinite values into not-a-number values.
One example of a function that must be stabilized against underflow and overflow is the softmax function. The softmax function is often used to predict the probabilities associated with a multinoulli distribution. The softmax function is defined to be
$$ softmax(x)_i = \frac{exp(x_i)}{\sum^n_{j=1}exp(x_j)} $$

## 4.2 Poor conditioning 
Conditioning refers to how rapidly a function changes with respect to small changes in its inputs. Functions that change rapidly when their inputs are perturbed slightly can be problematic for scientific computation because rounding errors in the inputs can result in large changes in the output

Poorly conditioned matrices amplify pre-existing errors when we multiply by the true matrix inverse. In practice, the error will be compounded further by numerical errors in the inversion process itself.

## 4.3 Gradient-Based Optimization
Most deep learning algorithms involve optimization of some sort. Optimization refers to the task of either minimizing or maximizing some function $f(x)$ by altering x. We usually phrase most optimization problems in terms of minimizing $f(x)$. Maximization may be accomplished via a minimization algorithm by minimizing $-f(x)$.

Suppose we have a function $y = f(x)$, where both x and y are real numbers. dy The derivative of this function is denoted as $f^{'}(x)$ or as $\frac{dy}{dx}$ . The derivative $f^{'}(x)$ gives the slope of f(x) at the point x. In other words, it specifies how to scale a small change in the input in order to obtain the corresponding change in the output: $f(x + \epsilon) ≈ f(x) + \epsilon f^{'}(x)$.

The derivative is therefore useful for minimizing a function because it tells us how to change x in order to make a small improvement in y. For example, we know that $f(x - \epsilon\ sign(f^{'}(x)))$ is less than f (x) for small enough . We can thus reduce f (x) by moving x in small steps with opposite sign of the derivative. This technique is called gradient descent

We often minimize functions that have multiple inputs: $f : R^n → R$. For the concept of “minimization” to make sense, there must still be only one (scalar) output.

For functions with multiple inputs, we must make use of the concept of partial derivatives. The partial derivative $\frac{∂}{∂x_i} f(x)$ measures how f changes as only the i variable $x_i$ increases at point x. The gradient generalizes the notion of derivative to the case where the derivative is with respect to a vector: **the gradient of f is the vector containing all of the partial derivatives, denoted $∇_x f (x)$**. Element i of the gradient is the partial derivative of f with respect to $x_i$. In multiple dimensions, critical points are points where every element of the gradient is equal to zero.

The directional derivative in direction u (a unit vector) is the slope of the function f in direction u. In other words, the directional derivative is the derivative of the function $f (x + αu)$ with respect to α , evaluated at α = 0. Using the chain rule, we can see that $\large \frac{∂}{∂α} f (x + α u) = u^T ∇_x f(x)$ . To minimize f , we would like to find the direction in which f decreases the fastest. We can do this using the directional derivative:
$$ \large \mathop{min} \limits_{u, u^Tu=1} u^T ∇_x f(x) \ 
        = \mathop{min} \limits_{u, u^Tu=1} ||u||_2  ||∇_x f(x)||_2 cos\theta$$
    
where θ is the angle between u and the gradient. Substituting in $||u||_2 = 1$ and ignoring factors that do not depend on u, this simplifies to $min_u cos θ$. This is minimized when u points in the opposite direction as the gradient. In other words, the gradient points directly uphill, and the negative gradient points directly downhill. We can decrease f by moving in the direction of the negative gradient. This is known as the method of steepest descent or gradient descent.

Steepest descent proposes a new point
$$ \large x^{'} = x - \epsilon ∇_x f(x)$$

where $\epsilon$ is the learning rate, a positive scalar determining the size of the step. We can choose in several different ways. A popular approach is to set to a small constant. Sometimes, we can solve for the step size that makes the directional derivative vanish. Another approach is to evaluate $f(x − \epsilon ∇_x f(x))$ for several values of and choose the one that results in the smallest objective function value. This last strategy is called a line search. 

Steepest descent converges when every element of the gradient is zero (or, in practice, very close to zero). In some cases, we may be able to avoid running this iterative algorithm, and just jump directly to the critical point by solving the equation $∇_x f (x) = 0$ for x.