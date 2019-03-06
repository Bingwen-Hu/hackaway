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