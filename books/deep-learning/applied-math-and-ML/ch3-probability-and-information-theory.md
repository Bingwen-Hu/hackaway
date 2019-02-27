## 3.1 probability
1. three possible sources of uncertainty
+ inherent stochasticity in the system being modeled.
+ incomplete observability
+ incomplete modeling   
2. random variable 
+ discrete
+ continuous

## 3.2 probability distributions
A probability distribution is a description of how likely a random variable or set of random variables is to take on each of its possible states. The way we describe probability distributions depends on whether the variables are discrete or continuous.

### 3.2.1 discrete variable and probability mass functions
To be a probability mass function on a random variable x, a function P must satisfy the following properties:
+ the domain of P must be the set of all possible states of x
+ $\large \forall x \in \bold{x}, 0 \le P \le 1$
+ $\large \sum_{x \in \bold{x}}P(x) = 1$

### 3.2.2 continuous variable and probability density functions
To be a probability density function, a function p must satisfy the following properties:
+ the domain of P must be the set of all possible states of x
+ $\forall x \in \bold{x}, p(x) \ge 0$. Note that we do not require $p(x) \le 1 $
+ $\int p(x)dx = 1$
the probability that x lies in the interval [a, b] is given by $\int _{a,b} p(x)dx$.

denote uniform distributions on [a, b]:
x ~ U(a,b)

## 3.3 marginal probability
For example, suppose we have discrete random variables x and y, and we know P(x,y) . We can find P(x) with the sum rule:
$$ \large \forall x \in \bold{x}, P(\bold{x} = x) = \sum \limits_y P(\bold{x} = x, \bold{y}=y)$$
For continuous variables, we need to use integration instead of summation:
$$ \large p(x) = \int p(x, y)dy$$

## 3.4 conditional probability
In many cases, we are interested in the probability of some event, given that some other event has happened. This is called a conditional probability.
$$ \large P(\bold y = y| \bold{x} = x) = \frac{P(\bold{y}=y, \bold{x}=x)}{P(\bold{x} = x)}$$
chain rule:
$$ P(a, b, c) = P(a|b,c) P(b|c) P(c)$$

## 3.5 independence and conditional independence
Two random variables x and y are independent if their probability distribution can be expressed as a product of two factors, one involving only x and one involving only y
$$ \large \forall x \in \bold{x}, y \in \bold{y}, p(\bold{x}=x, \bold{y}=y)=p(\bold{x}=x)p(\bold{y}=y)$$

Two random variables x and y are conditionally independent given a random variable z if the conditional probability distribution over x and y factorizes in this way for every value of z:
$$ \large \forall x \in \bold{x}, y \in \bold{y}, z \in \bold{z}, p(\bold{x}=x, \bold{y}=y| \bold{z}=z)=p(\bold{x}=x|\bold{z}=z)p(\bold{y}=y|\bold{z}=z)$$

compact notation:
+ $\large x⊥y$ : x and y are independent
+ $\large x⊥y|z$ : x and y are conditionally independent given by z.

## 3.6 expectation, variance and covariance
The expectation or expected value of some function f(x) with respect to a probability distribution P(x) is the average or mean value that f takes on when x is drawn from P.
+ For discrete variables this can be computed with a summation:
$$ \large E_{x~P}[f(x)] = \sum \limits_x P(x)f(x)$$
+ while for continuous variables, it is computed with an integral:
$$ \large E_{x~P}[f(x)] = \int p(x) f(x) dx $$

The variance gives a measure of how much the values of a function of a random variable x vary as we sample different values of x from its probability distribution.
$$ \large Var(f(x)) = E[(f(x) - E[f(x)]^2)]$$
When the variance is low, the values of f(x) cluster near their expected value. The square root of the variance is known as the *standard deviation*.

The covariance gives some sense of how much two values are linearly related to each other, as well as the scale of these variables:
$$ \large Cov(f(x), g(y)) = E[(f(x) - E[f(x)])(g(y) - E[g(y)])]$$

Other measures such as *correlation* normalize the contribution of each variable in order to measure only how much the variables are related, rather than also being affected by the scale of the separate variables

Independence is a stronger requirement than zero covariance, because independence also excludes nonlinear relationships. It is possible for two variables to be dependent but have zero covariance.

The covariance matrix of a random vector $x ∈ R^n$ is an n × n matrix, such that
$$ \large Cov(\bold{x}_{i,j}) = Cov(x_i, y_i)$$
The diagonal elements of the covariance give the variance:
$$ \large Cov(\bold{x_i}, \bold{x_i}) = Var(\bold{x_i})$$

## 3.7 common distribution
+ Bernoulli Distribution
$$ P(\bold{x} = 1) = φ $$
$$ P(\bold{x} = 0) = 1-φ $$
$$ P(\bold{x} = x) = φ^x (1-φ)^{1-x} $$
$$ E_x[\bold{x}] = φ $$
$$ Var_x(\bold x) = φ (1 - φ) $$

+ Guassioan Distribution
The most commonly used distribution over real numbers is the normal distribution, also known as the Gaussian distribution:
$$ \large N(x; \mu, \sigma^2) = \sqrt\frac{1}{2\pi\sigma^2}exp(-\frac{1}{2\sigma^2}(x-\mu)^2)$$

+ Exponential and Laplace Distribution
In the context of deep learning, we often want to have a probability distribution with a sharp point at x = 0. To accomplish this, we can use the exponential distribution:
$$ \large p(x; \lambda) = \lambda I_{x \ge 0} exp(-\lambda x)$$
The exponential distribution uses the indicator function $1_{x≥0}$ to assign probability zero to all negative values of x .

A closely related probability distribution that allows us to place a sharp peak of probability mass at an arbitrary point μ is the Laplace distribution
$$ \large Laplace(x; \mu, γ) = \frac{1}{2γ} exp (- \frac{|x-\mu|}{γ})$$

+ The Dirac Distribution and Empirical Distribution
In some cases, we wish to specify that all of the mass in a probability distribution clusters around a single point. This can be accomplished by defining a PDF using the Dirac delta function, δ(x):
$$ p(x) = δ (x − μ) $$

A common use of the Dirac delta distribution is as a component of an empirical distribution,
$$ \large \dot p(x) = \frac{1}{m} \sum \limits^m_{i=1} δ(x - x^{(i)}) $$

## 3.8 useful properties of common functions
+ logistic sigmoid:
$$ \large \sigma(x) = \frac{1}{1+exp(-x)}$$
The logistic sigmoid is commonly used to produce the φ parameter of a Bernoulli distribution because its range is (0,1)
softplus:
$$ ζ(x) = log(1 + exp(x))$$

The following properties are all useful enough that you may wish to memorize them
$$\large σ(x) = \frac {exp(x)} {exp(x) + exp(0)} $$
$$\large \frac{d}{dx}\sigma(x) = \sigma(x)(1-\sigma(x))$$
$$ 1 - σ(x) = σ(-x)$$
$$ log\ σ(x) = -ζ(-x)$$
$$ \frac{d}{dx}ζ(x) = σ(x)$$
$$ ∀ x ∈ (0 , 1) , σ^{-1}(x) = log(\frac{x}{1-x})$$
$$ ∀x > 0 , ζ^{-1}(x) = log (exp(x)-1)$$
$$ ζ(x) = \int^x_{-\infty} σ(y)dy$$
$$ ζ(x) - ζ(-x)= x$$