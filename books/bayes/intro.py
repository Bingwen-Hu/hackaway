# ==================== basic 
# prior probability: a belief of a probability about some event.
# evidence: another event relate to the event
# posterior probability: after seeing the evidence, we update our belief about the event
# using formula: P(A|X) = P(X|A)P(A) / P(X)

# in IPython environment
from IPython.core.pylabtools import figsize
import numpy as np
import matplotlib.pyplot as plt
figsize(12.5, 4)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 120
colors = ['#348ABD', '#A60628']
prior = [1/21, 20/21]
posterior = [0.087, 1-0.087]
plt.bar([0, .7], prior, alpha=0.70, width=0.25,
        color=colors[0], label='prior distribution', 
        lw='3', edgecolor="#348ABD")
plt.bar([0 + 0.25, .7 + 0.25], posterior, alpha=0.70, width=0.25, 
        color=colors[1], label='posterior distribution', 
        lw='3', edgecolor="#A60628")
plt.xticks([0.02, 0.95], ['librarian', 'farmer'])
plt.title("Prior and Posterior probabilities of Steve's occupation")
plt.ylabel("Probability")
plt.legend(loc='upper left')
plt.show()



# Probability Distributions
# let Z be the random variable, then associated with Z is a probability distribution
# function that assign probabilities to the different outcomes Z can take.

# Discrete Case
# Poisson distribution
# P(Z=k) = lambda^k * e^(-lambda) / k!
# the larger lambda gives the big number larger probability
# the expect value of poisson distribution is lambda

figsize(12.5, 4)

from scipy import stats
a = np.arange(16)
poi = stats.poisson
lambda_ = [1.5, 4.25]
colors = ['#348ABD', '#A60628']

plt.bar(a, poi.pmf(a, lambda_[0]), color=colors[0],
        label="$\lambda = %.1f$" % lambda_[0], alpha=0.60,
        edgecolor=colors[0], lw='3')
plt.bar(a, poi.pmf(a, lambda_[1]), color=colors[1],
        label="$\lambda = %.1f$" % lambda_[1], alpha=0.60,
        edgecolor=colors[1], lw='3')

plt.xticks(a + 0.4, a)
plt.legend()
plt.ylabel("Probability of $k$")
plt.xlabel("$k$")
plt.title("Probability mass function of a Poisson random variable "
          "differing $\lambda$ values")
plt.show()


# Continuous case
# A continuous random variable has a probability density function.
# density function for exponential random variable
# f(lambda) = lambda * e^(-lambda * z),  z >= 0
# the expect value of an exponential function is 1 / lambda
a = np.linspace(0, 4, 100)
expo = stats.expon
lambda_ = [0.5, 1]

for l, c in zip(lambda_, colors):
    plt.plot(a, expo.pdf(a, scale=1./l), lw=3, 
             color=c, label="$\lambda = %.1f$" % l)
    plt.fill_between(a, expo.pdf(a, scale=1./l), color=c, alpha=0.33)

plt.legend()
plt.ylabel("Probability density function at $z$")
plt.xlabel("$z$")
plt.ylim(0, 1, 2)
plt.title("Probability density function of an exponential random "
          "variable, differing $\lambda$ values")

plt.show()
