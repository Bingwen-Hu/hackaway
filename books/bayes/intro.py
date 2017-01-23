# Thanks for Cameron for his book: Probabilistic Programming Bayesian Method for Hacker
# This is not an easy book for me.

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


figsize(12.5, 3.5)
count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color=colors[0])
plt.xlabel("Time (days)")
plt.ylabel("Text message received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data)
plt.show()


# Probabilistic programming using PyMC
import pymc as pm

alpha = 1.0 / count_data.mean()

lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)

tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)

@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_count_data)
    out[:tau] = lambda_1
    out[tau:] = lambda_2

    return out

observation = pm.Poisson("obs", lambda_, value=count_data, 
                         observed=True)

model = pm.Model([observation, lambda_1, lambda_2, tau])
mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000)

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]
tau_samples = mcmc.trace('tau')[:]

figsize(14.5, 10)

ax = plt.subplot(311)
ax.set_autoscaley_on(False)
plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color=colors[1], normed=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distribution of the parameters """
          r"""$\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")
plt.ylabel("Density")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)

plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7a68a6", normed=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")
plt.ylabel("Density")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"posterior of $\tau$", color="#467821",
         weights=w, rwidth=2)
plt.xticks(np.arange(n_count_data))
plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data)-20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("Probability")

plt.show()


# what good are samples from the Posterior?

figsize(12.5, 5)
N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    ix = day < tau_samples
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum() +
                                   lambda_2_samples[~ix].sum()) / N
plt.plot(range(n_count_data), expected_texts_per_day, lw=4, 
         color='#E24A33', label="Expected number of text message received")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Number of text messages")
plt.title("Number of text messages received versus expected number received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", 
        alpha=0.65, label="Observed text messages per day")
plt.legend(loc="upper left")
print(expected_texts_per_day)
