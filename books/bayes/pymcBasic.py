# ==================== parent and child parameter
# parent variables: variables that influence another variable
# child variables: variables that are affected by other variables -- that is, are the subject of 
# parent variables.

# children and parent can be access as attributes in PyMC

import pymc as pm

lambda_ = pm.Exponential("poisson_param", 1)
data_generator = pm.Poisson("data_generator", lambda_)

data_plus_one = data_generator + 1

print("the parents of data_generator is {0}".format(data_generator.parents))
print("the children of data_generator is {0}".format(data_generator.children))


# ==================== PyMC variables
# Stochastic variables: variables that are not deterministic even if its parents are known.
# In this category are instances of Poisson, DiscreteUniform and Exponential
# Deterministic variables: variables that are not random when its parents are known.

# we can initialize an array of stochastic variable by size, for example
beta_1 = pm.Uniform("beta_1", 0, 1)
beta_2 = pm.Uniform("beta_2", 0, 1)

# simply
betas = pm.Uniform("betas", 0, 1, size=2)


# the random method generate a new value for the random variable
lambda_1 = pm.Exponential("lambda_1", 1)
lambda_2 = pm.Exponential("lambda_2", 1)
tau = pm.DiscreteUniform("tau", lower=0, upper=10)

print("the origial values: ")
print("lambda_1, lambda_2, tau are %.3f, %.3f, %.3f" % (lambda_1.value, lambda_2.value, tau.value))

lambda_1.random(), lambda_2.random(), tau.random()

print("the new values: ")
print("lambda_1, lambda_2, tau are %.3f, %.3f, %.3f" % (lambda_1.value, lambda_2.value, tau.value))

# what if we want the value of random variable fixed? using observed=True when initialize the class

# ==================== Deterministic variables
# using decorator, consider is as a variable not a function

# recall an example from intro.py
import numpy as np
n_data_points = 5
@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_data_points)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

# NOTE: the stochastic variable passed to the deterministic variable will be cast to a 
# normal number or array



# Give out an example
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
figsize(12.5, 4)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 100

samples = [lambda_1.random() for i in range(20000)]
plt.hist(samples, bins=70, normed=True)
plt.title("Prior distribution for $\lambda_1$")
plt.xlabel("Value")
plt.ylabel("Density")
plt.xlim(0, 8)
plt.show()

# finally ... put all the things in a model
data = np.array([10, 20, 30, 35, 50])
model = pm.Model([obs, lambda_1, lambda_2, tau])

