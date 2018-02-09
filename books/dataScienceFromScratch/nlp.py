"""
Just introduce some useful things:
Gibbs Sampling and a function choose sample according to its weights
"""

""" Gibbs Sampling 
Some distributions are harder to sample from. Gibbs Sampling is a technique for 
generating samples from multidimensional distributions when we only know some of
the conditional distributions.
"""

# Given an example of rolling two dice (x, y), x be the first die and y be the
# sum of two dice

import random

def roll_a_die():
    return random.choice([1, 2, 3, 4, 5, 6])

def direct_sample():
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

# but what if we only konw the conditional distributions? knowing x to infer y
# and vice versa

def random_y_given_x(x):
    return x + roll_a_die()

def random_x_given_y(y):
    if y <= 7:
        return random.randrange(1, y) # 1 to y-1
    else:
        return random.randrange(y-6, 7) # total to 6
    
# this function have the same effect as the direct one
def gibbs_sample(num_iters=100):
    x, y = 1, 2                 # initial value is arbitrary
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y



# a function randomly choose an index based on its weights

def sample_from(weights):
    total = sum(weights)
    rnd  = total * random.random()
    for i, w in enumerate(weights):
        rnd -= w
        if rnd <= 0: return 1
