import numpy as np


F_in = 64 # input of network layer   
F_out = 32 # output of network layer 
limit = np.sqrt(6 / float(F_in))
w = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))
