import math

input_size = (3, 100, 100)

def xavier_init(size):
    """
    input dimension is `dim`
    F = 1 / sqrt(dim / 2) 
    """
    dim = size[0]
    xavier_stddev = 1. / math.sqrt(dim / 2.)
    return xavier_stddev