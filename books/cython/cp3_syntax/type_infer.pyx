cimport cython

@cython.infer_types(True)
def inference():
    i = 1
    d = 2.0
    c = 3+4j
    r = i * d + c
    return r
    

