# because division in C and Python is different, 
# cython use Python division default.


cimport cython

@cython.cdivision(True)
def divides(int a, int b):
    return a / b

def remainder(int a, int b):
    with cython.cdivision(True):
        return a % b


