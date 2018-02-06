def py_fact(n):
    if n <= 1:
        return 1
    return n * py_fact(n - 1)


# %timeit py_fact(20) in iPython

def typed_fact(long n):
    if n <= 1:
        return 1
    return n * typed_fact(n - 1)


# more type, but can't not call directly from outside, so we need a wrapper
cdef long c_fact(long n):
    if n <= 1:
        return 1
    return n * c_fact(n - 1)

def wrap_c_fact(n):
    return c_fact(n)


# for convenient, cpdef is provide
cpdef long cp_fact(long n):
    if n <= 1:
        return 1
    return n * cp_fact(n - 1)
