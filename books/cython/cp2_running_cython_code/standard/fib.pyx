# cython fibo demo

def fib(int n):
    cdef int i
    cdef unsigned long a=0, b=1
    for i in range(n):
        a, b = a + b, a
    return a

