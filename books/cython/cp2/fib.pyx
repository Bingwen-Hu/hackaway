def fib(int n):
    cdef int i
    cdef double a=0.0, b=1.0
    for i in range(n):
        a, b = a+b, a
    return a

# some Note
# When using python3.6-config to config the CFLAGS
# add a flag -fPIC
# or else 
#    gcc fib.o -o fib.so 
# will get error!
