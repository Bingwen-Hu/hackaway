# using & to get reference
# using index to get value
from cython.operator cimport dereference

cdef double golden_ratio
cdef double *p_double


p_double = &golden_ratio
p_double[0] = 1.618
print(golden_ratio)
print(p_double[0])


# alternatively
print(dereference(p_double))
