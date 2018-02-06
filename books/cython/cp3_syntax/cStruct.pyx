cdef struct surpasser:
    int age
    int level

def struct_test():
    cdef surpasser mory = surpasser(age=12, level=0)
    cdef surpasser *p_mory = &mory
    cdef int age = mory.age
    cdef int level = p_mory.level
    print("No matter struct or pointer to a struct, access a "
          "proper is using a dot.")
    print("age is {}, level is {}".format(age, level))


# union
cdef union uu:
    int a
    short b, c

ctypedef struct mycpx:          # same as cdef
    float real
    float imag


def mycpx_test():
    cdef mycpx cpx = mycpx(real=1.0, imag=0.3)
    print(f"real part is: {cpx.real}, image part is {cpx.imag}")


