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
