def print_address(a):
    cdef void *v = <void*>a
    cdef long addr = <long>v
    print("Cython address: ", addr)
    print("Python id: ", id(a))


def cast_to_list(a):
    cdef list cast_list = <list>a
    print(type(a))
    print(type(cast_list))
    cast_list.append(1)

# notice the ?
def safe_cast_to_list(a):
    cdef list cast_list = <list?>a
    cast_list.append(1)
