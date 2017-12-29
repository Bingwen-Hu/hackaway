import cffi
ffi = cffi.FFI()
ffi.cdef('int printf(const char* format, ...);')
libc = ffi.dlopen(None)
print("mory {}".format(libc))

arg = ffi.new('char[]', b'spam')
libc.printf(arg)

