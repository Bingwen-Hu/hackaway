import cffi

ffi = cffi.FFI()
x = ffi.new('int[10]')
y = ffi.new('int[]', 10)
x[0:10] = range(10)
y[0:10] = range(10, 0, -1)
print(list(x))
print(list(y))
