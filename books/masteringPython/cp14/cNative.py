import ctypes
libc = ctypes.cdll.LoadLibrary('libc.so.6')


spam = ctypes.create_string_buffer(b'spam')
print(ctypes.sizeof(spam))
print(spam.raw)
print(spam.value)
libc.printf(spam)


# must convert python type to C type
# format
format_string = ctypes.create_string_buffer(b'Number: %d\n')
libc.printf(format_string, 123)

format_string = ctypes.create_string_buffer(b'Number: %.3f\n')
x = ctypes.c_double(123.45)
libc.printf(format_string, x)

