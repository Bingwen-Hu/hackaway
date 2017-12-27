import ctypes
print(ctypes.cdll)
libc = ctypes.cdll.LoadLibrary('libc.so.6')
print(libc)
print(libc.printf)


# another way
from ctypes import util
libc = ctypes.cdll.LoadLibrary(util.find_library('libc.so'))
print(libc)                     # fail
