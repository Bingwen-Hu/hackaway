import ctypes
from cStructure import Spam


TenNumbers = 10 * ctypes.c_double
numbers = TenNumbers()
print(numbers[0])


Spams = 5 * Spam
spams = Spams()
spams[0].eggs = 123.456
print(spams)
print(spams[0])
print(spams[0].eggs)
print(spams[0].spam)

# resize
# two condition: new size is larger; must be specific to bytes
ctypes.resize(numbers, 11 * ctypes.sizeof(ctypes.c_double))
print(numbers)
