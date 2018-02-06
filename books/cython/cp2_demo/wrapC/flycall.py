# import pyx file on the fly
import pyximport
pyximport.install()

import fib
print(fib.fib(3))
print(fib.fib(10))
