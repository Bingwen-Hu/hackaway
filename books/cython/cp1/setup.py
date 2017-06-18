from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("fib.pyx"))

# the command is:
# python3 setup.py build_ext --inplace
