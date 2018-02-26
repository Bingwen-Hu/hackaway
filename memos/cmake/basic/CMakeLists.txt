# Cmake lowest version
cmake_minimum_required(VERSION 2.8)

# project infomation
project (multifile)

# target
add_executable(calculate calculate.c MathFunction.c)