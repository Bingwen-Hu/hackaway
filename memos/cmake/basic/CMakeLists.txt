# Cmake lowest version
cmake_minimum_required(VERSION 2.8)

# project infomation
project (multifile)


# store a source file directory.
aux_source_directory(. DIR_SOURCES)

# target
add_executable(calculate ${DIR_SOURCES})