cmake_minimum_required(VERSION 3.0)

project(seqlist)

aux_source_directory(. DIR_SOURCES)

add_executable(seqlist_test ${DIR_SOURCES})