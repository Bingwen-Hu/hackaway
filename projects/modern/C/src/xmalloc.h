// I get this tools from the book: 
// Programming Projects in C for Students of Engineering, Science, and Mathematics

#pragma once
#include <stdlib.h>


void* malloc_or_exit(size_t nbytes, const char *file, int line);
// wrapper for malloc_or_exit
#define xmalloc(nbytes) malloc_or_exit((nbytes), __FILE__, __LINE__)
