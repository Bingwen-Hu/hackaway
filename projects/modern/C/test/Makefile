# Compiler, flags, library
CC=gcc
CFLAGS=-g -Wall -O2
LIBS=

# target
TARGET=array

# dependences
OF_ARRAY = xmalloc.o test-array.o


clean: 
	rm -rf ${TARGET}
	rm -rf *.o

link:
	ln -sf ../src/* .

array: ${OF_ARRAY}
	${CC} ${OF_ARRAY} ${CFLAGS} -o $@

# convenient command for detect dependences
# gcc -MM file.c

test-array.o: test-array.c array.h xmalloc.h
xmalloc.o: xmalloc.c xmalloc.h