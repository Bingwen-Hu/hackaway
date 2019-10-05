# Compiler, flags, library
CC=gcc
CFLAGS=-g -Wall -O2 -std=c11 -pedantic
LIBS=

# target
TARGET=array fetch-line

# dependences
OF_ARRAY = xmalloc.o test-array.o
OF_FETCH_LINE = fetch-line.o test-fetch-line.o


clean: 
	rm -rf ${TARGET}
	rm -rf *.o

link:
	ln -sf ../src/* .

array: ${OF_ARRAY}
	${CC} ${OF_ARRAY} ${CFLAGS} -o $@

fetch-line: ${OF_FETCH_LINE}
	${CC} ${OF_FETCH_LINE} ${CFLAGS} -o $@

# convenient command for detect dependences
# gcc -MM file.c

test-array.o: test-array.c array.h xmalloc.h
xmalloc.o: xmalloc.c xmalloc.h
test-fetch-line.o: test-fetch-line.c fetch-line.h