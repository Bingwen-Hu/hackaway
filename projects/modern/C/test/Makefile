# Compiler, flags, library
CC=gcc
CFLAGS=-g -Wall -std=c11 -pedantic


# target
TARGET=array fetch-line random sparse umfpack

# dependences
OF_ARRAY = xmalloc.o test-array.o
OF_FETCH_LINE = fetch-line.o test-fetch-line.o
OF_RANDOM = test-random.o xmalloc.o
OF_SPARSE = test-sparse.o sparse.o xmalloc.o
OF_UMFPACK = test-umfpack.o xmalloc.o sparse.o

clean: 
	rm -rf ${TARGET}
	rm -rf *.o

link:
	ln -sf ../src/* .

array: ${OF_ARRAY}
	${CC} ${OF_ARRAY} ${CFLAGS} -o $@

fetch-line: ${OF_FETCH_LINE}
	${CC} ${OF_FETCH_LINE} ${CFLAGS} -o $@

random: ${OF_RANDOM}
	${CC} ${OF_RANDOM} ${CFLAGS} -o $@

sparse: ${OF_SPARSE}
	${CC} ${OF_SPARSE} ${CFLAGS} -o $@

umfpack: ${OF_UMFPACK}
	${CC} ${OF_UMFPACK} ${CFLAGS} -lumfpack -o $@

# convenient command for detect dependences
# gcc -MM file.c
test-array.o: test-array.c array.h xmalloc.h
xmalloc.o: xmalloc.c xmalloc.h
test-fetch-line.o: test-fetch-line.c fetch-line.h
test-random.o: test-random.c array.h xmalloc.h random.h
sparse.o: sparse.c sparse.h
test-sparse.o: test-sparse.c sparse.h array.h xmalloc.h
test-umfpack.o: test-umfpack.c