# Compiler, flags, library
CC=gcc
CFLAGS=-g -Wall -std=c11 -pedantic


# target
TARGET=array fetch-line random sparse umfpack list
TARGET+=color wavelet

# dependences
OF_ARRAY = xmalloc.o test-array.o
OF_FETCH_LINE = fetch-line.o test-fetch-line.o
OF_RANDOM = test-random.o xmalloc.o
OF_SPARSE = test-sparse.o sparse.o xmalloc.o
OF_UMFPACK = test-umfpack.o xmalloc.o sparse.o
OF_LIST = test-list.o xmalloc.o list.o
OF_COLOR = test-color.o
OF_WAVELET = test-wavelet.o xmalloc.o wavelet.o

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

list: ${OF_LIST}
	${CC} ${OF_LIST} ${CFLAGS} -o $@

color: ${OF_COLOR}
	${CC} ${OF_COLOR} ${CFLAGS} -o $@
	
wavelet: ${OF_WAVELET}
	${CC} ${OF_WAVELET} ${CFLAGS} -o $@ -lm
# convenient command for detect dependences
# gcc -MM file.c
test-array.o: test-array.c array.h xmalloc.h
xmalloc.o: xmalloc.c xmalloc.h
test-fetch-line.o: test-fetch-line.c fetch-line.h
test-random.o: test-random.c array.h xmalloc.h random.h
sparse.o: sparse.c sparse.h
test-sparse.o: test-sparse.c sparse.h array.h xmalloc.h
test-umfpack.o: test-umfpack.c
test-list.o: test-list.c list.h
list.o: list.c list.h xmalloc.h
test-color.o: test-color.c termcolor.h
wavelet.o: wavelet.c wavelet.h
test-wavelet.o: test-wavelet.c array.h xmalloc.h wavelet.h