# Compiler, flags, library
CC=gcc
CFLAGS=-g -Wall -O2

# target
TARGET=fantest

# dependences
OF_FANTEST = fantest.o main-fantest.o


clean: 
	rm -rf ${TARGET}
	rm -rf *.o

fantest: ${OF_FANTEST}
	${CC} ${OF_FANTEST} ${CFLAGS} -o $@


# convenient command for detect dependences
# gcc -MM file.c
fantest.o: fantest.c fantest.h
main-fantest.o: main-fantest.c fantest.h