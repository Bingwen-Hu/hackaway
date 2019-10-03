C=gcc
CFLAGS=-g -Wall -O2
OBJS=fantest

clean: 
	rm -rf ${OBJS}
	rm -rf *.o

fantest: fantest.o
	$C fantest.c main-fantest.c ${CFLAGS} -o $@
