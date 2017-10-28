P=erf
OBJECTS=erf.o
CFLAGS=-g -Wall -O3
LDLIBS=-lm
CC=clang

$(P): $(OBJECTS)

env:
	OBJECTS=env.o
	$(CC) $(CFLAGS) $(LDLIBS) -o $@ env.c
