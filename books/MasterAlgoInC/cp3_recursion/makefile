CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=

clean:
	rm -rf *.o *.exe
	rm $(OBJECTS)

fact: fact.o fact_test.o
	$(CC) fact.c fact_test.o $(CFLAGS) -o $@