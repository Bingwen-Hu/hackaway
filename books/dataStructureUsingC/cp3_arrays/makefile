CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=matrix merge array_and_pointer

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

matrix: matrix.o
	$(CC) matrix.c $(CFLAGS) -o $@

merge: merge.o
	$(CC) merge.c $(CFLAGS) -o $@

array_and_pointer: array_and_pointer.o
	$(CC) array_and_pointer.c $(CFLAGS) -o $@