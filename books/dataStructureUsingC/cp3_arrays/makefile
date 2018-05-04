CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=matrix merge

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

matrix: matrix.o
	$(CC) matrix.c $(CFLAGS) -o $@

merge: merge.o
	$(CC) merge.c $(CFLAGS) -o $@