CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=adjMatrix

clean:
	rm -rf *.o
	rm -rf $(OBJECTS)

adjMatrix: adjMatrix.o queue.o
	$(CC) adjMatrix.c queue.c $(CFLAGS) -o $@

