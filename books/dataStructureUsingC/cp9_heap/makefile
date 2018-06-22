CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=heap

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

heap: heap.o heap_test.o
	$(CC) heap.c heap_test.c $(CFLAGS) -o $@

