CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=linklist_test

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

linklist_test: linklist_test.o linklist.o
	$(CC) linklist_test.c linklist.c $(CFLAGS) -o $@
