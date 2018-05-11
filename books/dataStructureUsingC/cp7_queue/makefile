CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=queue

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

queue: queue.o queue_test.o
	$(CC) queue.c queue_test.c $(CFLAGS) -o $@