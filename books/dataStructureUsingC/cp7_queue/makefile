CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=queue lqueue

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

queue: queue.o queue_test.o
	$(CC) queue.c queue_test.c $(CFLAGS) -o $@

lqueue: lqueue.o lqueue_test.o
	$(CC) lqueue.c lqueue_test.c $(CFLAGS) -o $@