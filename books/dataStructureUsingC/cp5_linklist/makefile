CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=linklist_test circularlist_test

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

linklist_test: linklist_test.o linklist.o
	$(CC) linklist_test.c linklist.c $(CFLAGS) -o $@


circularlist_test: circularlist.o circularlist_test.o
	$(CC) circularlist.c circularlist_test.c $(CFLAGS) -o $@
