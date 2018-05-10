CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=stack lstack

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

stack: stack.o stack_test.o
	$(CC) stack.c stack_test.c $(CFLAGS) -o $@

lstack: lstack.o lstack_test.o
	$(CC) lstack.c lstack_test.c $(CFLAGS) -o $@