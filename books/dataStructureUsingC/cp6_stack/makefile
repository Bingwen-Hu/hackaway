CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=stack

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

stack: stack.o stack_test.o
	$(CC) stack.c stack_test.c $(CFLAGS) -o $@