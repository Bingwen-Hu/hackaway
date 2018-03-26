CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=stack

clean:
	rm -rf *.o 
	rm -rf $(OBJECTS)

stack: stack.o
	$(CC) stack.c $(CFLAGS) -o $@