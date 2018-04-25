CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=

clean:
	rm -rf *.o *.exe
	rm $(OBJECTS)

function_pointer: function_pointer.o
	$(CC) function_pointer.c $(CFLAGS) -o $@