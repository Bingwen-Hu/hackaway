CC=gcc 
CFLAGS=-g -Wall -std=c11
OBJECTS=printf scanf generic_pointer

clean:
	rm -rf *.exe *.o
	rm -rf $(OBJECTS)

printf: printf.o
	$(CC) printf.c $(CFLAGS) -o $@


scanf: scanf.o
	$(CC) scanf.c $(CFLAGS) -o $@

generic_pointer: generic_pointer.o
	$(CC) generic_pointer.c $(CFLAGS) -o $@