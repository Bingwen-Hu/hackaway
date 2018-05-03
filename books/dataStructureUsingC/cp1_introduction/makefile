CC=gcc 
CFLAGS=-g -Wall -std=c11
OBJECTS=printf scanf generic_pointer mory_dec2bin

clean:
	rm -rf *.exe *.o
	rm -rf $(OBJECTS)

printf: printf.o
	$(CC) printf.c $(CFLAGS) -o $@


scanf: scanf.o
	$(CC) scanf.c $(CFLAGS) -o $@

generic_pointer: generic_pointer.o
	$(CC) generic_pointer.c $(CFLAGS) -o $@

mory_dec2bin: mory_dec2bin.o
	$(CC) mory_dec2bin.c $(CFLAGS) -o $@