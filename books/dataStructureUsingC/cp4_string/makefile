CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=mstring_test replace

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

mstring_test: mstring_test.o mstring.o
	$(CC) mstring_test.c mstring.c $(CFLAGS) -o $@

replace: replace.o
	$(CC) replace.c $(CFLAGS) -o $@