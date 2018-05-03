CC=gcc 
CFLAGS=-g -Wall -std=c11
OBJECTS=first

clean:
	rm -rf *.exe *.o
	rm -rf $(OBJECTS)

first:first.o
	$(CC) first.c $(CFLAGS) -o $@