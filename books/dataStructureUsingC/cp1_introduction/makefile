CC=gcc 
CFLAGS=-g -Wall -std=c11
OBJECTS=printf

clean:
	rm -rf *.exe *.o
	rm -rf $(OBJECTS)

printf:printf.o
	$(CC) printf.c $(CFLAGS) -o $@