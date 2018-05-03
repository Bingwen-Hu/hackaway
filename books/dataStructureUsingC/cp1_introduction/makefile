CC=gcc 
CFLAGS=-g -Wall -std=c11
OBJECTS=printf scanf

clean:
	rm -rf *.exe *.o
	rm -rf $(OBJECTS)

printf: printf.o
	$(CC) printf.c $(CFLAGS) -o $@


scanf: scanf.o
	$(CC) scanf.c $(CFLAGS) -o $@