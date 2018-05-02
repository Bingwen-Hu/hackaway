CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=fact

clean:
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

fact: fact.o fact_test.o
	$(CC) fact.c fact_test.c $(CFLAGS) -o $@