CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=insort

clean:
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

insort_test: insort.o insort_test.o
	$(CC) insort.c insort_test.c $(CFLAGS) -o $@