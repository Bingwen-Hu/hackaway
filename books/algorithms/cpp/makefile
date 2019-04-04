CC=c++
CFLAGS=-g -Wall
OBJS=quicksort

clean:
	rm -rf *.o
	rm -rf $(OBJS)

quicksort: quicksort.o
	$(CC) quicksort.cpp $(CFLAGS) -o $@