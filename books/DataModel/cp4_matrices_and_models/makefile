CC=gcc
CFLAGS=-g -Wall --std=c99 `pkg-config --cflags apophenia`
LIBS=`pkg-config --libs apophenia`

multiplicationtable: multiplicationtable.o
	$(CC) multiplicationtable.c $(CFLAGS) $(LIBS) -o $@

vector_map: vector_map.o utilities.o
	$(CC) vector_map.c utilities.c $(CFLAGS) $(LIBS) -o $@
	
clean:
	rm -rf multiplicationtable *.o
	rm -rf vector_map
