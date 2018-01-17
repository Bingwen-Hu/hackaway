CC=gcc
CFLAGS=-g -Wall --std=c99 `pkg-config --cflags apophenia`
LIBS=`pkg-config --libs apophenia`

multiplicationtable: multiplicationtable.o
	$(CC) multiplicationtable.c $(CFLAGS) $(LIBS) -o $@

	
clean:
	rm -rf multiplicationtable *.o
