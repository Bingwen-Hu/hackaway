CC=gcc
CFLAGS=-g -Wall -std=c11
OBEJCTS=matrix

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

matrix: matrix.o
	$(CC) matrix.c $(CFLAGS) -o $@