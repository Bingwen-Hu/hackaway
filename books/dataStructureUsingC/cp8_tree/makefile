CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=tree

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

tree: tree.o tree_test.o
	$(CC) tree.c tree_test.c $(CFLAGS) -o $@