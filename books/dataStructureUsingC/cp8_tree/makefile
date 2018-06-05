CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=tree thread_tree

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

tree: tree.o tree_test.o
	$(CC) tree.c tree_test.c $(CFLAGS) -o $@


thread_tree: thread_tree.o thread_tree_test.o
	$(CC) thread_tree.c thread_tree_test.c $(CFLAGS) -o $@