CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=stack lstack mstack application hanoi_tower

clean: 
	rm -rf *.o *.exe
	rm -rf $(OBJECTS)

stack: stack.o stack_test.o
	$(CC) stack.c stack_test.c $(CFLAGS) -o $@

lstack: lstack.o lstack_test.o
	$(CC) lstack.c lstack_test.c $(CFLAGS) -o $@

mstack: multistack.o multistack_test.o
	$(CC) multistack.c multistack_test.c $(CFLAGS) -o $@

application: applications.o stack.o
	$(CC) applications.c stack.c $(CFLAGS) -o $@

hanoi_tower: hanoi_tower.o
	$(CC) hanoi_tower.c $(CFLAGS) -o $@