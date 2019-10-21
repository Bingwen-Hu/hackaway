# Compiler, flags, library
CC=gcc
CFLAGS=-g -Wall -std=c11 -pedantic
LIBS=-lm



# target
TARGET=hash-table

# dependences
OF_HASH_TABLE = test-hash-table.o hash-table.o


clean: 
	rm -rf ${TARGET}
	rm -rf *.o

hash-table: ${OF_HASH_TABLE}
	${CC} ${OF_HASH_TABLE} ${CFLAGS} -o $@ ${LIBS}


# convenient command for detect dependences
# gcc -MM file.c
test-hash-table.o: test-hash-table.c hash-table.h
hash-table.o: hash-table.c hash-table.h