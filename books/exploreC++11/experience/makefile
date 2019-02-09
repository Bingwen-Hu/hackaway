CC=c++
CFLAGS=-Wall
OBJS=char_category locale function by_ref algorithms predicate
OBJS+=lambda


# in this makefile, latter script will appear later
clean:
	rm -rf *.o
	rm -rf $(OBJS)

char_category: char_category.o
	$(CC) char_category.cpp $(CFLAGS) -o $@


locale: locale.o
	$(CC) locale.cpp $(CFLAGS) -o $@

function: function.o
	$(CC) function.cpp $(CFLAGS) -o $@

by_ref: by_ref.o
	$(CC) by_ref.cpp $(CFLAGS) -o $@

algorithms: algorithms.o
	$(CC) algorithms.cpp $(CFLAGS) -o $@

predicate: predicate.o
	$(CC) predicate.cpp $(CFLAGS) -o $@
	
lambda: lambda.o
	$(CC) lambda.cpp $(CFLAGS) -o $@
