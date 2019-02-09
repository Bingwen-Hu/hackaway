CC=c++
CFLAGS=-Wall
OBJS=char_category locale function


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
