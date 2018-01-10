LDLIBS=`pkg-config --libs glib-2.0`
CC=gcc
CFLAGS=-g -Wall -std=gnu11 `pkg-config --cflags glib-2.0`
objects=string_utilities.o

string_utilities:
	$(CC) string_utilities.c $(CFLAGS) -Dtest_ok_array  $(LDLIBS) -o $@

unicode: $(objects)

clean:
	rm -rf string_utilities
