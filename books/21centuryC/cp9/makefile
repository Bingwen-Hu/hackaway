LDLIBS=`pkg-config --libs glib-2.0`
CC=gcc
CFLAGS=-g -Wall -std=gnu11 -Dtest_ok_array `pkg-config --cflags glib-2.0`

string_utilities:
	$(CC) string_utilities.c $(CFLAGS) $(LDLIBS) -o $@

clean:
	rm -rf string_utilities
