# LDLIBS=-L/usr/local/lib -lgsl -lgslcblas -lm
LDLIBS=-lm
CC=gcc
CFLAGS=-g -Wall -std=gnu11

stddev:
	$(CC) stddev.c $(CFLAGS) $(LDLIBS) -o $@

stddev_debug:
	$(CC) stddev_debug.c $(CFLAGS) $(LDLIBS) -o $@
	
clean:
	rm -f stddev stddev_debug
