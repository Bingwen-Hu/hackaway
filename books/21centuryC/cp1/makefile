LDLIBS=-L/usr/local/lib -lgsl -lgslcblas -lm
CC=gcc
CFLAGS=-g -Wall -std=gnu11

gsl_cdf:
	$(CC)  cdf.c $(CFLAGS) $(LDLIBS) -o $@ 



clean:
	rm -f gsl_cdf
	rm -f *.o
