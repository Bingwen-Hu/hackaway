LDLIBS=-L/usr/local/lib -lgsl -lgslcblas -lm
CC=gcc
CFLAGS=-g -Wall -std=gnu11

gsl_cdf:
	$(CC)  cdf.c $(CFLAGS) $(LDLIBS) -o $@ 

erf:
	$(CC)  erf.c $(CFLAGS) -lm -o $@


clean:
	rm -f gsl_cdf erf
	rm -f *.o
