# LDLIBS=-L/usr/local/lib -lgsl -lgslcblas -lm
LDLIBS=`pkg-config --libs gsl`
CC=gcc
CFLAGS=-g -Wall -std=gnu11

gsl_cdf:
	$(CC)  cdf.c $(CFLAGS) $(LDLIBS) -o $@ 

erf:
	$(CC)  erf.c $(CFLAGS) -lm -o $@

gsl_test:
	$(CC) gsl_test.c $(CFLAGS) $(LDLIBS) -o $@

env:
	$(CC) env.c $(CFLAGS) -o $@


clean:
	rm -f gsl_cdf erf gsl_test env 
	rm -f *.o
