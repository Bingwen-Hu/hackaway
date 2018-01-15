CC=gcc
CFLAGS=-g -Wall --std=c99

birthday:
	$(CC) birthday.c $(CFLAGS) -lm -o $@

bdaystruct:
	$(CC) bdaystruct.c -lm -o $@

swap: utilities.o
	$(CC) swap.c utilities.c -o $@

squares:
	$(CC) squares.c  $(CFLAGS) -o $@

primes:
	$(CC) primes.c $(CFLAGS) -lm -o $@

strncpy:
	$(CC) strncpy.c $(CFLAGS) -o $@

astrncpy:
	$(CC) astrncpy.c $(CFLAGS) -o $@

clean:
	rm -rf birthday bdaystruct swap squares primes
	rm -rf strncpy *.o astrncpy
