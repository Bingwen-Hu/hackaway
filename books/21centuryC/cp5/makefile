LDLIBS=-lm -ldl -lreadline
CFLAGS=-g -Wall -std=gnu11 -O3

dynamic:
	gcc dynamic.c $(CFLAGS) $(LDLIBS) -o $@

dynamic_test:
	gcc dynamic_test.c $(CFLAGS) $(LDLIBS) -o $@

clean:
	rm -f dynamic dynamic_test fn.*
