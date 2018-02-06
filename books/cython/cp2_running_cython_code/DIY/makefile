CFLAGS=`python3-config --cflags`
LDFLAGS=`python3-config --ldflags`

fib:
	cython fib.pyx
	gcc -fPIC -c fib.c ${CFLAGS}
	gcc fib.o -o fib.so -shared ${LDFLAGS}


clean:
	rm -f fib.[oc] fib.so
