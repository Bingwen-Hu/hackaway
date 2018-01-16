CC=gcc
CFLAGS=-g -Wall --std=c99 `pkg-config --cflags apophenia`
LIBS=`pkg-config --libs apophenia`

mysql_test: mysql.o
	$(CC) mysql.c $(CFLAGS) $(LIBS) -o $@ 


clean:
	rm -rf mysql_test *.o
