CC=gcc
CFLAGS=-g -Wall --std=c99 `pkg-config --cflags apophenia`
LIBS=`pkg-config --libs apophenia`

mysql_test: mysql.o utilities.o
	$(CC) mysql.c utilities.c $(CFLAGS) $(LIBS) -o $@ 


clean:
	rm -rf mysql_test *.o 
