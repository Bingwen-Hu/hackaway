CC=c++
C=gcc
CFLAGS=-g -Wall
OBJS=duplicate search_matrix replace_space reversePrint
OBJS+=reversePrint2 rebuildBinaryTree prize

clean:
	rm -rf *.o
	rm -rf $(OBJS)

duplicate: duplicate.o
	$(CC) duplicate.cpp -o $@


search_matrix: search_matrix.o
	$(CC) search_matrix.cpp -o $@

replace_space: replace_space.o
	$(CC) replace_space.cpp -o $@

reversePrint: reversePrint.o
	$(C) reversePrint.c $(CFLAGS) -o $@

reversePrint2: reversePrint2.o
	$(CC) reversePrint2.cpp  -o $@

rebuildBinaryTree: rebuildBinaryTree.o
	$(CC) rebuildBinaryTree.cpp  -o $@

prize: 
	echo "prize is still incompleted!"	
