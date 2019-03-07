CC=c++
OBJS=duplicate search_matrix replace_space

clean:
	rm -rf *.o
	rm -rf $(OBJS)

duplicate: duplicate.o
	$(CC) duplicate.cpp -o $@


search_matrix: search_matrix.o
	$(CC) search_matrix.cpp -o $@

replace_space: replace_space.o
	$(CC) replace_space.cpp -o $@
