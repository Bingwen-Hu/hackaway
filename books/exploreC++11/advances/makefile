CC=c++
OBJECTS=vector copy only_iterators


clean:
	rm -rf $(OBJECTS)
	rm -rf *.o

vector: vector.o
	$(CC) vector.cpp -o $@

copy: copy.o
	$(CC) copy.cpp -o $@

only_iterators: only_iterators.o
	$(CC) only_iterators.cpp -o $@
