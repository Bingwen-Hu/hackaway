CC=c++
OBJECTS=vector copy only_iterators foreach increment error_decre


clean:
	rm -rf $(OBJECTS)
	rm -rf *.o

vector: vector.o
	$(CC) vector.cpp -o $@

copy: copy.o
	$(CC) copy.cpp -o $@

only_iterators: only_iterators.o
	$(CC) only_iterators.cpp -o $@

foreach: foreach.o
	$(CC) foreach.cpp -o $@

increment: increment.o
	$(CC) increment.cpp -o $@

error_decre: error_decre.o
	$(CC`) error_decre.cpp -o $@