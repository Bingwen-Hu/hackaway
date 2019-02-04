CC=c++
OBJECTS=vector copy only_iterators foreach increment error_decre reverse_vector
OBJECTS+=only_iterators2 print_bool as_bool

clean:
	rm -rf $(OBJECTS)
	rm -rf *.o

as_bool: as_bool.o
	$(CC) as_bool.cpp -o $@ 

print_bool: print_bool.o
	$(CC) print_bool.cpp -o $@

only_iterators2: only_iterators2.o
	$(CC) only_iterators2.cpp -o $@

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
	$(CC) error_decre.cpp -o $@

reverse_vector: reverse_vector.o
	$(CC) reverse_vector.cpp -o $@