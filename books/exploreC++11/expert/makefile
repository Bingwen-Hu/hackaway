CC=c++
OBJECTS=pointer_basic pointer2nothing list rvalue_explore
OBJECTS+=unique_ptr

clean:
	rm -rf $(OBJECTS)
	rm -rf *.o

pointer_basic: pointer_basic.o
	$(CC) pointer_basic.cpp -o $@

pointer2nothing: pointer2nothing.o
	$(CC) pointer2nothing.cpp -o $@

list: list.o
	$(CC) list.cpp -o $@
	
rvalue_explore: rvalue_explore.o
	$(CC) rvalue_explore.cpp -o $@
	
unique_ptr: unique_ptr.o
	$(CC) unique_ptr.cpp -o $@
