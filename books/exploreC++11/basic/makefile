CC=c++
OBJECTS=readtext readtext2 boole int_arithmetic str_output
OBJECTS+=str_empty 

clean:
	rm -rf $(OBJECTS)
	rm -rf *.o

readtext: readtext.o
	$(CC) readtext.cpp -o $@

readtext2: readtext2.o
	$(CC) readtext2.cpp -o $@

boole: boole.o
	$(CC) boole.cpp -o $@

int_arithmetic: int_arithmetic.o
	$(CC) int_arithmetic.cpp -o $@

str_output: str_output.o
	$(CC) str_output.cpp -o $@

str_empty: str_empty.o
	$(CC) str_empty.cpp -o $@