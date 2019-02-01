CC=c++
OBJECTS= readtext readtext2 boole int_arithmetic

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