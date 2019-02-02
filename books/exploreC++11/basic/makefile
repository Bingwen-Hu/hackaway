CC=c++
OBJECTS=readtext readtext2 boole int_arithmetic str_output
OBJECTS+=str_empty io str_io error for format format_fill
OBJECTS+=alignment matrix

clean:
	rm -rf $(OBJECTS)
	rm -rf *.o

alignment: alignment.o
	$(CC) alignment.cpp -o $@

matrix: matrix.o
	$(CC) matrix.cpp -o $@

format_fill: format_fill.o
	$(CC) format_fill.cpp -o $@

format: format.o
	$(CC) format.cpp -o $@

for: for.o
	$(CC) for.cpp -o $@

io: io.o
	$(CC) io.cpp -o $@

error: error.o
	$(CC) error.cpp -o $@

str_io: str_io.o
	$(CC) str_io.cpp -o $@

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