CC=g++
CXXFLAGS=-g -Wall -std=c++14
TARGETS=hello initstyle constant for vector Vector enum


clean:
	rm -rf $(TARGETS)
	rm -rf *.o


hello: hello.o
	$(CC) hello.cpp $(CXXFLAGS) -o $@

initstyle: initstyle.o
	$(CC) initstyle.cpp $(CXXFLAGS) -o $@

constant: constant.o
	$(CC) constant.cpp $(CXXFLAGS) -o $@

for: for.o
	$(CC) for.cpp $(CXXFLAGS) -o $@

vector: vector.o
	$(CC) vector.cpp $(CXXFLAGS) -o $@

Vector: Vector.o
	$(CC) Vector.cpp $(CXXFLAGS) -o $@

enum: enum.o
	$(CC) enum.cpp $(CXXFLAGS) -o $@