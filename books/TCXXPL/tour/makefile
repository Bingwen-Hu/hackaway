CC=g++
CXXFLAGS=-g -Wall -std=c++14
TARGETS=hello

hello: hello.o
	$(CC) hello.cpp $(CXXFLAGS) -o $@


clean:
	rm -rf $(TARGETS)
	rm -rf *.o
