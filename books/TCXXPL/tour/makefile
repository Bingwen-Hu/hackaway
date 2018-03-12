CC=g++
CXXFLAGS=-g -Wall -std=c++14
TARGETS=hello initstyle

hello: hello.o
	$(CC) hello.cpp $(CXXFLAGS) -o $@

initstyle: initstyle.o
	$(CC) initstyle.cpp $(CXXFLAGS) -o $@

clean:
	rm -rf $(TARGETS)
	rm -rf *.o
