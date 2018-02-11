CXX=g++
CFLAGS=-g -Wall --std=c++14

hello: hello.o
	$(CXX) $(CFLAGS) hello.cpp -o $@

test_namespaces: namespaces.o
	$(CXX) $(CFLAGS) namespaces.cpp test_namespaces.cpp -o $@


clean:
	rm -rf hello test_namespaces
	rm -rf *.o
