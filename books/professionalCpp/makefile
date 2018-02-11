CXX=g++
CFLAGS=-g -Wall --std=c++14

hello: hello.o
	$(CXX) $(CFLAGS) hello.cpp -o $@

test_namespaces: namespaces.o
	$(CXX) $(CFLAGS) namespaces.cpp test_namespaces.cpp -o $@

cast: cast.o
	$(CXX) $(CFLAGS) cast.cpp -o $@

digits: digits.o
	$(CXX) $(CFLAGS) digits.cpp -o $@

enums: enums.o
	$(CXX) $(CFLAGS) enums.cpp -o $@

clean:
	rm -rf hello test_namespaces cast digits enums
	rm -rf *.o
