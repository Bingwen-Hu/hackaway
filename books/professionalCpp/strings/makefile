CXX=g++
CFLAGS=-g -Wall --std=c++14


literals: literals.o
	$(CXX) $(CFLAGS) literals.cpp -o $@

raw: raw.o
	$(CXX) $(CFLAGS) raw.cpp -o $@

clean:
	rm -rf literals raw
	rm -rf *.o
