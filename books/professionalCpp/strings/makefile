CXX=clang++
CXXFLAGS=-g -Wall --std=c++14


literals: literals.o
	$(CXX) $(CXXFLAGS) literals.cpp -o $@

raw: raw.o
	$(CXX) $(CXXFLAGS) raw.cpp -o $@

clean:
	rm -rf literals raw
	rm -rf *.o
