CXX=g++
CXXFLAGS=-g -Wall -std=c++14

hello: hello.o
	$(CXX) $(CXXFLAGS) hello.cpp -o $@

test_namespaces: namespaces.o
	$(CXX) $(CXXFLAGS) namespaces.cpp test_namespaces.cpp -o $@

cast: cast.o
	$(CXX) $(CXXFLAGS) cast.cpp -o $@

digits: digits.o
	$(CXX) $(CXXFLAGS) digits.cpp -o $@

enums: enums.o
	$(CXX) $(CXXFLAGS) enums.cpp -o $@


std_array: std_array.o
	$(CXX) $(CXXFLAGS) std_array.cpp -o $@

functions: functions.o
	$(CXX) $(CXXFLAGS) functions.cpp -o $@

pass_by_ref: pass_by_ref.o
	$(CXX) $(CXXFLAGS) pass_by_ref.cpp -o $@

exceptions: exceptions.o
	$(CXX) $(CXXFLAGS) exceptions.cpp -o $@

const: const.o
	$(CXX) $(CXXFLAGS) const.cpp -o $@

test_surpasser: test_surpasser.o surpasser.o
	$(CXX) $(CXXFLAGS) test_surpasser.cpp surpasser.cpp -o $@

vector: vector.o
	$(CXX) $(CXXFLAGS) vector.cpp -o $@


clean:
	rm -rf hello test_namespaces cast digits enums
	rm -rf std_array functions pass_by_ref exceptions
	rm -rf const test_surpasser vector
	rm -rf *.o
