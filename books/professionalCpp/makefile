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


std_array: std_array.o
	$(CXX) $(CFLAGS) std_array.cpp -o $@

functions: functions.o
	$(CXX) $(CFLAGS) functions.cpp -o $@

pass_by_ref: pass_by_ref.o
	$(CXX) $(CFLAGS) pass_by_ref.cpp -o $@

clean:
	rm -rf hello test_namespaces cast digits enums
	rm -rf std_array functions pass_by_ref
	rm -rf *.o
