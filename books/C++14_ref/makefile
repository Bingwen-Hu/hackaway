CXX=g++
CFLAGS=-g -Wall --std=c++14

namespace:
	$(CXX) $(CFLAGS) namespace.cpp -o $@

var_scope:
	$(CXX) $(CFLAGS) var_scope.cpp -o $@

numeric_literals:
	$(CXX) $(CFLAGS)  numeric_literals.cpp -o $@

pointer_basic:
	$(CXX) $(CFLAGS) pointer_basic.cpp -o $@

nullpointer:
	$(CXX) $(CFLAGS) nullpointer.cpp -o $@

references:
	$(CXX) $(CFLAGS) references.cpp -o $@

rvalue:
	$(CXX) $(CFLAGS) rvalue.cpp -o $@

arrays: arrays.o
	$(CXX) $(CFLAGS) arrays.cpp -o $@
	
string: string.o
	$(CXX) $(CFLAGS) string.cpp -o $@

raw_string: raw_string.o
	$(CXX) $(CFLAGS) raw_string.cpp -o $@

string_encodings: string_encodings.o
	$(CXX) $(CFLAGS) string_encodings.cpp -o $@


forloop: forloop.o
	$(CXX) $(CFLAGS) forloop.cpp -o $@

auto: auto.o
	$(CXX) $(CFLAGS) auto.cpp -o $@

decltype: decltype.o
	$(CXX) $(CFLAGS) decltype.cpp -o $@

lambda: lambda.o
	$(CXX) $(CFLAGS) lambda.cpp -o $@

capture: capture.o
	$(CXX) $(CFLAGS) capture.cpp -o $@

clean:
	rm -rf namespace var_scope numeric_literals
	rm -rf pointer_basic nullpointer references
	rm -rf rvalue arrays string raw_string
	rm -rf string_encodings forloop auto
	rm -rf decltype lambda capture
	rm -rf *.o
