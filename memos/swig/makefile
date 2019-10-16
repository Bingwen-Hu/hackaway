CC=g++

clean:
	rm -rf test *.o *.gch 

test:
	${CC} test-vec.cpp vec.cpp vec.h -o $@

build:
	@echo build with swig
	swig -c++ -python vec.i
	g++ -fpic -c vec.h vec.cpp vec_wrap.cxx  -I/usr/include/python3.6
	g++ -shared vec.o vec_wrap.o -o _vec.so -lstdc++
	
	
