CXX=g++
CXXFLAGS=-g -std=c++14 -Wall
ALL_TARGET=q_cmath q_complex

q_cmath: q_cmath.o
	${CXX} ${CXXFLAGS} q_cmath.cpp -o $@

q_complex: q_complex.o
	${CXX} ${CXXFLAGS} q_complex.cpp -o $@

clean:
	rm -rf *.o 
	rm -rf ${ALL_TARGET}