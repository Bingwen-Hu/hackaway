CC=g++
CXXFLAGS=-std=c++11 -O3 -I/usr/local/include/opencv4 -I/usr/local/include
LDLIBS=`pkg-config --libs opencv`


clean:
	rm -rf *.o picture

picture: picture.o idw.o
	echo "compile picture"
	${CC} picture.cpp idw.cpp idw.h ${CXXFLAGS} ${LDLIBS} -o $@ 