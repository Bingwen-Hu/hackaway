CXX=g++
CFLAGS=-g -Wall --std=c++14


test_surpasser: test_surpasser.o surpasser.o
	$(CXX) $(CFLAGS) test_surpasser.cpp surpasser.cpp -o test_surpasser


clean:
	rm -rf test_surpasser
	rm -rf *.o
