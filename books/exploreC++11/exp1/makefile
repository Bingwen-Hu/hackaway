CC=c++
OBJECTS=MoryNothing
OBJECTS+=readtext
OBJECTS+=readtext2

clean:
	rm -rf $(OBJECTS)
	rm -rf *.o

readtext: readtext.o
	$(CC) readtext.cpp -o $@

readtext2: readtext2.o
	$(CC) readtext2.cpp -o $@
