CXXFLAGS=-g -Wall -std=c++14 `pkg-config --cflags opencv`
LDLIBS=`pkg-config --libs opencv`
CXX=g++
OBJECTS=display_img display_video

display_img: display_img.o
	$(CXX) display_img.cpp $(CXXFLAGS) $(LDLIBS)  -o $@

display_video: display_video.o
	$(CXX) display_video.cpp $(CXXFLAGS) $(LDLIBS)  -o $@

clean:
	rm -rf *.o
	rm -rf $(OBJECTS)