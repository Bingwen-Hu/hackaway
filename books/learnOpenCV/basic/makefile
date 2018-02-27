CXXFLAGS=-g -Wall -std=c++14 `pkg-config --cflags opencv`
LDLIBS=`pkg-config --libs opencv`
CXX=g++
OBJECTS=display_img display_video video_trackbar simple_transform
OBJECTS+=downsample canny_demo camera

display_img: display_img.o
	$(CXX) display_img.cpp $(CXXFLAGS) $(LDLIBS)  -o $@

display_video: display_video.o
	$(CXX) display_video.cpp $(CXXFLAGS) $(LDLIBS)  -o $@

video_trackbar: video_trackbar.o
	$(CXX) video_trackbar.cpp $(CXXFLAGS) $(LDLIBS)  -o $@

simple_transform: simple_transform.o
	$(CXX) simple_transform.cpp $(CXXFLAGS) $(LDLIBS)  -o $@

downsample: downsample.o
	$(CXX) downsample.cpp $(CXXFLAGS) $(LDLIBS)  -o $@

canny_demo: canny_demo.o
	$(CXX) canny_demo.cpp $(CXXFLAGS) $(LDLIBS)  -o $@

camera: camera.o
	$(CXX) camera.cpp $(CXXFLAGS) $(LDLIBS)  -o $@

clean:
	rm -rf *.o
	rm -rf $(OBJECTS)