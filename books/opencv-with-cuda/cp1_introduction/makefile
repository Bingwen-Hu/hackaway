OBJECTS=a.out
OBJECTS+=hello
OBJECTS+=adding

clean:
	rm -rf $(OBJECTS)


hello:
	nvcc hello.cu -o $@
adding: 	
	nvcc adding.cu -o $@
