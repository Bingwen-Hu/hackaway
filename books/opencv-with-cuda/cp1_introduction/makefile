OBJECTS=a.out
OBJECTS+=hello
OBJECTS+=adding
OBJECTS+=reference
OBJECTS+=findDevice
OBJECTS+=deviceProp


clean:
	rm -rf $(OBJECTS)


hello:
	nvcc hello.cu -o $@
adding: 	
	nvcc adding.cu -o $@
reference:
	nvcc reference.cu -o $@
findDevice:
	nvcc findDevice.cu -o $@
deviceProp:
	nvcc deviceProp.cu -o $@
