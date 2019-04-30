# Face Landmark Detection
Convert Caffe model from this [repo](https://github.com/lsy17096535/face-landmark/blob/master/face_landmark.py)


## Getting Started
This repo use the following package, make sure you have already installed all of them.
+ dlib
+ opencv
+ torch

```bash
pip install dlib opencv-python torch
```
Note that `cmake` is needed to compile `dlib`. For `ubuntu`, you can install it as follow:
```bash
sudo apt install cmake
```

## Usage
We provide two function `show` and `detect`. Refer to `demo.py` for usage. 
```
python demo.py
```

## Results
![](./results/timg.jpeg)

## Installation
```
cd landmark
pip install .
```