# Emotion classifier


## Getting Started
This repo use the following package, make sure you have already installed all of them.
+ dlib
+ opencv
+ keras

```bash
pip install dlib opencv-python keras
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
![](./examples/emotion_test.jpg)


## Installation
```
cd emotion
python setup.py develop
```

## Reference
https://github.com/oarriaga/face_classification