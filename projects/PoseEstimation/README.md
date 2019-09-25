<div align="center">

# Pytorch Pose Estimation

[中文版](README_Zh.md)

</div>

### 1. Status
+ [x] inference
+ [ ] training
+ [ ] testing

### 2. Dependences
+ eatable
+ pytorch >= 1.0
+ opencv
+ scipy
+ pycocotools

### 3. Installation
```sh
git clone https://github.com/siriusdemon/pytorch-rtpose
cd pytorch-rtpose
python setup.py install
```

### 4. Inference
Very simple! 

```py
import rtpose
img = 'path/to/img.png'
canvas, keypoint = rtpose.estimation(img)
```

### 5. Training
I have no time to implement it right now. PRs are welcomed.

### 6. Testing
I have no time to implement it right now. PRs are welcomed.

### 7. Wishes
For anyone who hear, see or use this repo, I wish they gain temporary and everlasting happiness.