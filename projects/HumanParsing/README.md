<div align="center">

# Pytorch Human Parsing

Pytorch implementation for [CVPR2017: Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)

<!-- [中文版](README_Zh.md) -->

</div>

### 1. Status
+ [x] inference
+ [ ] training
+ [ ] testing

### 2. Dependences
+ pytorch >= 1.0
+ pillow
+ opencv

### 3. Installation
```sh
git clone https://github.com/siriusdemon/pytorch-psp
cd pytorch-psp
python setup.py install
```

### 4. Inference
Very simple! 

```py
import psp
img = 'path/to/img.png'
prediction = psp.parse(img)
psp.save(prediction)
```

### 5. Training
I have no time to implement it right now. PRs are welcomed.

### 6. Testing
I have no time to implement it right now. PRs are welcomed.

### 7. Wishes
For anyone who hear, see or use this repo, I wish they gain temporary and everlasting happiness.