<div align="center">

# Pytorch Pose Estimation

Pytorch implementation for [CVPR2017: Realtime Multi-Person Pose Estimation](https://arxiv.org/abs/1611.08050)

[English Version](README.md)

</div>

### 1. 项目状态
+ [x] 推断功能
+ [ ] 训练功能
+ [ ] 测试功能

### 2. 依赖库
+ eatable
+ pytorch >= 1.0
+ opencv
+ scipy
+ pycocotools

### 3. 安装
```sh
git clone https://github.com/siriusdemon/pytorch-rtpose
cd pytorch-rtpose
python setup.py install
```

### 4. 推断
推断非常简单，`rtpose`提供了`estimation`方法。

```py
import rtpose
img = 'path/to/img.png'
canvas, keypoint = rtpose.estimation(img)
```

### 5. 训练
目前还没有时间，不过欢迎 PRs。

### 6. Testing
目前还没有时间，不过欢迎 PRs。

### 7. Wishes
对于那些听说，看见或者使用这个仓库的人，我祝愿他们能够获得暂时的快乐与永恒不变的快乐。