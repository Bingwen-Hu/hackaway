# Pose Estimation

可食用姿态估计

## 总括

### 网络结构
在论文中，作者采用了 VGG 的前10个卷积层输出的特征图作为输入，并且这10个卷积层是可以微调（finetune）的。随后，采用了6个不同的阶段（stage）来逐渐改进识别的效果，这种思想借鉴了 Pose Machine。我个人还觉得，这种做法很像 resnet 的残差结构。


在rtpose的每个stage中，都有两个分支，一个输出关系场（PAF），一个输出置信热力图（confidence map）。这里，我们叫它作CFM。

stage2至stage6的网络结构完全一致，这里我们采用动态生成的方式

这里的i代码第i个stage