# FaceNet: A Unified Embedding for Face Recognition and Clustering
[2015][[paper]](https://arxiv.org/pdf/1503.03832.pdf)


## 什么是人脸识别？
人脸识别是指从图片或者视频帧中识别（identify-这是谁）或者核验（verify-这是同一个人吗）人脸的一种技术。人脸识别系统通常是将图片或者视频帧中提取出人脸的某些特征，再和人脸数据库中的数据进行对比，从而能够判定图片中是否出现了某个人。
这项技术的精确度目前低于其他生理特征识别技术，比如指纹。
近来，人脸识别技术在商业标识（commerical identification）和市场营销（marketing）方面得到广泛应用。其他应用包括人机交互、视频监控、自动索引图片等。

abstract from [wikipedia](https://en.wikipedia.org/wiki/Facial_recognition_system)


## 以往的人脸识别模型及缺陷
在本论文之前的模型，都是使用有限个体（identity）的数据集来训练，比如使用含有4000个个体的数据集，通过训练神经网络模型对这4000人进行分类。训练完成后，将瓶颈层（通常是倒数第二层，也叫嵌入层embedding）作为人脸的通用表示，使用这个瓶颈层去生成不在训练数据集内的人脸的特征向量。

这种方法不仅不直接（indirectness），而且效率低（inefficiency）。训练集的样本直接影响瓶颈层的输出。为了学习到更加通用（generated）的特征，瓶颈层的维度通常是1000或以上。即把每张人脸映射为一个1000维以上的向量。以这种方式构造人脸库会占用大量的存储资源和计算资源。


## 论文提出的解决方案
直接使用卷积神经网络来训练嵌入层（embedding），并设计了一个损失函数（triplets loss）来训练和优化嵌入层。由于这种方法能够训练一个更通用的嵌入层，因而，仅仅使用**128维**的向量表示就已经取得当前最优。

  - 网络结构
  
  ![Model structure](https://github.com/weijiang2009/URunResearchPrototypeCode/blob/master/kejian/face-recognition/graphs/model_structure.png) 
  
  1. batch input layer：筛选三元组作为网络输入，见**难点**一节
  2. deep architecture: 论文探索了两种架构，参见[graphs](https://github.com/weijiang2009/URunResearchPrototypeCode/tree/master/kejian/face-recognition/graphs)文件夹的nn1和nn2
  3. l2 normalization: 限制输出向量，使得向量的模的平方为1
  4. embedding: 嵌入层，为l2 normalization的输出，维度为128
  5. triplet loss: 训练时用来优化嵌入层

  - 损失函数
  
  ![Triplet Loss](https://github.com/weijiang2009/URunResearchPrototypeCode/blob/master/kejian/face-recognition/graphs/triplet_loss.png)
  
  损失函数的优化目标为最小化anchor和正例positive的距离，同时最大化anchor与反例negative的距离。

  ![Loss Formula](https://github.com/weijiang2009/URunResearchPrototypeCode/blob/master/kejian/face-recognition/graphs/loss_formula.png)
  
  优化思路：使用数据集生成三元组，一个是anchor，一个是正例，一个是反例。优化的目标是anchor与正例的距离总是大于与反例的距离。
  
  随着网络的学习，大多数简单的三元组将无法对loss的优化有所贡献，因此会造成大量无效输入和延缓网络的收敛，所以需要有策略性地生成那些比较难优化的三元组（hard triplets）。**难点**一节讨论这个。


## 难点
triplet loss的训练比较困难（tricky），为了快速收敛，就需要选择难优化的三元组，也就是不满足公式1（见Loss Formula）的那些来作为模型的输入。

具体做法是，给定一个anchor，找出这个anchor所有正例中与anchor距离最大的那个，同理，找出所有负例中与anchor距离最小的那个，这样就得到了这个anchor最难的三元组（anchor, postive_max, negative_min）。

但是，对整个数据集中的每一个anchor计算positive_max和negative_min是不现实的。不仅因为计算量巨大，还因为这会引入大量误标注数据。

论文介绍了online triplets generation：每次生成一个mini-batch，其中，每个人（identity）包括40张脸作为正例，再随机加入负例。然后，在这个mini-batch中，对每个anchor计算对应的positive_max和negative_min。

然后，在实践中，选择最难的负样本训练时会导致较差的局部最小值（bad local minima），尤其是会造成模型的崩溃。（为什么呢？？）因此，选择了半困难的负样本（semi-hard）x_n，满足

![semi-hard](https://github.com/weijiang2009/URunResearchPrototypeCode/blob/master/kejian/face-recognition/graphs/semi_hard.png)

论文实验中的mini-batch大小为1800。



## 未来的提升点


## 相关应用&场景
- 人证核验
- 人脸识别
- 自动检引
- 人脸识别在防暴，人群监控

