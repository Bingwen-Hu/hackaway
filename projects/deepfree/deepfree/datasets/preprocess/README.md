# deepfree.datasets.preprocess

COCO datasets contains a lot of information including object class, bounding box, and mask. This module leverage this information to generate task-specific output such as mask.


## modules
+ augment
+ normalize: rescale values into [0, 1]
+ standardize: rescale data to have a mean of 0 and variance of 1.
+ whiten: 

