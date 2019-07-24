# deepfree.datasets.coco

COCO is already a classic dataset format. And it's good enough. Let's build library on it.


## TODOs
+ [ ] Implement Heatmap and vectormap


## What may help
The `keypoints` in COCO annotations is a triple (x, y, v). While x and y means the position of the point, `v` means visible(2), invisible(1) or missing(0). The `num_keypoints` field only counts from visible and invisible points.