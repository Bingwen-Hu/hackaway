# deepfree.datasets.coco

COCO is already a classic dataset format. And it's good enough. Let's build library on it.


## TODOs
+ [ ] Implement Heatmap and vectormap


## What may help
The `keypoints` in COCO annotations is a triple (x, y, v). While x and y means the position of the point, `v` means visible(2), invisible(1) or missing(0). The `num_keypoints` field only counts from visible and invisible points.

```py
MS COCO annotation order:
0: nose         1: l eye        2: r eye    3: l ear    4: r ear
5: l shoulder   6: r shoulder   7: l elbow  8: r elbow
9: l wrist      10: r wrist     11: l hip   12: r hip   13: l knee
14: r knee      15: l ankle     16: r ankle
```
Example 
![](./graphs/example.png)