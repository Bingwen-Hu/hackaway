import json
import cv2

import rtpose



img = 'imgs/ski.jpg'
canvas, keypoint = rtpose.estimation(img)

cv2.imwrite('canvas.png', canvas)
with open('keypoint.json', 'w') as f:
    json.dump(keypoint, f)