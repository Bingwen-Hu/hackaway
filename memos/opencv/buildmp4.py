import cv2

import os
from os.path import basename

def sort_helper(path):
    name = basename(path)
    name = name.lstrip('dance')
    num = name.rstrip('_org.jpg')
    num = int(num)
    return num


img_dir = "/home/mory/Downloads/output/org"
video_path = "/home/mory/Downloads/org.avi"

fps = 30

images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
images = sorted(images, key=sort_helper)

num = len(images)

height, width = cv2.imread(images[0]).shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter(video_path, fourcc, fps, (width, height))


for img in images:
    frame = cv2.imread(img)
    videowriter.write(frame)

videowriter.release()
