import cv2

import os
from os.path import basename

def sort_helper(path):
    name = basename(path)
    name = name.lstrip('dance')
    num = name.rstrip('_skel.jpg')
    num = int(num)
    return num


img_dir = "/home/mory/Downloads/output"
video_path = "/home/mory/Downloads/dance.avi"
# use for fps
org_video_path = "/home/mory/Downloads/dance.mp4"

org = cv2.VideoCapture(org_video_path)
fps = org.get(cv2.CAP_PROP_FPS)
imgsize = int(org.get(cv2.CAP_PROP_FRAME_WIDTH)), int(org.get(cv2.CAP_PROP_FRAME_HEIGHT))


images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
images = sorted(images, key=sort_helper)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter(video_path, fourcc, fps, imgsize)


for img in images:
    frame = cv2.imread(img)
    videowriter.write(frame)

videowriter.release()
org.release()