# memory file
import cv2
import rtpose
import numpy as np


img = '../imgs/fat.jpg'
img = 'wei.jpg'
img = 'bress.jpg'

im = cv2.imread(img)
if im is None:
    raise "Image is None"


joint = rtpose.rtpose.KeyPointParams.joint
canvans, jsonfile = rtpose.estimation(img)
points = jsonfile['people'][0]

def get_XY(points):
    neck = points[joint.Neck * 3], points[joint.Neck * 3 + 1]
    lshoulder = points[joint.LShoulder * 3], points[joint.LShoulder * 3 + 1]
    rshoulder = points[joint.RShoulder * 3], points[joint.RShoulder * 3 + 1]
    lhip = points[joint.LHip * 3], points[joint.LHip * 3 + 1]    
    rhip = points[joint.RHip * 3], points[joint.RHip * 3 + 1]    
    lknee = points[joint.LKnee * 3], points[joint.LKnee * 3 + 1]
    rknee = points[joint.RKnee * 3], points[joint.RKnee * 3 + 1]
    lankle = points[joint.LAnkle * 3], points[joint.LAnkle * 3 + 1]
    rankle = points[joint.RAnkle * 3], points[joint.RAnkle * 3 + 1]
    return neck, lshoulder, rshoulder, lhip, rhip, lknee, rknee, lankle, rankle

parts = get_XY(points)

# for part in parts:
#     cv2.circle(im, part, 4, (255, 128, 200), thickness=4)

# cv2.imwrite("wei.jpg", im)

def midpoint(src, dst, ratio):
    """x, y in image format"""
    xspace = np.linspace(src[0], dst[0], 11)
    yspace = np.linspace(src[1], dst[1], 11)
    return [int(xspace[ratio]), int(yspace[ratio])]
    


def slim_Waist(parts, offset):
    neck, lshoulder, rshoulder, lhip, rhip = parts[0:5]
    vd = lhip[0] - rhip[0]
    hd = lshoulder[1] - lhip[1]
    dst_lwaist = midpoint(lshoulder, lhip, 8)
    dst_rwaist = midpoint(rshoulder, rhip, 8)
    src_lwaist = int(dst_lwaist[0] + vd*offset), dst_lwaist[1]+1
    src_rwaist = int(dst_rwaist[0] - vd*offset), dst_rwaist[1]+1

    src = [neck,
           src_lwaist, src_rwaist]
    dst = [neck,
           dst_lwaist, dst_rwaist]
    return src, dst

points = slim_Waist(parts, 0.3)

import pickle
with open("points.pkl", 'wb') as f:
    pickle.dump(points, f)

for s, d in zip(*points):
    cv2.circle(im, tuple(s), 10, (0, 255, 0), thickness=4)
    cv2.circle(im, tuple(d), 4, (0, 0, 255), thickness=4)
cv2.imwrite("newimg.png", im)

