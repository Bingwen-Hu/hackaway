# memory file
import cv2
import rtpose


img = '../imgs/fat.jpg'

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

for part in parts:
    cv2.circle(im, part, 4, (255, 128, 200), thickness=4)

cv2.imwrite("newimg.png", im)