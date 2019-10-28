import numpy as np


EPS=1e-7
weight_ = 2
start_points = []
end_points = []


def distance(p, q):
    diff = p - q
    return np.linalg.norm(diff)
    

def getControlPointWeight(input):
    weightSum = 0
    weightMap = []

    for ep in end_points:
        temp = 1 / distance(ep, input) + EPS
        temp = np.power(temp, weight_)
        weightSum += temp
        weightMap.append(temp)
        
    weightMap = np.array(weightMap) / weightSum
    return weightMap

def getTransformPoint(input):
    x = y = 0
    weightMap = getControlPointWeight(input)
    for (sp, ep, w) in zip(start_points, end_points, weightMap):
        offset = (sp - ep) * w
        x = x + offset[1]
        y = y + offset[0]

    return np.array([x, y])


def transform(image):
    height, width, channel = image.shape
    output = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            p = getTransformPoint(np.array([x,y]))
            if (p[0]>=0 and p[1]>=0 and
                p[0])<=width and p[1]<=height):
                output[y, x, :] = image[y, x, :]
    return output


    
