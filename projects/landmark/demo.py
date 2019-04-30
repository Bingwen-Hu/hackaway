import cv2

import landmark


def demo_show(image:str):
    img = landmark.show(image)
    return img

def demo_detect(image:str):
    img = cv2.imread(imgpath)
    result = landmark.detect(imgpath)
    points = result['landmark']
    x1, y1, x2, y2 = result['bbox']
    for i in range(0, len(points), 2):
        x = points[i] * (x2 - x1) + x1
        y = points[i+1] * (y2 - y1) + y1
        cv2.circle(img, (int(x), int(y)), 1, (128, 255, 255), 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    imgpath = 'images/timg.jpeg'
    img = demo_show(imgpath)
    cv2.imwrite("results/timg.jpeg", img)
    # demo_detect(imgpath)