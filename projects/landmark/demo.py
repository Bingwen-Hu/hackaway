import cv2

import landmark


def demo_show(image:str):
    img = landmark.show(image)
    return img

def demo_detect(image:str):
    img = cv2.imread(imgpath)
    results = landmark.detect(imgpath)
    for result in results:
        points = result['landmark']
        x1, y1, x2, y2 = result['bbox']
        for i in range(0, len(points), 2):
            x, y = int(points[i]), int(points[i+1])
            cv2.circle(img, (x, y), 1, (128, 255, 255), 2)
            cv2.putText(img, f"{i//2+1}", (x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=1, color=(0, 255, 0), thickness=1), 
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("results/timg.jpeg", img)

if __name__ == '__main__':
    imgpath = 'images/timg.jpeg'
    # img = demo_show(imgpath)
    # cv2.imwrite("results/timg.jpeg", img)
    demo_detect(imgpath)