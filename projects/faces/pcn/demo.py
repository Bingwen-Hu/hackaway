import os
import cv2
import pcn

if __name__ == '__main__':
    # usage settings
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 pcn.py path/to/img")
        sys.exit()
    else:
        imgpath = sys.argv[1]
    # network detection
    img = cv2.imread(imgpath)
    faces = pcn.detect(img)
    # save image
    name = os.path.basename(imgpath)
    pcn.draw(img, faces)
    cv2.imwrite('result/ret_{}'.format(name), img)
