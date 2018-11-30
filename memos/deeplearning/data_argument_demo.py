from sklearn.feature_extraction.image import extract_patches_2d
import cv2
import numpy as np


def random_clip(imgpath):
    img = cv2.imread(imgpath)
    clip = extract_patches_2d(img, patch_size=(100, 100), max_patches=6)
    for i, img in enumerate(clip):
        cv2.imwrite(f"crop-{i}.png", img)


def test_crop(imgpath):
    target_width = 500
    target_height = 400
    img = cv2.imread(imgpath)
    crops = []
    h, w = img.shape[:2]
    # center and four corner
    coords = [
        [0, 0, target_width, target_height],
        [w - target_width, 0, w, target_height],
        [w - target_width, h - target_height, w, h],
        [0, h - target_height, target_width, h]]
    
    dW = int(0.5 * (w - target_width))
    dH = int(0.5 * (h - target_height))
    coords.append([dW, dH, w - dW, h - dH])

    for startX, startY, endX, endY in coords:
        crop = img[startY:endY, startX:endX]
        crop = cv2.resize(crop, (target_width, target_height),
            interpolation=cv2.INTER_AREA)
        crops.append(crop)

    mirrors = [cv2.flip(c, 1) for c in crops]
    crops.extend(mirrors)
    for i, crop in enumerate(crops):
        cv2.imwrite(f'{i}.png', crop)

if __name__ == "__main__":
    imgpath = 'dog-cycle-car.png'
    test_crop(imgpath)