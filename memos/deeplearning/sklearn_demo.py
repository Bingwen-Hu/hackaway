from sklearn.feature_extraction.image import extract_patches_2d
import cv2
import numpy as np



if __name__ == "__main__":
    imgpath = 'dog-cycle-car.png'
    img = cv2.imread(imgpath)
    clip = extract_patches_2d(img, patch_size=(100, 100), max_patches=6)
    for i, img in enumerate(clip):
        cv2.imwrite(f"crop-{i}.png", img)


