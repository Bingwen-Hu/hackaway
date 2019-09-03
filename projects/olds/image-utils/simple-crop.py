import os
import os.path as osp
import cv2
from uuid import uuid1
from imutils import paths

width = 100
codelen = 5

def generate_xy(width, codelen):
    """Generate start and end indice of certain image width
    Args:
        width: image width
        piece: max number of piece to divide into
    Returns:
        list of pairs (start, end)        
    """
    # build the basic pieces
    unit = width // codelen
    base = [i*unit for i in range(codelen)]
    base.append(width)
    base = [(s, e) for (s, e) in zip(base, base[1:])]
    # TODO: merge basic pieces to generate larger pieces
    xys = base
    return xys


def crop_and_save(impath, xys, directory, codelen):
    """crop image into pieces and save them
    Args:
        impath: image path with code
        xys: (x, y) coordinates
        directory: directory to save pieces
        codelen: length of captcha code
    """
    im = cv2.imread(impath)
    codes = osp.basename(impath)[:codelen]
    for i, (s, e) in enumerate(xys):
        crop = im[:, s:e]
        path = osp.join(directory, f"{codes[i]}-{uuid1()}.png")
        print(path)
        cv2.imwrite(path, crop)



if __name__ == "__main__":
    image_directory = "/home/mory/Downloads/labledPicture"
    images = list(paths.list_images(image_directory))
    crops_xy = generate_xy(width, codelen)

    for i, impath in enumerate(images):
        crop_and_save(impath, crops_xy, "pieces", codelen)
        break
        
