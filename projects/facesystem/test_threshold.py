"a quick and dirty script to test threshold"

import os
import facesystem
import arcface
import itertools
from imutils import paths

# there are sevaral directory that content image
def crop(dirs):
    for dir_ in dirs:
        files = list(paths.list_images(dir_))
        for file in files:
            _, label, filename = file.split(os.path.sep)
            crop = facesystem.face_detect(filename)
            os.makedirs(label, exist_ok=True)
            cv2.imwrite(f"{label}/{filename}")

def test_threshold(dirs):
    for dir_ in dirs:
        files = list(paths.list_images(dir_))
        print("test {}".format(dir_))
        for f in files:
            for f_ in files:
                f1 = arcface.featurize(f)
                f2 = arcface.featurize(f_)
                sim = arcface.compare(f1, f2)
                if sim < 0.4: 
                    print(sim, dir_, f, f_)

def test_threshold_cross(dirs):
    combines = itertools.combinations(dirs, 2)
    for d1, d2 in combines:
        print('test {} {}'.format(d1, d2))
        file1 = list(paths.list_files(d1))
        file2 = list(paths.list_files(d2))
        for f1 in file1:
            for f2 in file2:
                fe1 = arcface.featurize(f1)
                fe2 = arcface.featurize(f2)
                sim = arcface.compare(fe1, fe2)
                if sim > 0.4:
                    print(sim, f1, f2)
        

if __name__ == '__main__':
    test_threshold_cross(['mayun', 'obama', 'trump', 'xijinping'])