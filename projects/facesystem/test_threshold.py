"a quick and dirty script to test threshold"

import os
import facesystem
import arcface
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

