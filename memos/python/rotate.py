from PIL import Image
import os



def image_rotate(path, newpath, angle=180):
    image = Image.open(path)
    new = image.rotate(180)
    new.save(newpath)


images = os.walk('data/train/')
target = "data/rotate"
for dir, _, files in images:
    for f in files:
        path = os.path.join(dir, f)
        code = f[:5]
        newpath = os.path.join(target, f"{code}_rotate.png")
        image_rotate(path, newpath)