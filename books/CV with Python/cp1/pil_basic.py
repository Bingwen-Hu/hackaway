from PIL import Image

img = Image.open("E:/Mory/rose.jpg")
width, height = img.size
print("size of img: (width: {}, height: {})".format(width, height))
# grayscale
img_gray = img.convert("L")

# save it
img.save("E:/Mory/rose_save.png")

# create thumbnails, make the image smaller
img_gray.thumbnail((width/2, height/2))

# copy and paste region
# box is (left, upper, right, lower) responding to (x1, y1, x2, y2) on screen
box = (20, 100, 80, 160)
region = img.crop(box)

# rotate angle 180 and paste back
region = region.transpose(Image.ROTATE_180)
img.paste(region, box)


# resize
img_resize = img.resize((width*2, height*2), Image.BICUBIC)
img_rotate = img.rotate(45)
