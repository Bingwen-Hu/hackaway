from torchvision import transforms
from PIL import Image

img = Image.open('2fd9.jpg')
width, height = img.size
assert width == 104, height == 30
target = (224, 224)
newsize = (208, 60)

# resize = transforms.Resize(newsize)
# img_resize = resize(img)
# width, height = img_resize.size
resize = transforms.Resize((60, 208))
img_resize = resize(img)
width, height = img_resize.size
assert width == 208, height == 60 
img_resize.save("image_resize.jpg")

pad = transforms.Pad(((224-208)//2, (224-60)//2))
img_pad = pad(img_resize)
width, height = img_pad.size
# pad = transforms.Pad(((224-60)//2, (224-208)//2))
# img_pad = pad(img_resize)
# width, height = img_pad.size
assert width == 224, height == 224
img_pad.save("image_pad.jpg")