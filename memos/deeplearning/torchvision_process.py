from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.ColorJitter(),
    transforms.Grayscale(),
    transforms.Resize((60, 100)),
    transforms.ToTensor(),
])

img = Image.open("E:/captcha-data/dwnews/test/asto.jpg")
img = transform(img)
print(img.size())