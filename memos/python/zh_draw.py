from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

font = "/home/mory/.mory/font/msyh.ttc"

img = '/home/mory/hackaway/projects/emotion/Emotion.jpg'

im = Image.open(img)
x, y = 77, 99

font = ImageFont.truetype(font=font, size=30)
draw = ImageDraw.Draw(im)
draw.text((x, y), '平静 67.7%', (0, 255, 0), font=font)

im.save("test.jpg")