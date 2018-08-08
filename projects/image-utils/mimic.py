import glob
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
from uuid import uuid1

def merge_char(charimg, img, color):
    width, height = img.size
    char_w, char_h = charimg.size

    x = random.randint(char_h // 2, height - char_h)
    y = random.randint(0, width - char_w * 3)

    imgdata = np.array(img)
    chardata = np.array(charimg)

    for i in range(char_h):
        for j in range(char_w):
            rgb = chardata[i][j]
            if np.sum(rgb) > 10:
                org = imgdata[i+x][j+y]
                imgdata[i+x][j+y] = color
    newimg = Image.fromarray(imgdata)
    return newimg, (x, y), (char_h, char_w)


def generate_char(char, size, font):
    OFFSET = {
        50: (0, -9),
        45: (0, -7),
        55: (0, -10),
    }
    xy = OFFSET[size]
    charimg = Image.new('RGB', size=(size, size), color=(0, 0, 0))
    font = ImageFont.truetype(font=font, size=size)
    draw = ImageDraw.Draw(charimg)
    draw.text(xy, char, font=font, fill=(255, 255, 255))
    return charimg

if __name__ == '__main__':

    with open('common.txt', encoding='utf-8') as f:
        chars = f.read().strip()
    
    fonts = ['msyhbd', 'msyh', 'E:/fonts/wallow.ttf']
    background_glob_dir = "E:/captcha-data/img500/org/*.jpg"
    sizes = [45, 50, 55]
    background_imgs = glob.glob(background_glob_dir)
    angles = [45, 90, 180, 270, -45]
    fontcolors = [(0, 0, 0), (100, 220, 200), (50, 160, 255), (255, 130, 133), (200, 200, 100), (200, 130, 220), (100, 200, 255), (10, 40, 190), (130, 41, 41)]
    
    
    for i in range(4):
        angle = random.choice(angles)
        char = random.choice(chars)
        font = random.choice(fonts)
        size = random.choice(sizes)
        bg = random.choice(background_imgs)
        bg_img = Image.open(bg)
        color = random.choice(fontcolors)
        charimg = generate_char(char, size, font)
            
        charimg = charimg.rotate(angle, resample=Image.BICUBIC, expand=True)
        newimg, (x, y), (height, width) = merge_char(charimg, bg_img, color)
        
        crop = newimg.crop((y, x, y+width, x+height))
        # crop.save(f'E:/captcha-data/img500/genchars/{char}{uuid1()}.jpg')
        crop.save(f'./{char}.jpg')
