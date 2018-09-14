import glob
from random import choice
import string

from PIL import Image, ImageFont, ImageDraw


def font_setup(fontpath, fontsize):
    """
    Args:
        fontpath: path to font.tff file
        fontsize: font size

    Returns:
        ImageFont.truetype font
    """
    return ImageFont.truetype(fontpath, fontsize)

def generate(mode, text, font, imgsize, position, bgcolor, fgcolor):
    """
    Args:
        mode: image mode, RGB, L, P, RGBA etc.
        text: text render on the image
        font: the font render the text, ImageFont.truetype object
        imgsize: image size
        position: where the text be placed
        bgcolor: background color
        fgcolor: foreground color

    Returns:
        A captcha image, PIL.Image object.
    """
    image = Image.new(mode, imgsize, bgcolor)
    draw_handle = ImageDraw.Draw(image)
    text = " ".join(text)
    draw_handle.text(xy=position, text=text, fill=fgcolor, font=font)
    return image



def generate_chars(chars, fonts, font_sizes, imgsize, init_xy, bgcolor, fgcolors):
    """在图片上生起多个字符的验证码"""
    image = Image.new('RGB', imgsize, bgcolor)
    draw_handle = ImageDraw.Draw(image)
    xy = init_xy
    
    for c in chars:
        font_size = choice(font_sizes)
        font = font_setup(choice(fonts), font_size)
        fgcolor = choice(fgcolors)
        offset = choice([5, 7, 8, 9, 10])
        draw_handle.text(xy=xy, text=c, fill=fgcolor, font=font)
        xy = xy[0]+font_size//2+offset, xy[1]
    return image

if __name__ == '__main__':
    chars = string.ascii_uppercase + string.digits
    fonts = glob.glob('E:/fonts/sogou/*')
    imgsize = (140, 44)
    init_xys = [(1, 1), (4, 1), (10, 1)]
    font_sizes = [18, 24, 32, 30]
    fgcolors = [(122, 64, 48), (116, 42, 31), (90, 86, 48), (92, 63, 33), (80, 60, 71)]
    num = 6
    
    char6 = [choice(chars) for _ in range(num)]
    img = generate_chars(char6, fonts, font_sizes, imgsize, init_xys[0], (170, 170, 170), fgcolors)
    img.save('gen.png')